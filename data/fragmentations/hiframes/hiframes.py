#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from io import StringIO
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, Literal

try:
    from typing import Literal  # Python >= 3.8
except ImportError:
    from typing_extensions import Literal  # Python 3.7

from rdkit import Chem

from rdkit.Chem import rdchem



ExocyclicDoubleBondKind = Literal[
    "terminal_exocyclic_db",
    "inter_ring_linkage_exocyclic_db",
    "chain_exocyclic_db",
]

LinkerExocyclicDoubleBondKind = Literal[
    "terminal_linker_exocyclic_db",
    "chain_linker_exocyclic_db",
]


ReducedGraphCycleMode = Literal["single_node", "self_loop", "two_node"]
ReducedGraphNodeKind = Literal["original_endpoint", "cycle_surrogate"]


@dataclass(frozen=True)
class ReducedGraphNode:
    node_id: int
    kind: ReducedGraphNodeKind
    orig_atom_indices: Tuple[int, ...]


@dataclass(frozen=True)
class ReducedGraphEdge:
    edge_id: int
    src_node_id: int
    dst_node_id: int
    # Includes the endpoint atoms if stored.
    orig_atom_path: Tuple[int, ...] = ()
    # Bond indices along orig_atom_path, same order.
    orig_bond_indices: Tuple[int, ...] = ()


@dataclass
class ReducedGraph:
    nodes: List[ReducedGraphNode] = field(default_factory=list)
    edges: List[ReducedGraphEdge] = field(default_factory=list)


def _package_fragmentation_result(
    work: Chem.Mol,
    fragments: List[Fragment],
    reduced_graph: Optional[ReducedGraph],
    *,
    return_work_mol: bool,
    return_reduced_graph: bool,
) -> FragmentationResult:
    if return_work_mol and return_reduced_graph:
        if reduced_graph is None:
            raise RuntimeError("reduced_graph is None despite return_reduced_graph=True")
        return work, fragments, reduced_graph

    if return_work_mol:
        return work, fragments

    if return_reduced_graph:
        if reduced_graph is None:
            raise RuntimeError("reduced_graph is None despite return_reduced_graph=True")
        return fragments, reduced_graph

    return fragments


@dataclass(frozen=True)
class LinkerExocyclicDoubleBond:
    bond_idx: int
    linker_atom_idx: int        # atom currently classified as a linker atom (Stage 2 base linker set)
    exocyclic_atom_idx: int     # atom currently in pruned/chain set, double-bonded to linker_atom_idx
    exocyclic_heavy_degree: int
    kind: LinkerExocyclicDoubleBondKind


def _classify_linker_exocyclic_double_bonds(
    mol: Chem.Mol,
    *,
    heavy: Set[int],
    adj: Dict[int, List[int]],
    linker_atoms: Set[int],
    chain_candidate_atoms: Set[int],
) -> List[LinkerExocyclicDoubleBond]:
    """
    Classify linker-bound exocyclic double bonds after Stage 1 pruning.

    A bond is considered here when exactly one heavy atom endpoint is in `linker_atoms`
    (Stage 2 base linker atoms from `remaining`) and the other endpoint is in
    `chain_candidate_atoms` (typically the Stage-1 pruned atoms).

    This mirrors the ring exocyclic retention concept, but for linker atoms:
      - terminal: exocyclic atom heavy degree <= 1 (e.g., linker C=O oxygen)
      - chain:    exocyclic atom heavy degree > 1 (branching/non-terminal chain side)

    Only the exocyclic atom itself is eligible for promotion/retention with the linker;
    downstream chain atoms remain chains.
    """
    if not linker_atoms or not chain_candidate_atoms:
        return []

    heavy_deg = {i: len(adj.get(i, [])) for i in heavy}
    out: List[LinkerExocyclicDoubleBond] = []

    for b in mol.GetBonds():
        if b.GetBondType() != Chem.rdchem.BondType.DOUBLE:
            continue

        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()

        linker_idx = None
        exo_idx = None
        if a1 in linker_atoms and a2 in chain_candidate_atoms:
            linker_idx, exo_idx = a1, a2
        elif a2 in linker_atoms and a1 in chain_candidate_atoms:
            linker_idx, exo_idx = a2, a1
        else:
            continue

        exo_deg = heavy_deg.get(exo_idx, 0)
        kind: LinkerExocyclicDoubleBondKind = (
            "terminal_linker_exocyclic_db" if exo_deg <= 1 else "chain_linker_exocyclic_db"
        )

        out.append(
            LinkerExocyclicDoubleBond(
                bond_idx=b.GetIdx(),
                linker_atom_idx=linker_idx,
                exocyclic_atom_idx=exo_idx,
                exocyclic_heavy_degree=exo_deg,
                kind=kind,
            )
        )

    return out

@dataclass(frozen=True)
class RingExocyclicDoubleBond:
    bond_idx: int
    ring_atom_idx: int          # the ring atom on the double bond
    exocyclic_atom_idx: int     # the non-ring atom on the double bond
    exocyclic_heavy_degree: int
    kind: ExocyclicDoubleBondKind


def _build_ring_atom_adjacency(mol: Chem.Mol, ring_atoms_all: Set[int]) -> Dict[int, List[int]]:
    ring_adj = {i: [] for i in ring_atoms_all}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in ring_atoms_all and a2 in ring_atoms_all:
            ring_adj[a1].append(a2)
            ring_adj[a2].append(a1)
    return ring_adj

def _ring_system_ids_from_ring_atoms(ring_atoms_all: Set[int], ring_adj: Dict[int, List[int]]) -> Dict[int, int]:
    comp_id: Dict[int, int] = {}
    cid = 0
    seen: Set[int] = set()
    for start in ring_atoms_all:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp_id[start] = cid
        while stack:
            u = stack.pop()
            for v in ring_adj.get(u, []):
                if v in ring_atoms_all and v not in seen:
                    seen.add(v)
                    comp_id[v] = cid
                    stack.append(v)
        cid += 1
    return comp_id

def _classify_ring_exocyclic_double_bonds(
    mol: Chem.Mol,
    *,
    heavy: Set[int],
    ring_atoms_all: Set[int],
    adj: Dict[int, List[int]],
) -> List[RingExocyclicDoubleBond]:
    heavy_deg = {i: len(adj.get(i, [])) for i in heavy}

    ring_adj_true = _build_ring_atom_adjacency(mol, ring_atoms_all)
    ring_system_id = _ring_system_ids_from_ring_atoms(ring_atoms_all, ring_adj_true)

    # NEW: characterize non-ring connected components and which ring systems they touch.
    non_ring_atoms = {i for i in heavy if i not in ring_atoms_all}
    non_ring_comp_id: Dict[int, int] = {}
    non_ring_comp_touched_ring_systems: Dict[int, Set[int]] = {}

    seen_nr: Set[int] = set()
    cid = 0
    for start in non_ring_atoms:
        if start in seen_nr:
            continue
        stack = [start]
        seen_nr.add(start)
        comp_nodes: Set[int] = set()
        touched_systems: Set[int] = set()

        while stack:
            u = stack.pop()
            comp_nodes.add(u)

            for v in adj.get(u, []):
                if v in ring_atoms_all:
                    rsid = ring_system_id.get(v)
                    if rsid is not None:
                        touched_systems.add(rsid)
                elif v in non_ring_atoms and v not in seen_nr:
                    seen_nr.add(v)
                    stack.append(v)

        for u in comp_nodes:
            non_ring_comp_id[u] = cid
        non_ring_comp_touched_ring_systems[cid] = touched_systems
        cid += 1

    out: List[RingExocyclicDoubleBond] = []

    for b in mol.GetBonds():
        if b.GetBondType() != Chem.rdchem.BondType.DOUBLE:
            continue

        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()

        ring_idx = None
        exo_idx = None
        if a1 in ring_atoms_all and a2 in heavy and a2 not in ring_atoms_all:
            ring_idx, exo_idx = a1, a2
        elif a2 in ring_atoms_all and a1 in heavy and a1 not in ring_atoms_all:
            ring_idx, exo_idx = a2, a1
        else:
            continue

        exo_deg = heavy_deg.get(exo_idx, 0)

        if exo_deg <= 1:
            kind: ExocyclicDoubleBondKind = "terminal_exocyclic_db"
        else:
            # Immediate-ring-neighbor test (existing behavior)
            exo_ring_neighbors = [n for n in adj.get(exo_idx, []) if n in ring_atoms_all]
            touched_systems_direct = {
                ring_system_id[n] for n in exo_ring_neighbors if n in ring_system_id
            }

            # NEW: component-level bridge test
            comp_touched_systems: Set[int] = set()
            comp_id = non_ring_comp_id.get(exo_idx)
            if comp_id is not None:
                comp_touched_systems = non_ring_comp_touched_ring_systems.get(comp_id, set())

            if len(touched_systems_direct) > 1 or len(comp_touched_systems) > 1:
                kind = "inter_ring_linkage_exocyclic_db"
            else:
                kind = "chain_exocyclic_db"

        out.append(
            RingExocyclicDoubleBond(
                bond_idx=b.GetIdx(),
                ring_atom_idx=ring_idx,
                exocyclic_atom_idx=exo_idx,
                exocyclic_heavy_degree=exo_deg,
                kind=kind,
            )
        )

    return out

def _merge_chain_components_by_shared_attachment_with_attachments(
    mol: Chem.Mol,
    chain_comps: List[Set[int]],
    *,
    adj: Dict[int, List[int]],
    remaining: Set[int],
    merge_at_carbon_attachments: bool = False,
    merge_at_hetero_attachments: bool = False,
    atom_to_remaining_neighbors: Optional[Dict[int, Iterable[int]]] = None,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    """
    Like _merge_chain_components_by_shared_attachment(), but also returns the attachment-atom
    set for each (possibly merged) component.

    This avoids recomputing attachment sets twice in fragment_molecule():
      - once during merge decisions
      - again while emitting chain fragments / attachments

    atom_to_remaining_neighbors (optional):
      A cache mapping each chain atom -> iterable of neighboring atoms in `remaining`.
      If provided, we avoid repeatedly scanning adj[u] and set-membership on `remaining`.
    """
    if not chain_comps:
        return [], []

    comp_list = [set(c) for c in chain_comps]

    def _comp_attach_atoms(comp: Set[int]) -> Set[int]:
        out: Set[int] = set()
        if atom_to_remaining_neighbors is None:
            for u in comp:
                for v in adj.get(u, []):
                    if v in remaining:
                        out.add(v)
        else:
            for u in comp:
                for v in atom_to_remaining_neighbors.get(u, ()):
                    if v in remaining:
                        out.add(int(v))
        return out

    comp_attach_atoms: List[Set[int]] = [_comp_attach_atoms(c) for c in comp_list]

    # Legacy behavior unless explicitly enabled
    if len(comp_list) <= 1 or not (merge_at_carbon_attachments or merge_at_hetero_attachments):
        return comp_list, comp_attach_atoms

    # Map attachment atom -> list of component indices that touch it
    attach_to_comp_indices: Dict[int, List[int]] = {}
    for ci, attach_atoms in enumerate(comp_attach_atoms):
        for aidx in attach_atoms:
            attach_to_comp_indices.setdefault(aidx, []).append(ci)

    # Build a graph over component indices indicating "should merge"
    comp_graph: Dict[int, Set[int]] = {i: set() for i in range(len(comp_list))}
    any_edges = False

    for aidx, comp_idxs in attach_to_comp_indices.items():
        if len(comp_idxs) < 2:
            continue

        atomic_num = mol.GetAtomWithIdx(aidx).GetAtomicNum()
        is_carbon = (atomic_num == 6)

        should_merge_here = (
            (is_carbon and merge_at_carbon_attachments) or
            ((not is_carbon) and merge_at_hetero_attachments)
        )
        if not should_merge_here:
            continue

        # Star-connect all components touching this same attachment atom
        root = comp_idxs[0]
        for j in comp_idxs[1:]:
            if j == root:
                continue
            comp_graph[root].add(j)
            comp_graph[j].add(root)
            any_edges = True

    if not any_edges:
        return comp_list, comp_attach_atoms

    # Connected components over component-index graph => merged chain groups (+ merged attachment sets)
    merged: List[Set[int]] = []
    merged_attach: List[Set[int]] = []
    seen: Set[int] = set()

    for start in range(len(comp_list)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        merged_atoms: Set[int] = set()
        merged_att_atoms: Set[int] = set()

        while stack:
            i = stack.pop()
            merged_atoms.update(comp_list[i])
            merged_att_atoms.update(comp_attach_atoms[i])
            for j in comp_graph[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)

        merged.append(merged_atoms)
        merged_attach.append(merged_att_atoms)

    return merged, merged_attach

def _relax_noimplicit_on_cutpoints(parent: Chem.Mol, frag: Chem.Mol, keep_orig: set[int], orig_to_new: dict[int, int]) -> None:
    """
    After deleting atoms, some cutpoint atoms (often bracketed stereocarbons) retain NoImplicit=True.
    That prevents RDKit from adding implicit Hs during sanitization, so RDKit leaves radical electrons.
    We only relax NoImplicit on atoms that had at least one neighbor removed (cutpoints).
    """
    cutpoints: set[int] = set()
    for i in keep_orig:
        a = parent.GetAtomWithIdx(i)
        if any(n.GetIdx() not in keep_orig for n in a.GetNeighbors()):
            cutpoints.add(i)

    changed = False
    for orig_idx in cutpoints:
        new_idx = orig_to_new.get(orig_idx)
        if new_idx is None:
            continue
        a = frag.GetAtomWithIdx(new_idx)

        # Focus on neutral carbons (this is the common culprit for [CH] artifacts)
        if a.GetAtomicNum() == 6 and a.GetFormalCharge() == 0 and a.GetNoImplicit():
            a.SetNoImplicit(False)
            # If a bracketed stereo carbon carried explicit H, move it back to implicit bookkeeping
            if a.GetNumExplicitHs() > 0:
                a.SetNumExplicitHs(0)
            # If radicals were already assigned, clear them and let sanitize recompute valence
            if a.GetNumRadicalElectrons() > 0:
                a.SetNumRadicalElectrons(0)
            changed = True

    if changed:
        frag.UpdatePropertyCache(strict=False)




FragmentKind = Literal["chain", "linker", "ring", "ring_core"]


@dataclass(frozen=True)
class AttachmentPoint:
    """
    Represents a cut bond from a fragment atom to some neighbor atom in the original molecule.

    - fragment_atom_idx: atom index within the fragment mol (on the fragment/core side)
    - fragment_atom_orig_idx: corresponding atom index in the original mol
    - neighbor_orig_idx: the neighbor atom index in the original mol (often the scaffold/ring-side atom)
    - bond_type: RDKit bond type between them in the original mol
    - neighbor_frag_atom_idx: atom index within the fragment mol if the neighbor atom
      was explicitly included in this fragment (otherwise None)
    """
    fragment_atom_idx: int
    fragment_atom_orig_idx: int
    neighbor_orig_idx: int
    bond_type: Chem.rdchem.BondType
    neighbor_frag_atom_idx: Optional[int] = None

@dataclass
class Fragment:
    kind: FragmentKind
    mol: Chem.Mol
    orig_atom_indices: Tuple[int, ...]
    atom_map_orig_to_frag: Dict[int, int]
    attachments: List[AttachmentPoint] = field(default_factory=list)

    # If enabled for chain/linker fragments, these are the "endcap" / attachment atoms
    included_attachment_atom_orig_indices: Tuple[int, ...] = ()
    com_cache: dict = field(default_factory=dict, repr=False)
    dist_cache: dict = field(default_factory=dict, repr=False)

    # No Kebule output
    # def smiles(self, isomeric: bool = True) -> str:
    #     return Chem.MolToSmiles(self.mol, isomericSmiles=isomeric)
    def smiles(self, isomeric: bool = True, kekule: bool = True, canonical: bool = True) -> str:
        if not kekule:
            return Chem.MolToSmiles(
                self.mol,
                isomericSmiles=isomeric,
                kekuleSmiles=False,
                canonical=canonical,
            )

        # Prefer RDKit's built-in Kekulé SMILES generation; fall back to rescue if needed.
        try:
            return Chem.MolToSmiles(
                self.mol,
                isomericSmiles=isomeric,
                kekuleSmiles=True,
                canonical=canonical,
            )
        except Exception:
            return _kekule_smiles_rescue(self.mol, isomeric=isomeric, canonical=canonical)


    def __repr__(self) -> str:
        return (
            f"Fragment(kind={self.kind!r}, "
            f"n_atoms={self.mol.GetNumAtoms()}, "
            f"attachments={len(self.attachments)}, "
            f"smiles={self.smiles()!r})"
        )

    def center_of_mass(
        self,
        work_mol: Chem.Mol,
        *,
        include_attachment_atoms: bool = True,
        confId: int = 0,
        mass_weighted: bool = True,
        include_implicit_h_mass: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Computes COM in the coordinate frame of `work_mol` (the same mol used for fragmentation).

        include_attachment_atoms=True  -> use all atoms in this fragment (orig_atom_indices)
        include_attachment_atoms=False -> exclude included_attachment_atom_orig_indices
        """
        key = (confId, include_attachment_atoms, mass_weighted, include_implicit_h_mass)
        if key in self.com_cache:
            return self.com_cache[key]

        atoms = set(self.orig_atom_indices)
        if not include_attachment_atoms:
            atoms -= set(self.included_attachment_atom_orig_indices)

        if not atoms:
            raise ValueError(
                f"Fragment {self.kind} has no atoms under include_attachment_atoms={include_attachment_atoms}."
            )

        com = center_of_mass(
            work_mol,
            atom_indices=sorted(atoms),
            confId=confId,
            mass_weighted=mass_weighted,
            include_implicit_h_mass=include_implicit_h_mass,
        )

        self.com_cache[key] = com
        return com

    def distance_to_molecule_com(
        self,
        work_mol: Chem.Mol,
        *,
        include_attachment_atoms: bool = True,
        confId: int = 0,
        mass_weighted: bool = True,
        include_implicit_h_mass: bool = True,
    ) -> float:
        # molecule COM depends on confId + mass settings; fragment COM also depends on include_attachment_atoms
        key = (confId, include_attachment_atoms, mass_weighted, include_implicit_h_mass)
        if key in self.dist_cache:
            return self.dist_cache[key]

        com_mol = center_of_mass(
            work_mol,
            confId=confId,
            mass_weighted=mass_weighted,
            include_implicit_h_mass=include_implicit_h_mass,
        )
        com_fr = self.center_of_mass(
            work_mol,
            include_attachment_atoms=include_attachment_atoms,
            confId=confId,
            mass_weighted=mass_weighted,
            include_implicit_h_mass=include_implicit_h_mass,
        )
        d = euclidean_distance(com_fr, com_mol)
        self.dist_cache[key] = d
        return d

FragmentationResult = Union[
    List[Fragment],
    Tuple[Chem.Mol, List[Fragment]],
    Tuple[List[Fragment], ReducedGraph],
    Tuple[Chem.Mol, List[Fragment], ReducedGraph],
]

def mol_to_3dmol_block(mol: Chem.Mol, *, confId: int = 0, kekulize: bool = False) -> str:
    """
    Returns an SDF/MolBlock string suitable for py3Dmol. Avoids kekulization by default.
    Uses MolToMolBlock if available; falls back to SDWriter otherwise.
    """
    m = Chem.Mol(mol)  # copy

    # Preferred (newer RDKit): MolToMolBlock(..., kekulize=...)
    try:
        return Chem.MolToMolBlock(m, confId=confId, kekulize=kekulize)
    except TypeError:
        # Fallback: SDWriter supports SetKekulize in many RDKit builds
        sio = StringIO()
        w = Chem.SDWriter(sio)
        try:
            w.SetKekulize(bool(kekulize))
        except Exception:
            pass
        w.write(m, confId=confId)
        w.flush()
        w.close()
        return sio.getvalue()


def ensure_3d_conformer(
    mol: Chem.Mol,
    *,
    confId: int = 0,
    seed: int = 0,
    minimize: bool = True,
    maxIters: int = 200,
) -> int:
    """
    Ensures `mol` has a 3D conformer with id `confId`.
    - If conformer exists: returns confId
    - If none exist: embeds one (ETKDGv3). Optionally minimizes with MMFF94s, else UFF.
    """
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    if conf_ids:
        if confId not in conf_ids:
            raise ValueError(f"Requested confId={confId} not found; available={conf_ids}")
        return confId

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True

    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError("3D embedding failed (ETKDG).")

    if minimize:
        # Prefer MMFF for drug-like organics; fall back to UFF if needed.
        if AllChem.MMFFHasAllMoleculeParams(mol):
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=int(maxIters))
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=int(maxIters))

    return cid


def center_of_mass(
    mol: Chem.Mol,
    *,
    atom_indices: Optional[Sequence[int]] = None,
    confId: int = 0,
    mass_weighted: bool = True,
    include_implicit_h_mass: bool = True,
) -> Tuple[float, float, float]:
    """
    Center of mass (or centroid if mass_weighted=False) from coordinates in mol's conformer `confId`.

    If include_implicit_h_mass=True and mass_weighted=True, we add implicit-H mass onto the heavy atom
    position (since implicit H has no coordinates).
    """
    conf = mol.GetConformer(confId)  # raises if missing

    if atom_indices is None:
        atom_indices = list(range(mol.GetNumAtoms()))
    if not atom_indices:
        raise ValueError("atom_indices is empty; cannot compute COM.")

    pt = Chem.GetPeriodicTable()
    H_mass = float(pt.GetAtomicWeight(1))

    sum_x = sum_y = sum_z = 0.0
    sum_w = 0.0

    for idx in atom_indices:
        a = mol.GetAtomWithIdx(int(idx))
        p = conf.GetAtomPosition(int(idx))

        if mass_weighted:
            w = float(a.GetMass())
            if include_implicit_h_mass:
                # TotalNumHs includes implicit H (and explicit if present).
                nH = int(a.GetTotalNumHs())
                w += nH * H_mass
        else:
            w = 1.0

        sum_x += w * float(p.x)
        sum_y += w * float(p.y)
        sum_z += w * float(p.z)
        sum_w += w

    if sum_w == 0.0:
        raise ValueError("Total weight is zero; cannot compute COM.")

    return (sum_x / sum_w, sum_y / sum_w, sum_z / sum_w)


def euclidean_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _heavy_atom_indices(mol: Chem.Mol) -> List[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]


def _build_heavy_adjacency(mol: Chem.Mol) -> Dict[int, List[int]]:
    """
    Heavy-atom adjacency list keyed by atom idx.
    """
    adj: Dict[int, List[int]] = {i: [] for i in _heavy_atom_indices(mol)}
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if mol.GetAtomWithIdx(a1).GetAtomicNum() == 1:
            continue
        if mol.GetAtomWithIdx(a2).GetAtomicNum() == 1:
            continue
        adj[a1].append(a2)
        adj[a2].append(a1)
    return adj

def _group_bond_linker_edges_by_shared_attachment(
    mol: Chem.Mol,
    bond_linker_edges: Set[Tuple[int, int]],
    *,
    merge_at_carbon_attachments: bool = False,
    merge_at_hetero_attachments: bool = False,
) -> List[Set[int]]:
    """
    Optionally group bond-linker edges into larger linker fragments when multiple
    bond-linker edges share the same attachment atom.

    Example use case:
      A promoted inter-ring exocyclic carbon (retained by preserve_ring_linkage_db)
      may connect by single bonds to multiple ring atoms, yielding several separate
      2-atom bond-linker fragments by default. This helper can merge them into one
      branched linker fragment.

    Returns:
      List of atom sets (each atom set is the union of one or more bond-linker edges).
      Defaults preserve legacy behavior: one atom-set per edge.
    """
    if not bond_linker_edges:
        return []

    edge_list = sorted(tuple(sorted(e)) for e in bond_linker_edges)

    # Legacy behavior unless explicitly enabled
    if not (merge_at_carbon_attachments or merge_at_hetero_attachments):
        return [set(e) for e in edge_list]

    # Map atoms -> incident bond-linker edge indices
    atom_to_edge_idxs: Dict[int, List[int]] = {}
    for ei, (a1, a2) in enumerate(edge_list):
        atom_to_edge_idxs.setdefault(a1, []).append(ei)
        atom_to_edge_idxs.setdefault(a2, []).append(ei)

    # Build graph over edge indices: connect edges that share an allowed attachment atom
    edge_graph: Dict[int, Set[int]] = {i: set() for i in range(len(edge_list))}
    any_links = False

    for aidx, eidxs in atom_to_edge_idxs.items():
        if len(eidxs) < 2:
            continue

        atomic_num = mol.GetAtomWithIdx(aidx).GetAtomicNum()
        is_carbon = (atomic_num == 6)

        should_merge_here = (
            (is_carbon and merge_at_carbon_attachments) or
            ((not is_carbon) and merge_at_hetero_attachments)
        )
        if not should_merge_here:
            continue

        root = eidxs[0]
        for j in eidxs[1:]:
            if j == root:
                continue
            edge_graph[root].add(j)
            edge_graph[j].add(root)
            any_links = True

    if not any_links:
        return [set(e) for e in edge_list]

    # Connected components over edge-index graph => grouped bond-linker fragments
    grouped_atom_sets: List[Set[int]] = []
    seen: Set[int] = set()

    for start in range(len(edge_list)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        atom_union: Set[int] = set()

        while stack:
            ei = stack.pop()
            atom_union.update(edge_list[ei])
            for ej in edge_graph[ei]:
                if ej not in seen:
                    seen.add(ej)
                    stack.append(ej)

        grouped_atom_sets.append(atom_union)

    return grouped_atom_sets

def _connected_components(nodes: Set[int], adj: Dict[int, List[int]]) -> List[Set[int]]:
    comps: List[Set[int]] = []
    seen: Set[int] = set()
    for start in list(nodes):
        if start in seen:
            continue
        stack = [start]
        comp: Set[int] = set()
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in adj.get(u, []):
                if v in nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps

def _can_kekulize(m: Chem.Mol) -> bool:
    test = Chem.Mol(m)
    try:
        Chem.Kekulize(test, clearAromaticFlags=True)
        return True
    except Exception:
        return False


def _sanitize_no_kek(m: Chem.Mol) -> None:
    m.UpdatePropertyCache(strict=False)
    ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    err = Chem.SanitizeMol(m, sanitizeOps=ops, catchErrors=True)
    if err != Chem.SanitizeFlags.SANITIZE_NONE:
        # fall back to full sanitize to surface real valence/charge issues
        Chem.SanitizeMol(m)


def _kekule_smiles_rescue(
    mol: Chem.Mol, *, isomeric: bool, canonical: bool
) -> str:
    """
    Always try to return kekuleSmiles=True.
    If direct kekulization fails, try tautomer enumeration as a rescue (covers many 'non-pyrrolic' azole cases).
    """
    base = Chem.Mol(mol)
    _sanitize_no_kek(base)

    if _can_kekulize(base):
        Chem.Kekulize(base, clearAromaticFlags=True)
        return Chem.MolToSmiles(base, isomericSmiles=isomeric, kekuleSmiles=True, canonical=canonical)

    # Rescue 1: try moving aromatic H / charges via tautomer enumeration
    # (very common for 5-member N-rich rings: imidazole/triazole/tetrazole families, fused variants)
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize  # part of RDKit in most installs

        te = rdMolStandardize.TautomerEnumerator()
        for t in te.Enumerate(base):
            tt = Chem.Mol(t)
            _sanitize_no_kek(tt)
            if _can_kekulize(tt):
                Chem.Kekulize(tt, clearAromaticFlags=True)
                return Chem.MolToSmiles(
                    tt, isomericSmiles=isomeric, kekuleSmiles=True, canonical=canonical
                )
    except Exception:
        pass

    # Rescue 2: last-chance roundtrip (sometimes re-perceives aromaticity more sanely)
    try:
        arom = Chem.MolToSmiles(base, isomericSmiles=isomeric, kekuleSmiles=False, canonical=canonical)
        m2 = Chem.MolFromSmiles(arom)
        if m2 is not None:
            _sanitize_no_kek(m2)
            if _can_kekulize(m2):
                Chem.Kekulize(m2, clearAromaticFlags=True)
                return Chem.MolToSmiles(
                    m2, isomericSmiles=isomeric, kekuleSmiles=True, canonical=canonical
                )
    except Exception:
        pass

    # If nothing works, you can either:
    # - raise (strict mode), or
    # - return aromatic SMILES as a fallback.
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric, kekuleSmiles=False, canonical=canonical)


def _extract_submol_by_bondcuts(
    mol: Chem.Mol,
    atom_indices: Sequence[int],
    *,
    add_explicit_hs: bool = False,
    set_atommap_to_orig_idx: bool = False,
    force_non_aromatic_atoms: Optional[Set[int]] = None,
) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Extract a fragment by cutting boundary bonds with RDKit's FragmentOnBonds.

    This is more robust than deleting atoms in-place for many heteroaromatic cases because RDKit's
    bond fragmentation inserts dummy atoms to preserve valence bookkeeping at the cut site.

    Safety guard: we only use this path when ALL boundary bonds are non-ring. If a fragment boundary
    would require cutting bonds inside a ring system, we fall back to the delete-atoms extractor.
    """
    keep = set(atom_indices)
    if not keep:
        raise ValueError("atom_indices is empty")

    force_non_aromatic_atoms = force_non_aromatic_atoms or set()

    # Identify bonds crossing the keep/non-keep boundary.
    cut_bond_ids: List[int] = []
    for b in mol.GetBonds():
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        if (a1 in keep) ^ (a2 in keep):
            if b.IsInRing():
                raise ValueError("boundary intersects ring bond; fallback to delete-atoms extraction")
            cut_bond_ids.append(b.GetIdx())

    if not cut_bond_ids:
        raise ValueError("no boundary bonds to cut; fallback to delete-atoms extraction")

    # Work on a copy to avoid mutating upstream molecules/properties.
    work = Chem.Mol(mol)

    fragged = Chem.FragmentOnBonds(work, cut_bond_ids, addDummies=True)

    # Select the component with maximal overlap with keep.
    frags_atoms = Chem.GetMolFrags(fragged, asMols=False, sanitizeFrags=False)
    overlaps = [len(set(tup).intersection(keep)) for tup in frags_atoms]
    frag_idx = int(max(range(len(overlaps)), key=lambda i: overlaps[i]))
    if overlaps[frag_idx] == 0:
        raise ValueError("could not locate fragment component after bond cutting")

    frags_mols = Chem.GetMolFrags(fragged, asMols=True, sanitizeFrags=False)
    frag = frags_mols[frag_idx]

    # Replace dummy atoms at cut points with H and then make them implicit.
    for a in frag.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(1)
            a.SetIsotope(0)
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
            a.SetNoImplicit(False)
            try:
                a.SetNumExplicitHs(0)
            except Exception:
                pass

    # _relax_noimplicit_on_cutpoints(mol, frag, keep, {})  # no mapping yet, but we just want to relax cutpoints before sanitization

    frag = Chem.RemoveAllHs(frag)

    # Build mapping orig_idx -> new_idx using stored _orig_idx property
    orig_to_new: Dict[int, int] = {}
    for new_idx, a in enumerate(frag.GetAtoms()):
        if a.HasProp("_orig_idx"):
            orig_idx = int(a.GetIntProp("_orig_idx"))
        else:
            orig_idx = int(a.GetIdx())
            a.SetIntProp("_orig_idx", int(orig_idx))
        orig_to_new[orig_idx] = new_idx
        if set_atommap_to_orig_idx:
            a.SetAtomMapNum(orig_idx + 1)

    _relax_noimplicit_on_cutpoints(mol, frag, keep, orig_to_new)

    # Optionally force certain atoms/bonds non-aromatic (by original indices)
    if force_non_aromatic_atoms:
        for orig_idx in force_non_aromatic_atoms:
            if orig_idx in orig_to_new:
                a = frag.GetAtomWithIdx(orig_to_new[orig_idx])
                a.SetIsAromatic(False)
        for b in frag.GetBonds():
            if (b.GetBeginAtom().GetIsAromatic() is False) or (b.GetEndAtom().GetIsAromatic() is False):
                b.SetIsAromatic(False)
                if b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    b.SetBondType(Chem.rdchem.BondType.SINGLE)

    frag.UpdatePropertyCache(strict=False)

    # Sanitize, then ensure the fragment object itself is stored in Kekulé form.
    Chem.SanitizeMol(frag)
    Chem.Kekulize(frag, clearAromaticFlags=True)

    Chem.AssignStereochemistry(frag, force=True, cleanIt=True)

    if add_explicit_hs:
        frag = Chem.AddHs(frag, addCoords=False)

    frag.UpdatePropertyCache(strict=False)
    return frag, orig_to_new


def _extract_submol(
    mol: Chem.Mol,
    atom_indices: Sequence[int],
    *,
    add_explicit_hs: bool = False,
    set_atommap_to_orig_idx: bool = False,
    force_non_aromatic_atoms: Optional[Set[int]] = None,
) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Extract a fragment and return (frag_mol, orig_idx->frag_idx map).

    Strategy:
      1) Try a robust bond-cut based extractor (FragmentOnBonds) when the fragment boundary
         only crosses non-ring bonds.
      2) Fall back to the historical delete-atoms extractor otherwise.
    """
    try:
        return _extract_submol_by_bondcuts(
            mol,
            atom_indices,
            add_explicit_hs=add_explicit_hs,
            set_atommap_to_orig_idx=set_atommap_to_orig_idx,
            force_non_aromatic_atoms=force_non_aromatic_atoms,
        )
    except Exception:
        frag, mapping = _extract_submol_delete_atoms(
            mol,
            atom_indices,
            add_explicit_hs=add_explicit_hs,
            set_atommap_to_orig_idx=set_atommap_to_orig_idx,
            force_non_aromatic_atoms=force_non_aromatic_atoms,
        )
        # Ensure the fragment is Kekulé (no aromatic bond types) even on fallback.
        try:
            Chem.Kekulize(frag, clearAromaticFlags=True)
            Chem.AssignStereochemistry(frag, force=True, cleanIt=True)
        except Exception:
            pass
        frag.UpdatePropertyCache(strict=False)
        return frag, mapping



def _extract_submol_delete_atoms(
    mol: Chem.Mol,
    atom_indices: Sequence[int],
    *,
    add_explicit_hs: bool = False,
    set_atommap_to_orig_idx: bool = False,
    force_non_aromatic_atoms: Optional[Set[int]] = None,
) -> Tuple[Chem.Mol, Dict[int, int]]:
    """
    Build a sub-molecule by copying atoms/bonds from `mol` for the given atom indices.

    force_non_aromatic_atoms:
      original atom indices whose aromatic flag should be cleared in the fragment.
      (Useful when including isolated ring atoms as "endcaps" in chain/linker fragments.)
    """
    keep = set(atom_indices)
    force_non_aromatic_atoms = force_non_aromatic_atoms or set()

    # Copy parent to preserve atom props/valence state as much as possible
    rw = Chem.RWMol(Chem.Mol(mol))

    # Delete atoms not in fragment (descending to keep indices valid)
    for idx in sorted((i for i in range(rw.GetNumAtoms()) if i not in keep), reverse=True):
        rw.RemoveAtom(idx)

    frag = rw.GetMol()

    # Build mapping orig_idx -> new_idx using stored _orig_idx property
    # If it doesn't exist yet, set it now before we lose track.
    orig_to_new: Dict[int, int] = {}
    for new_idx, a in enumerate(frag.GetAtoms()):
        if a.HasProp("_orig_idx"):
            orig_idx = int(a.GetIntProp("_orig_idx"))
        else:
            # If parent didn't have it, recover by atom map number if present, else fall back
            # (best practice: set _orig_idx on the parent once upstream)
            orig_idx = a.GetIdx()
            a.SetIntProp("_orig_idx", int(orig_idx))
        orig_to_new[orig_idx] = new_idx
        if set_atommap_to_orig_idx:
            a.SetAtomMapNum(orig_idx + 1)

    # Optionally force certain atoms/bonds non-aromatic (by original indices)
    if force_non_aromatic_atoms:
        for orig_idx in force_non_aromatic_atoms:
            if orig_idx in orig_to_new:
                a = frag.GetAtomWithIdx(orig_to_new[orig_idx])
                a.SetIsAromatic(False)
        for b in frag.GetBonds():
            if (b.GetBeginAtom().GetIsAromatic() is False) or (b.GetEndAtom().GetIsAromatic() is False):
                b.SetIsAromatic(False)
                if b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    b.SetBondType(Chem.rdchem.BondType.SINGLE)


    _relax_noimplicit_on_cutpoints(mol, frag, keep, orig_to_new)

    frag.UpdatePropertyCache(strict=False)

    Chem.SanitizeMol(frag)
    Chem.AssignStereochemistry(frag, force=True, cleanIt=True)

    # If the fragment still cannot be kekulized, try the targeted [nH] repair for 5-member rings.
    # _repair_pyrrolic_n_in_5_member_aromatics(frag)
    frag.UpdatePropertyCache(strict=False)

    return frag, orig_to_new

def _collect_attachments_to_nonmembers(
    mol: Chem.Mol,
    *,
    frag_core_atom_set: Set[int],
    heavy_atom_set: Set[int],
    orig_to_frag: Dict[int, int],
) -> List[AttachmentPoint]:
    """
    Variant of _collect_attachments() for the common ring-fragment case where the "neighbor set"
    is simply (heavy_atom_set - frag_core_atom_set).

    This avoids allocating `external = set(heavy) - comp_set` for every ring component, while
    preserving attachment behavior exactly.
    """
    attachments: List[AttachmentPoint] = []
    for orig_idx in frag_core_atom_set:
        for nbr in mol.GetAtomWithIdx(orig_idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in heavy_atom_set:
                continue
            if nbr_idx in frag_core_atom_set:
                continue
            bond = mol.GetBondBetweenAtoms(orig_idx, nbr_idx)
            if bond is None:
                continue
            attachments.append(
                AttachmentPoint(
                    fragment_atom_idx=orig_to_frag[orig_idx],
                    fragment_atom_orig_idx=orig_idx,
                    neighbor_orig_idx=nbr_idx,
                    bond_type=bond.GetBondType(),
                    neighbor_frag_atom_idx=orig_to_frag.get(nbr_idx),
                )
            )
    uniq = {(a.fragment_atom_orig_idx, a.neighbor_orig_idx, a.bond_type): a for a in attachments}
    return list(uniq.values())

def _collect_attachments(
    mol: Chem.Mol,
    frag_core_atom_set: Set[int],
    neighbor_atom_set: Set[int],
    orig_to_frag: Dict[int, int],
) -> List[AttachmentPoint]:
    """
    For each bond from an atom in frag_core_atom_set to an atom in neighbor_atom_set,
    create an AttachmentPoint.

    If the neighbor atom is also present in the fragment molecule, neighbor_frag_atom_idx is populated.
    """
    attachments: List[AttachmentPoint] = []
    for orig_idx in frag_core_atom_set:
        for nbr in mol.GetAtomWithIdx(orig_idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in neighbor_atom_set:
                continue
            bond = mol.GetBondBetweenAtoms(orig_idx, nbr_idx)
            if bond is None:
                continue
            attachments.append(
                AttachmentPoint(
                    fragment_atom_idx=orig_to_frag[orig_idx],
                    fragment_atom_orig_idx=orig_idx,
                    neighbor_orig_idx=nbr_idx,
                    bond_type=bond.GetBondType(),
                    neighbor_frag_atom_idx=orig_to_frag.get(nbr_idx),
                )
            )
    # Deduplicate
    uniq = {(a.fragment_atom_orig_idx, a.neighbor_orig_idx, a.bond_type): a for a in attachments}
    return list(uniq.values())

###################################################

# Reduced Graph

###################################################

def _rg_edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _build_heavy_bond_index_map(mol: Chem.Mol) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    for b in mol.GetBonds():
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        if mol.GetAtomWithIdx(a1).GetAtomicNum() == 1:
            continue
        if mol.GetAtomWithIdx(a2).GetAtomicNum() == 1:
            continue
        out[_rg_edge_key(a1, a2)] = b.GetIdx()
    return out


def _canonical_cycle_order(component: Set[int], adj: Dict[int, List[int]]) -> List[int]:
    """
    Return a deterministic atom ordering around a simple all-degree-2 cycle component.
    The returned list does NOT repeat the start atom at the end.
    """
    comp = set(component)
    if len(comp) < 3:
        raise ValueError("Cycle component must have at least 3 atoms.")

    start = min(comp)
    nbrs = sorted(n for n in adj.get(start, []) if n in comp)
    if len(nbrs) != 2:
        raise ValueError("Expected a 2-regular cycle component.")

    def _walk(first_nbr: int) -> List[int]:
        order = [start]
        prev = start
        curr = first_nbr

        while True:
            order.append(curr)
            next_candidates = [n for n in adj.get(curr, []) if n in comp and n != prev]
            if not next_candidates:
                raise ValueError("Broken cycle component encountered.")
            nxt = next_candidates[0]
            if nxt == start:
                break
            prev, curr = curr, nxt

        return order

    order_a = _walk(nbrs[0])
    order_b = _walk(nbrs[1])
    return order_a if tuple(order_a) <= tuple(order_b) else order_b


def _choose_two_node_cycle_split(cycle_order: List[int]) -> int:
    """
    Choose the second retained atom index (position within cycle_order) for two_node mode.
    We prefer the most balanced split; ties are broken deterministically.
    """
    n = len(cycle_order)
    if n < 3:
        raise ValueError("two_node cycle mode requires a cycle of length >= 3.")

    best_score = None
    best_k = None

    for k in range(1, n):
        cw_len = k
        ccw_len = n - k
        score = (
            abs(cw_len - ccw_len),   # prefer balanced arcs
            max(cw_len, ccw_len),    # then prefer shorter longest arc
            cycle_order[k],          # deterministic tie-break
            k,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_k = k

    return int(best_k)


def _bond_index_path_for_atoms(
    atom_path: List[int],
    bond_idx_map: Dict[Tuple[int, int], int],
) -> Tuple[int, ...]:
    if len(atom_path) < 2:
        return ()
    return tuple(
        bond_idx_map[_rg_edge_key(atom_path[i], atom_path[i + 1])]
        for i in range(len(atom_path) - 1)
    )


def _cycle_arc_paths(cycle_order: List[int], split_k: int) -> Tuple[List[int], List[int]]:
    """
    For cycle_order = [a0, a1, ..., a(n-1)] and split_k in [1, n-1],
    return the two complementary a0->a_k paths.
    """
    if split_k <= 0 or split_k >= len(cycle_order):
        raise ValueError("split_k must be between 1 and len(cycle_order)-1.")

    path_a = list(cycle_order[: split_k + 1])
    path_b = [cycle_order[0]] + list(reversed(cycle_order[split_k:]))
    return path_a, path_b


def _build_reduced_graph(
    mol: Chem.Mol,
    *,
    heavy: Set[int],
    adj: Dict[int, List[int]],
    cycle_mode: ReducedGraphCycleMode = "single_node",
    include_paths: bool = True,
) -> ReducedGraph:
    """
    Build a reduced graph from the ORIGINAL heavy-atom graph.

    Endpoint nodes:
      - every heavy atom with degree != 2
        (this includes leaves, branch/junction atoms, and isolated heavy atoms)

    Reduced edges:
      - maximal heavy-atom paths whose internal atoms all have degree 2

    Cycle-only connected components (all atoms degree 2):
      - single_node -> one cycle surrogate node, no edges
      - self_loop   -> one cycle surrogate node + one self-loop edge carrying the full closed path
      - two_node    -> two surrogate nodes on the cycle + two complementary parallel edges
    """
    if cycle_mode not in {"single_node", "self_loop", "two_node"}:
        raise ValueError(f"Unsupported reduced-graph cycle_mode: {cycle_mode!r}")

    rg = ReducedGraph()
    deg = {i: len(adj.get(i, ())) for i in heavy}
    bond_idx_map = _build_heavy_bond_index_map(mol) if include_paths else {}
    visited_heavy_edges: Set[Tuple[int, int]] = set()

    next_node_id = 0
    next_edge_id = 0

    def add_node(orig_atom_indices: Tuple[int, ...], kind: ReducedGraphNodeKind) -> int:
        nonlocal next_node_id
        node_id = next_node_id
        next_node_id += 1
        rg.nodes.append(
            ReducedGraphNode(
                node_id=node_id,
                kind=kind,
                orig_atom_indices=tuple(orig_atom_indices),
            )
        )
        return node_id

    def add_edge(src_node_id: int, dst_node_id: int, atom_path: List[int]) -> None:
        nonlocal next_edge_id

        if include_paths:
            stored_atom_path = tuple(atom_path)
            stored_bond_path = _bond_index_path_for_atoms(atom_path, bond_idx_map)
        else:
            stored_atom_path = ()
            stored_bond_path = ()

        rg.edges.append(
            ReducedGraphEdge(
                edge_id=next_edge_id,
                src_node_id=src_node_id,
                dst_node_id=dst_node_id,
                orig_atom_path=stored_atom_path,
                orig_bond_indices=stored_bond_path,
            )
        )
        next_edge_id += 1

    for comp in _connected_components(set(heavy), adj):
        comp_set = set(comp)
        endpoints = sorted(i for i in comp_set if deg.get(i, 0) != 2)

        # Pure cycle component: no natural endpoint atoms
        if not endpoints:
            cycle_order = _canonical_cycle_order(comp_set, adj)

            if cycle_mode == "single_node":
                add_node(tuple(cycle_order), kind="cycle_surrogate")
                continue

            if cycle_mode == "self_loop":
                node_id = add_node(tuple(cycle_order), kind="cycle_surrogate")
                add_edge(node_id, node_id, list(cycle_order) + [cycle_order[0]])
                continue

            # two_node
            split_k = _choose_two_node_cycle_split(cycle_order)
            node_a = add_node((cycle_order[0],), kind="cycle_surrogate")
            node_b = add_node((cycle_order[split_k],), kind="cycle_surrogate")
            path_a, path_b = _cycle_arc_paths(cycle_order, split_k)
            add_edge(node_a, node_b, path_a)
            add_edge(node_a, node_b, path_b)
            continue

        # Normal component with natural endpoints
        endpoint_node_id: Dict[int, int] = {
            atom_idx: add_node((atom_idx,), kind="original_endpoint")
            for atom_idx in endpoints
        }

        for start in endpoints:
            start_node_id = endpoint_node_id[start]

            for nbr in adj.get(start, ()):
                first_edge_key = _rg_edge_key(start, nbr)
                if first_edge_key in visited_heavy_edges:
                    continue

                atom_path = [start]
                prev = start
                curr = nbr
                visited_heavy_edges.add(first_edge_key)

                while True:
                    atom_path.append(curr)

                    if curr in endpoint_node_id:
                        add_edge(start_node_id, endpoint_node_id[curr], atom_path)
                        break

                    next_candidates = [x for x in adj.get(curr, ()) if x != prev]
                    if len(next_candidates) != 1:
                        raise ValueError(
                            f"Expected a single continuation through degree-2 atom {curr}, "
                            f"got neighbors {next_candidates!r}."
                        )

                    nxt = next_candidates[0]
                    edge_key = _rg_edge_key(curr, nxt)
                    if edge_key in visited_heavy_edges:
                        raise ValueError(
                            "Encountered an already-visited heavy edge while tracing a reduced path. "
                            "This suggests a malformed degree-2 component."
                        )

                    visited_heavy_edges.add(edge_key)
                    prev, curr = curr, nxt

    return rg


#################################


def fragment_molecule(
    mol: Chem.Mol,
    *,
    add_explicit_hs: bool = False,
    keep_terminal_exocyclic_db: bool = False,
    keep_chain_exocyclic_db: bool = False,
    preserve_ring_linkage_db: bool = False,
    keep_terminal_linker_exocyclic_db: bool = False,
    keep_chain_linker_exocyclic_db: bool = False,
    set_atommap_to_orig_idx: bool = False,
    include_attachment_atoms_in_chain_fragments: bool = False,
    include_attachment_atoms_in_linker_fragments: bool = False,
    merge_chain_branches_at_carbon_attachments: bool = False,
    merge_chain_branches_at_hetero_attachments: bool = False,
    merge_bond_linkers_at_carbon_attachments: bool = False,
    merge_bond_linkers_at_hetero_attachments: bool = False,
    return_ring_core: bool = False,
    ensure_3d: bool = False,
    embed_seed: int = 0,
    minimize_3d: bool = True,
    maxIters_3d: int = 200,
    confId: int = 0,
    return_work_mol: bool = False,
    return_reduced_graph: bool = False,
    reduced_graph_cycle_mode: ReducedGraphCycleMode = "self_loop",
    reduced_graph_include_paths: bool = True,
) -> FragmentationResult:
    """
    Pipeline:
      1) Iteratively prune heavy atoms of degree 1 (leaf stripping), excluding ring-associated atoms.
         -> chain fragments = connected components of all pruned atoms
         Optional: include the attachment atoms (neighbors in remaining scaffold) in the chain fragment mol.
      2) On the remaining scaffold, remove non-ring atoms (linkers) to split fused/spiro ring systems.
         -> linker fragments = connected components of linker atoms
         Optional: include the ring-side attachment atoms in the linker fragment mol.
      3) Remaining components are ring systems.
         -> ring fragments = connected components of ring-associated atoms

    Attachment points are stored as metadata; if attachment atoms are included in the fragment mol,
    their fragment indices are also recorded in AttachmentPoint.neighbor_frag_atom_idx.
    """
    if mol is None:
        raise ValueError("mol is None")

    work = Chem.RemoveHs(mol)

    for _a in work.GetAtoms():
        _a.SetIntProp("_orig_idx", int(_a.GetIdx()))

    if ensure_3d:
        ensure_3d_conformer(
            work,
            confId=confId,
            seed=embed_seed,
            minimize=minimize_3d,
            maxIters=maxIters_3d,
        )



    # Build a kekulized (non-aromatic) copy of the parent specifically for fragment extraction.
    #
    # Why: extracting submols from an aromatic parent and then sanitizing them can produce
    # non-kekulizable heteroaromatics after cuts (e.g., N-rich 5-member rings where removing
    # an N-substituent requires an [nH]/tautomer shift). Extracting from a kekulized parent
    # gives RDKit a more stable starting point to recompute valence/H assignment at boundaries.
    work_extract = Chem.Mol(work)
    try:
        Chem.Kekulize(work_extract, clearAromaticFlags=True)
    except Exception:
        # Very defensive: if parent kekulization fails, fall back to the aromatic parent.
        work_extract = Chem.Mol(work)

    heavy = set(_heavy_atom_indices(work))
    adj = _build_heavy_adjacency(work)

    reduced_graph = (
        _build_reduced_graph(
            work,
            heavy=heavy,
            adj=adj,
            cycle_mode=reduced_graph_cycle_mode,
            include_paths=reduced_graph_include_paths,
        ) if return_reduced_graph
        else None
    )

    ring_atoms_all = {a.GetIdx() for a in work.GetAtoms() if a.GetAtomicNum() != 1 and a.IsInRing()}

    exocyclic_double_bonds = _classify_ring_exocyclic_double_bonds(
        work, heavy=heavy, ring_atoms_all=ring_atoms_all, adj=adj
    )

    ring_associated: Set[int] = set(ring_atoms_all)

    for info in exocyclic_double_bonds:
        if info.kind == "terminal_exocyclic_db" and keep_terminal_exocyclic_db:
            ring_associated.add(info.exocyclic_atom_idx)
        elif info.kind == "chain_exocyclic_db" and keep_chain_exocyclic_db:
            ring_associated.add(info.exocyclic_atom_idx)

    # -------- Stage 1: leaf pruning --------
    remaining = set(heavy)
    pruned: Set[int] = set()

    while True:
        deg1 = []
        for i in list(remaining):
            if i in ring_associated:
                continue
            d = sum(1 for j in adj.get(i, []) if j in remaining)
            if d <= 1:
                deg1.append(i)
        if not deg1:
            break
        for i in deg1:
            remaining.remove(i)
            pruned.add(i)

    fragments: List[Fragment] = []

    # -------- Stage 2 pre-classification (used to support optional linker exocyclic DB retention) --------
    # We compute the Stage-2 base linker set from the Stage-1 `remaining` scaffold *before* emitting
    # chain fragments so that selected pruned exocyclic DB atoms can be promoted out of chain fragments
    # and retained with linker fragments. This preserves the existing ring behavior model: only the
    # exocyclic atom is promoted, while downstream atoms remain chains.
    ring_associated_stage2 = set(ring_associated)
    promoted_inter_ring_exo_atoms: Set[int] = set()

    if preserve_ring_linkage_db:
        for info in exocyclic_double_bonds:
            if info.kind == "inter_ring_linkage_exocyclic_db" and info.exocyclic_atom_idx in remaining:
                ring_associated_stage2.add(info.exocyclic_atom_idx)
                promoted_inter_ring_exo_atoms.add(info.exocyclic_atom_idx)

    ring_core = {i for i in remaining if i in ring_associated_stage2}
    base_linkers = set(remaining) - ring_core

    linker_exocyclic_double_bonds = _classify_linker_exocyclic_double_bonds(
        work,
        heavy=heavy,
        adj=adj,
        linker_atoms=base_linkers,
        chain_candidate_atoms=pruned,
    )

    promoted_linker_exo_atoms: Set[int] = set()
    for info in linker_exocyclic_double_bonds:
        if info.kind == "terminal_linker_exocyclic_db" and keep_terminal_linker_exocyclic_db:
            promoted_linker_exo_atoms.add(info.exocyclic_atom_idx)
        elif info.kind == "chain_linker_exocyclic_db" and keep_chain_linker_exocyclic_db:
            promoted_linker_exo_atoms.add(info.exocyclic_atom_idx)

    # Final chain pool after optional linker exocyclic-DB promotions.
    pruned_for_chain = set(pruned) - promoted_linker_exo_atoms

    # When promoted linker exocyclic atoms are removed from the chain pool, they act as attachment
    # scaffold atoms for residual chain fragments (especially for non-terminal promoted linker exo carbons).
    chain_attachment_scaffold = set(remaining) | promoted_linker_exo_atoms

    # Chain fragments = components of pruned atoms
    # Optional post-processing: merge components that share the same attachment atom
    # (e.g., multiple =O / substituent branches attached to the same retained atom).
    if pruned_for_chain:
        chain_components = _connected_components(pruned_for_chain, adj)

        # Cache chain->scaffold neighbor sets once (used for merges and for emitting chain fragments).
        atom_to_scaffold_neighbors: Dict[int, Tuple[int, ...]] = {
            u: tuple(v for v in adj.get(u, []) if v in chain_attachment_scaffold)
            for u in pruned_for_chain
        }

        chain_components, chain_attach_atoms = _merge_chain_components_by_shared_attachment_with_attachments(
            work,
            chain_components,
            adj=adj,
            remaining=chain_attachment_scaffold,
            merge_at_carbon_attachments=merge_chain_branches_at_carbon_attachments,
            merge_at_hetero_attachments=merge_chain_branches_at_hetero_attachments,
            atom_to_remaining_neighbors=atom_to_scaffold_neighbors,
        )

        for comp, attach_atoms in zip(chain_components, chain_attach_atoms):
            comp_set = set(comp)
            comp_sorted = tuple(sorted(comp_set))

            # Attachment atoms (neighbors in the chain attachment scaffold) were precomputed above.
            if include_attachment_atoms_in_chain_fragments:
                atom_indices = tuple(sorted(comp_set | attach_atoms))
                # If we include ring atoms as endcaps, clear aromatic flag on those endcap atoms
                force_non_arom = attach_atoms & ring_atoms_all
                neighbor_set_for_attachments = attach_atoms
                included_attach_tuple = tuple(sorted(attach_atoms))
            else:
                atom_indices = comp_sorted
                force_non_arom = set()
                neighbor_set_for_attachments = chain_attachment_scaffold
                included_attach_tuple = ()

            frag_mol, orig_to_frag = _extract_submol(
                work_extract,
                atom_indices,
                add_explicit_hs=add_explicit_hs,
                set_atommap_to_orig_idx=set_atommap_to_orig_idx,
                force_non_aromatic_atoms=force_non_arom,
            )

            attachments = _collect_attachments(
                work,
                frag_core_atom_set=comp_set,
                neighbor_atom_set=set(neighbor_set_for_attachments),
                orig_to_frag=orig_to_frag,
            )

            fragments.append(
                Fragment(
                    kind="chain",
                    mol=frag_mol,
                    orig_atom_indices=atom_indices,
                    atom_map_orig_to_frag=orig_to_frag,
                    attachments=attachments,
                    included_attachment_atom_orig_indices=included_attach_tuple,
                )
            )

    if not remaining:
        return _package_fragmentation_result(
            work,
            fragments=fragments,
            reduced_graph=reduced_graph,
            return_work_mol=return_work_mol,
            return_reduced_graph=return_reduced_graph,
        )


    if return_ring_core:
        # --- Return the post-chain-removal core as a Fragment(kind="ring_core") ---
        core_after_chain = set(remaining)
        core_atom_indices = tuple(sorted(core_after_chain))

        core_mol, core_orig_to_frag = _extract_submol(
            work_extract,
            core_atom_indices,
            add_explicit_hs=add_explicit_hs,
            set_atommap_to_orig_idx=set_atommap_to_orig_idx,
        )

        # Attachments from core -> pruned chain atoms (lets you see where chains were removed)
        core_attachments = _collect_attachments(
            work,
            frag_core_atom_set=core_after_chain,
            neighbor_atom_set=set(pruned_for_chain),
            orig_to_frag=core_orig_to_frag,
        )

        fragments.append(
            Fragment(
                kind="ring_core",
                mol=core_mol,
                orig_atom_indices=core_atom_indices,
                atom_map_orig_to_frag=core_orig_to_frag,
                attachments=core_attachments,
            )
        )


    # -------- Stage 2: split ring systems by removing linkers (non-ring atoms) --------
    linkers = set(base_linkers) | promoted_linker_exo_atoms
    # --- Special handling: "bond linkers" (ring_core <-> ring_core bonds not in any ring) ---
    # This includes:
    #   - direct ring-ring exocyclic bonds
    #   - ring <-> promoted exocyclic atom bonds (when preserve_ring_linkage_double_bonds=True)
    #
    # Terminal exocyclic atoms (e.g., carbonyl O) are excluded by the degree check so they remain
    # embedded in the ring fragment.
    bond_linker_edges: Set[Tuple[int, int]] = set()

    # Ring-only adjacency excluding bond-linker edges (built once here to avoid a second GetBonds() pass).
    ring_adj: Dict[int, List[int]] = {i: [] for i in ring_core}

    # Cache degree within the Stage-1 remaining scaffold to avoid repeated adj scans inside the bond loop.
    deg_in_remaining: Dict[int, int] = {
        i: sum(1 for j in adj.get(i, []) if j in remaining)
        for i in ring_core
    }

    for b in work.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()

        if not (a1 in ring_core and a2 in ring_core):
            continue

        # Ring bonds are never cut as bond-linkers; include them directly in ring adjacency.
        if b.IsInRing():
            ring_adj[a1].append(a2)
            ring_adj[a2].append(a1)
            continue

        # avoid peeling off terminal retained exocyclic atoms as "linkers"
        if deg_in_remaining.get(a1, 0) <= 1 or deg_in_remaining.get(a2, 0) <= 1:
            ring_adj[a1].append(a2)
            ring_adj[a2].append(a1)
            continue

        # preserve double-bond ring linkages if requested
        if b.GetBondType() == Chem.rdchem.BondType.DOUBLE and preserve_ring_linkage_db:
            ring_adj[a1].append(a2)
            ring_adj[a2].append(a1)
            continue

        a1_is_true_ring = a1 in ring_atoms_all
        a2_is_true_ring = a2 in ring_atoms_all
        a1_is_promoted = a1 in promoted_inter_ring_exo_atoms
        a2_is_promoted = a2 in promoted_inter_ring_exo_atoms

        is_ring_linker_edge = False

        # direct ring-ring bond linker (legacy behavior)
        if a1_is_true_ring and a2_is_true_ring:
            is_ring_linker_edge = True

        # ring <-> promoted inter-ring exocyclic atom bond linker
        elif (a1_is_true_ring and a2_is_promoted) or (a2_is_true_ring and a1_is_promoted):
            is_ring_linker_edge = True

        # promoted inter-ring exocyclic atom <-> promoted inter-ring exocyclic atom
        # (e.g., ring=CH-CH=ring where preserve_ring_linkage_db promotes both terminal linker carbons).
        # The internal linker bond must remain a linker cut edge so the rings do not collapse into one ring fragment.
        elif a1_is_promoted and a2_is_promoted:
            is_ring_linker_edge = True

        if is_ring_linker_edge:
            bond_linker_edges.add(tuple(sorted((a1, a2))))
            continue

        # Not a bond-linker cut edge => keep connectivity in ring adjacency
        ring_adj[a1].append(a2)
        ring_adj[a2].append(a1)

    # --- 1) Atom-based linker fragments (non-ring atoms in `linkers`) ---
    if linkers:
        for comp in _connected_components(linkers, adj):
            comp_set = set(comp)
            comp_sorted = tuple(sorted(comp_set))

            # Linker attachments are ring-core neighbors, plus (only for promoted linker exocyclic DB atoms)
            # residual chain-side neighbors that remain in the final chain pool.
            attach_atoms: Set[int] = set()
            for u in comp_set:
                for v in adj.get(u, []):
                    if v in ring_core:
                        attach_atoms.add(v)
                    elif u in promoted_linker_exo_atoms and v in pruned_for_chain:
                        attach_atoms.add(v)

            if include_attachment_atoms_in_linker_fragments:
                atom_indices = tuple(sorted(comp_set | attach_atoms))
                # Included ring atoms as endcaps -> clear aromatic flag on those endcaps
                force_non_arom = attach_atoms & ring_atoms_all
                neighbor_set_for_attachments = attach_atoms
                included_attach_tuple = tuple(sorted(attach_atoms))
            else:
                atom_indices = comp_sorted
                force_non_arom = set()
                neighbor_set_for_attachments = attach_atoms
                included_attach_tuple = ()

            frag_mol, orig_to_frag = _extract_submol(
                work_extract,
                atom_indices,
                add_explicit_hs=add_explicit_hs,
                set_atommap_to_orig_idx=set_atommap_to_orig_idx,
                force_non_aromatic_atoms=force_non_arom,
            )

            attachments = _collect_attachments(
                work,
                frag_core_atom_set=comp_set,
                neighbor_atom_set=set(neighbor_set_for_attachments),
                orig_to_frag=orig_to_frag,
            )

            fragments.append(
                Fragment(
                    kind="linker",
                    mol=frag_mol,
                    orig_atom_indices=atom_indices,
                    atom_map_orig_to_frag=orig_to_frag,
                    attachments=attachments,
                    included_attachment_atom_orig_indices=included_attach_tuple,
                )
            )

    # --- 2) Bond-linker fragments (ring–ring single bond "linkers" with no linker atoms) ---
    # Optionally group multiple bond-linker edges into a single linker fragment when they
    # share the same attachment atom (e.g., a retained promoted exocyclic carbon attached
    # to multiple ring atoms).
    bond_linker_atom_groups = _group_bond_linker_edges_by_shared_attachment(
        work,
        bond_linker_edges,
        merge_at_carbon_attachments=merge_bond_linkers_at_carbon_attachments,
        merge_at_hetero_attachments=merge_bond_linkers_at_hetero_attachments,
    )

    for comp_set in bond_linker_atom_groups:
        comp_set = set(comp_set)
        atom_indices = tuple(sorted(comp_set))

        # Isolated ring atoms / promoted exo atoms in linker fragment -> clear aromatic flags on true ring atoms
        force_non_arom = (comp_set & ring_atoms_all)

        frag_mol, orig_to_frag = _extract_submol(
            work_extract,
            atom_indices,
            add_explicit_hs=add_explicit_hs,
            set_atommap_to_orig_idx=set_atommap_to_orig_idx,
            force_non_aromatic_atoms=force_non_arom,
        )

        # Attachments: each atom in this grouped bond-linker fragment attaches into the rest of ring_core
        attachments = _collect_attachments(
            work,
            frag_core_atom_set=comp_set,
            neighbor_atom_set=(ring_core - comp_set),
            orig_to_frag=orig_to_frag,
        )

        fragments.append(
            Fragment(
                kind="linker",
                mol=frag_mol,
                orig_atom_indices=atom_indices,
                atom_map_orig_to_frag=orig_to_frag,
                attachments=attachments,
                included_attachment_atom_orig_indices=(),
            )
        )



    # -------- Stage 3: ring system fragments --------
    if ring_core:
        for comp in _connected_components(ring_core, ring_adj):
            comp_set = set(comp)
            comp_sorted = tuple(sorted(comp_set))

            frag_mol, orig_to_frag = _extract_submol(
                work_extract,
                comp_sorted,
                add_explicit_hs=add_explicit_hs,
                set_atommap_to_orig_idx=set_atommap_to_orig_idx,
            )
            attachments = _collect_attachments_to_nonmembers(
                work,
                frag_core_atom_set=comp_set,
                heavy_atom_set=heavy,
                orig_to_frag=orig_to_frag,
            )

            fragments.append(
                Fragment(
                    kind="ring",
                    mol=frag_mol,
                    orig_atom_indices=comp_sorted,
                    atom_map_orig_to_frag=orig_to_frag,
                    attachments=attachments,
                )
            )

    return _package_fragmentation_result(
        work,
        fragments=fragments,
        reduced_graph=reduced_graph,
        return_work_mol=return_work_mol,
        return_reduced_graph=return_reduced_graph,
    )

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract HiFrAMes (chain/ring/linker) fragments from a CSV of molecules and count frequencies."
    )
    ap.add_argument("--smi", required=True, help="Input SMILES string to process (for quick testing).")
    ap.add_argument("--smiles-col", default="smiles", help="Column name for SMILES. Default: smiles")
    ap.add_argument("--outdir", default="fragment_outputs", help="Output directory. Default: fragment_outputs")

    # Output SMILES control
    ap.add_argument("--isomeric", action="store_true", help="Keep stereochemistry in fragment SMILES.")
    ap.add_argument("--add-explicit-hs", action="store_true", help="Add explicit Hs in fragment mols before SMILES.")

    # Defaults chosen to match your “looks correct” test
    ap.add_argument(
        "--keep-terminal-ring-exocyclic-db",
        action="store_true",
        default=False,
        help="Keep terminal ring exocyclic double-bond atoms with the ring system (e.g., carbonyl O).",
    )
    ap.add_argument(
        "--keep-chain-exocyclic-db",
        action="store_true",
        default=False,
        help="Keep terminal exocyclic double-bond atoms on chain fragments (e.g., carbonyl O).",
    )
    ap.add_argument(
        "--preserve-ring-linkage-db",
        action="store_true",
        default=False,
        help=(
            "Preserve ring-linkage double bonds with ring fragments (direct ring–ring exocyclic "
            "double bonds and linker-like ring=X attachments). Default: OFF for pure topological fragmentation."
        ),
    )
    ap.add_argument(
        "--include-chain-attachment-atoms",
        action="store_true",
        default=True,
        help="(Default: ON) Include attachment atoms in chain fragments (endcaps).",
    )
    ap.add_argument(
        "--no-include-chain-attachment-atoms",
        action="store_false",
        dest="include_chain_attachment_atoms",
        help="Disable chain endcaps.",
    )
    ap.add_argument(
        "--include-linker-attachment-atoms",
        action="store_true",
        default=True,
        help="(Default: ON) Include attachment atoms in linker fragments (endcaps).",
    )
    ap.add_argument(
        "--no-include-linker-attachment-atoms",
        action="store_false",
        dest="include_linker_attachment_atoms",
        help="Disable linker endcaps.",
    )
    ap.add_argument(
        "--merge-chain-branches-at-carbon-attachments",
        action="store_true",
        default=False,
        help=(
            "Merge Stage-1 chain branches that share the same attachment atom when that "
            "attachment atom is carbon. Default: OFF (preserves current behavior)."
        ),
    )
    ap.add_argument(
        "--merge-chain-branches-at-hetero-attachments",
        action="store_true",
        default=False,
        help=(
            "Merge Stage-1 chain branches that share the same attachment atom when that "
            "attachment atom is a heteroatom (non-carbon). Default: OFF."
        ),
    )
    ap.add_argument(
        "--merge-bond-linkers-at-carbon-attachments",
        action="store_true",
        default=False,
        help=(
            "Merge bond-linker fragments that share the same attachment atom when the "
            "shared attachment atom is carbon. Default: OFF."
        ),
    )
    ap.add_argument(
        "--merge-bond-linkers-at-hetero-attachments",
        action="store_true",
        default=False,
        help=(
            "Merge bond-linker fragments that share the same attachment atom when the "
            "shared attachment atom is a heteroatom (non-carbon). Default: OFF."
        ),
    )
    ap.add_argument(
        "--keep-terminal-linker-exocyclic-db",
        action="store_true",
        default=False,
        help=(
            "Keep terminal linker exocyclic double-bond atoms with linker fragments "
            "(e.g., linker carbonyl oxygen on a linker atom). Default: OFF."
        ),
    )
    ap.add_argument(
        "--keep-chain-linker-exocyclic-db",
        action="store_true",
        default=False,
        help=(
            "Keep non-terminal/chain linker exocyclic double-bond atoms with linker fragments "
            "(promote only the exocyclic atom; downstream atoms remain chains). Default: OFF."
        ),
    )

    ap.add_argument("--quiet-rdkit", action="store_true", help="Suppress RDKit warnings.")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap for debugging (0 = no cap).")

    # Progress bar controls
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bar.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="tqdm miniters (roughly update every N molecules). Default: 200",
    )
    # # take in single SMILES string
    # ap.add_argument("smi", nargs="?", help="Optional single SMILES string to process instead of CSV input (for quick testing).")

    args = ap.parse_args()

    # turn smi to RDKit Mol
    if args.smi:
        mol = Chem.MolFromSmiles(args.smi)
        if mol is None:
            print(f"Error: Invalid SMILES string: {args.smi}")
            return 1


    frags = fragment_molecule(
        mol,
        add_explicit_hs=args.add_explicit_hs,
        keep_terminal_exocyclic_db=args.keep_terminal_ring_exocyclic_db,
        keep_chain_exocyclic_db=args.keep_chain_exocyclic_db,
        preserve_ring_linkage_db=args.preserve_ring_linkage_db,
        keep_terminal_linker_exocyclic_db=args.keep_terminal_linker_exocyclic_db,
        keep_chain_linker_exocyclic_db=args.keep_chain_linker_exocyclic_db,
        set_atommap_to_orig_idx=False,
        include_attachment_atoms_in_chain_fragments=args.include_chain_attachment_atoms,
        include_attachment_atoms_in_linker_fragments=args.include_linker_attachment_atoms,
        merge_chain_branches_at_carbon_attachments=args.merge_chain_branches_at_carbon_attachments,
        merge_chain_branches_at_hetero_attachments=args.merge_chain_branches_at_hetero_attachments,
        merge_bond_linkers_at_carbon_attachments=args.merge_bond_linkers_at_carbon_attachments,
        merge_bond_linkers_at_hetero_attachments=args.merge_bond_linkers_at_hetero_attachments,
        return_ring_core=False,
        return_reduced_graph=True,
        reduced_graph_cycle_mode="two_node",
    )

    # # # print results
    # for i, frag in enumerate(frags.fragments):
    #     print(f"Fragment {i}: kind={frag.kind}, orig_atoms={frag.orig_atom_indices}, attachments={frag.attachments}")
    #     print(Chem.MolToSmiles(frag.mol))


    # Display Reduced Multigraph (if generated)

    # if isinstance(frags, tuple):
    #     if len(frags) == 3 and isinstance(frags[2], ReducedGraph):
    #         reduced_graph = frags[2]
    #     elif len(frags) == 2 and isinstance(frags[1], ReducedGraph):
    #         reduced_graph = frags[1]
    #     else:
    #         reduced_graph = None
    # else:
    #     reduced_graph = None

    # if reduced_graph is not None:
    #     import networkx as nx
    #     import matplotlib.pyplot as plt

    #     G = nx.MultiGraph()
    #     for node in reduced_graph.nodes:
    #         G.add_node(node.node_id, kind=node.kind, orig_atom_indices=node.orig_atom_indices)
    #     for edge in reduced_graph.edges:
    #         G.add_edge(edge.src_node_id, edge.dst_node_id, edge_id=edge.edge_id, orig_atom_path=edge.orig_atom_path)

    #     from collections import defaultdict
    #     pos = nx.spring_layout(G)

    #     plt.figure(figsize=(8, 6))
    #     nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
    #     nx.draw_networkx_labels(G, pos)

    #     # Count parallel edges per unordered pair
    #     pair_counts = defaultdict(int)
    #     for u, v, k in G.edges(keys=True):
    #         pair_counts[tuple(sorted((u, v)))] += 1

    #     pair_seen = defaultdict(int)
    #     rad_step = 0.45  # increase if you still want more separation

    #     for u, v, k, data in G.edges(keys=True, data=True):
    #         pair = tuple(sorted((u, v)))
    #         m = pair_counts[pair]
    #         i = pair_seen[pair]
    #         pair_seen[pair] += 1

    #         rad = 0.0 if m == 1 else (i - (m - 1) / 2) * rad_step

    #         nx.draw_networkx_edges(
    #             G, pos,
    #             edgelist=[(u, v)],
    #             arrows=True,              # <-- forces FancyArrowPatch
    #             arrowstyle='-',           # <-- line, no arrowheads
    #             arrowsize=10,             # irrelevant-ish with '-', but required by nx
    #             connectionstyle=f"arc3,rad={rad}",
    #             edge_color="gray",
    #             width=2,
    #             min_source_margin=8,
    #             min_target_margin=8,
    #         )

    #         # Optional edge_id label
    #         edge_id = data.get("edge_id", k)
    #         mx = (pos[u][0] + pos[v][0]) / 2
    #         my = (pos[u][1] + pos[v][1]) / 2
    #         plt.text(mx, my + rad * 0.15, str(edge_id), fontsize=9, ha="center", va="center")

    #     plt.title("Reduced Graph Visualization (parallel edges shown)")
    #     plt.axis("off")
    #     plt.show()

if __name__ == "__main__":
    raise SystemExit(main())