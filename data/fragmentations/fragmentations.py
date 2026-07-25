import pickle
from collections import Counter
from itertools import combinations, permutations
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
from data.fragmentations.ErtlEFG import ertl_efg
import torch
from rdkit import Chem
from rdkit.Chem.BRICS import BreakBRICSBonds, BRICSDecompose
from rdkit.Chem.rdmolops import GetMolFrags
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from data.fragmentations.bbb.breaking_bridge_bonds import (
    MotifExtractionSettings, MotifVocabularyExtractor,
    find_motifs_from_vocabulary)
from data.fragmentations.magnet.magnet import MolDecomposition
from data.fragmentations.psm.mol_bpe import Tokenizer, graph_bpe_smiles
from data.fragmentations.hiframes import hiframes
import os


ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Cu", "Zn", 'Co', "Mn", 'As', 'Al', 'Ni', 'Se', 'Si', 'H', 'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Fe', 'Ga', 'Ge', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
             'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']

fragment2type = {"ring": 0, "path": 1, "junction": 2, "efg": 3}

BRICS_OOV_TOKEN = "__BRICS_OOV__"
ERTL_EFG_OOV_TOKEN = "__ERTL_EFG_OOV__"

HIFRAMES_LABEL_MODE_ALIASES = {
    "family": "family",
    "kind": "family",
    "family_only": "family",
    "family-only": "family",
    "type": "family",
    "coarse": "family",
    "family_size": "family_size",
    "family+size": "family_size",
    "family_plus_size": "family_size",
    "family-plus-size": "family_size",
    "kind_size": "family_size",
    "kind+size": "family_size",
    "size": "family_size",
    "sized": "family_size",
}


def normalize_hiframes_label_mode(label_mode: str) -> str:
    mode = str(label_mode).strip().lower()
    if mode not in HIFRAMES_LABEL_MODE_ALIASES:
        raise RuntimeError(
            "Unsupported HiFrAMes label_mode: "
            f"{label_mode!r}. Supported values are 'family' and 'family_size'."
        )
    return HIFRAMES_LABEL_MODE_ALIASES[mode]


def _hiframes_int_param(params: Dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        if key in params and params[key] is not None:
            return int(params[key])
    return int(default)


def get_hiframes_size_bins(params: Dict[str, Any]) -> Tuple[int, int, int]:
    shared_default = _hiframes_int_param(
        params,
        'size_bins',
        'size_cap',
        'max_size_bin',
        default=15,
    )
    chain_bins = _hiframes_int_param(
        params,
        'chain_size_bins',
        'chain_size_cap',
        default=shared_default,
    )
    linker_bins = _hiframes_int_param(
        params,
        'linker_size_bins',
        'linker_size_cap',
        default=shared_default,
    )
    ring_bins = _hiframes_int_param(
        params,
        'ring_size_bins',
        'ring_size_cap',
        default=shared_default,
    )
    if min(chain_bins, linker_bins, ring_bins) <= 0:
        raise RuntimeError(
            'HiFrAMes size-bin counts must all be positive. '
            f'Got chain={chain_bins}, linker={linker_bins}, ring={ring_bins}.'
        )
    return chain_bins, linker_bins, ring_bins


def infer_hiframes_vocab_size(params: Dict[str, Any]) -> int:
    mode = normalize_hiframes_label_mode(params.get('label_mode', 'family_size'))
    if mode == 'family':
        return 3
    chain_bins, linker_bins, ring_bins = get_hiframes_size_bins(params)
    return chain_bins + linker_bins + ring_bins


def _prepare_optional_oov_vocab(vocab, oov_token, use_oov_bucket: bool):
    """
    Backward-compatible helper.

    - If use_oov_bucket=False: leave vocab unchanged, return oov_idx=None.
    - If use_oov_bucket=True and token already exists: reuse it.
    - If use_oov_bucket=True and token is absent:
        preserve total vocab size by replacing the last entry with the OOV token.
        (Cleaner long-term: rebuild the vocab file once with an explicitly reserved slot.)
    """
    vocab = list(vocab)

    if not use_oov_bucket:
        return vocab, None

    if len(vocab) == 0:
        return [oov_token], 0

    if oov_token in vocab:
        return vocab, vocab.index(oov_token)

    vocab[-1] = oov_token
    return vocab, len(vocab) - 1

def is_leaf(node_id, graph):

    neighbors = get_neighbors(node_id, graph)
    if len(neighbors) == 1:
        neighbor = neighbors[0]
        if graph.mol.GetAtomWithIdx(neighbor).IsInRing():
            return True
        nns = get_neighbors(neighbor, graph)
        degree_nn = [get_degree(nn, graph) for nn in nns]
        if len([degree for degree in degree_nn if degree >= 2]) >= 2:
            return True
        # one neighbor neighbor with degree one is not a leaf
        potential_leafs = [nn for nn in nns if get_degree(nn, graph) == 1]
        atom_types = [(ATOM_LIST.index(graph.mol.GetAtomWithIdx(
            nn).GetSymbol()), nn) for nn in potential_leafs]
        sorted_idx = np.sort(atom_types)
        if sorted_idx[-1][1] == node_id:
            # node at end of path
            return False
        else:
            return True
    return False


def vocab_to_file(vocab, file_name):
    if not file_name:
        file_name = f"./{vocab.__name__}_vocab_{vocab.max_vocab_size}"

    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    with open(file_name, "wb") as f:
        pickle.dump(vocab.get_vocab(), f)


def get_vocab_from_file(file_name):
    return pickle.load(open(file_name, "rb"))


def get_neighbors(node_id, graph):
    return (graph.edge_index[1, graph.edge_index[0, :] == node_id]).tolist()


def get_degree(node_id, graph):
    return len(get_neighbors(node_id, graph))


class BreakingBridgeBondsVocab(BaseTransform):
    def __init__(self, min_frequency=None, min_num_atoms=3, cut_leaf_edges=False, vocab_size=200):
        settings = MotifExtractionSettings(
            min_frequency=min_frequency, min_num_atoms=min_num_atoms, cut_leaf_edges=cut_leaf_edges, max_vocab_size=vocab_size)
        self.extractor = MotifVocabularyExtractor(settings)
        self.max_vocab_size = vocab_size

    def __call__(self, graph):
        mol = graph.mol
        self.extractor.update(mol)
        return graph

    def get_vocab(self):
        return self.extractor.output()


class BRICSVocab(BaseTransform):
    def __init__(self, vocab_size=200):
        self.max_vocab_size = vocab_size
        self.counter = Counter()

    def __call__(self, graph):
        mol = graph.mol
        fragments = [Chem.MolToSmiles(fragment) for fragment in GetMolFrags(
            BreakBRICSBonds(mol), asMols=True)]
        # filter fragments with only one atom
        # TODO
        self.counter.update(fragments)
        return graph

    def get_vocab(self):
        # print([motif for motif, _ in self.counter.most_common(self.max_vocab_size)])
        return [motif for motif, _ in self.counter.most_common(self.max_vocab_size)]


class PrincipalSubgraphVocab(BaseTransform):
    def __init__(self, vocab_size=200, vocab_path="./principal_subgraph_vocab.txt", cpus=4, kekulize=False):
        self.max_vocab_size = vocab_size
        self.smis = []
        self.vocab_path = vocab_path
        self.cpus = cpus
        self.kekulize = kekulize

    def __call__(self, graph):
        self.smis.append(Chem.MolToSmiles(graph.mol))
        return graph

    def get_vocab(self):
        graph_bpe_smiles(self.smis, vocab_len=self.max_vocab_size,
                         vocab_path=self.vocab_path, cpus=self.cpus, kekulize=self.kekulize)
        return self.vocab_path


class MagnetVocab(BaseTransform):
    def __init__(self, vocab_size=200):
        self.max_vocab_size = vocab_size
        self.hash_counter = Counter()
        self.hash_to_smiles = {}

    def __call__(self, graph):
        mols = Chem.Mol(graph.mol)  # create copy of molecule
        for mol in Chem.rdmolops.GetMolFrags(mols, asMols=True):
            decomposition = MolDecomposition(mol)
            hashes = decomposition.id_to_hash.values()
            for (id, hash) in decomposition.id_to_hash.items():
                if hash not in self.hash_to_smiles and hash != -1:
                    self.hash_to_smiles[hash] = decomposition.id_to_fragment[id]
            self.hash_counter.update(hashes)

    def get_vocab(self):
        return [hash for (hash, _) in self.hash_counter.most_common(self.max_vocab_size)]


class ErtlEFGVocab(BaseTransform):
    """
    Vocabulary of EFGs using Ertl's pseudo-SMILES
    """
    def __init__(self, vocab_size: int = 200):
        self.max_vocab_size = vocab_size
        self.counter = Counter()

    def __call__(self, graph):
        mol = graph.mol
        try:
            _, _, psmis, _ = ertl_efg.get_dec_fgs(mol)
        except Exception:
            return graph
        self.counter.update(psmis)
        return graph

    def get_vocab(self):
        return [fg for fg, _ in self.counter.most_common(self.max_vocab_size)]


class Magnet(BaseTransform):
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hash_to_index = {hash: id for id, hash in enumerate(self.vocab)}

    def __call__(self, graph):
        mols = Chem.Mol(graph.mol)  # create copy of molecule
        for mol in Chem.rdmolops.GetMolFrags(mols, asMols=True):
            # There can be multiple disconnected parts of a molecule
            decomposition = MolDecomposition(mol)
            ids_in_vocab = [
                id for (id, hash) in decomposition.id_to_hash.items() if hash in self.vocab]

            node_substructures = []
            fragment_to_index = {}
            for fragments in decomposition.nodes.values():
                fragment_info = []
                for frag_id in fragments:
                    if frag_id in ids_in_vocab:
                        hash = decomposition.id_to_hash[frag_id]
                        if frag_id not in fragment_to_index:
                            fragment_to_index[frag_id] = len(fragment_to_index)
                        fragment_info.append(
                            (fragment_to_index[frag_id], self.hash_to_index[hash]))
                node_substructures.append(fragment_info)

        graph.substructures = node_substructures
        return graph


class MagnetWithoutVocab(BaseTransform):
    def __init__(self, vocab_size=None):
        pass

    def __call__(self, graph):
        mols = Chem.Mol(graph.mol)  # create copy of molecule
        fragment_types = []
        node_substructures = []
        for mol in Chem.rdmolops.GetMolFrags(mols, asMols=True):
            # There can be multiple disconnected parts of a molecule
            decomposition = MolDecomposition(mol)

            fragment_to_index = {}
            fragment_to_type = {}
            for fragments in decomposition.nodes.values():
                fragment_info = []
                for frag_id in fragments:
                    if frag_id == -1:
                        # don't use leafs ?!
                        continue

                    if frag_id not in fragment_to_type:
                        frag_mol = Chem.MolFromSmiles(
                            decomposition.id_to_fragment[frag_id])
                        if frag_mol.GetAtomWithIdx(0).IsInRing():
                            fragment_to_type[frag_id] = [
                                fragment2type["ring"], frag_mol.GetNumAtoms()]
                        elif all([a.GetDegree() in [1, 2] for a in frag_mol.GetAtoms()]):
                            fragment_to_type[frag_id] = [
                                fragment2type["path"], frag_mol.GetNumAtoms()]
                        else:
                            fragment_to_type[frag_id] = [
                                fragment2type["junction"], frag_mol.GetNumAtoms()]

                    if frag_id not in fragment_to_index:
                        fragment_to_index[frag_id] = len(fragment_types)
                        fragment_types.append(fragment_to_type[frag_id])

                    fragment_info.append(
                        (fragment_to_index[frag_id], fragment_to_type[frag_id][0]))

                node_substructures.append(fragment_info)

        if fragment_types:
            graph.fragment_types = torch.tensor(
                fragment_types, dtype=torch.long)
        else:
            graph.fragment_types = torch.empty((0, 2), dtype=torch.long)
        graph.substructures = node_substructures
        return graph


class BRICS(BaseTransform):
    def __init__(self, vocab: List[str], use_oov_bucket: bool = False):
        self.use_oov_bucket = bool(use_oov_bucket)
        self.vocab, self.oov_idx = _prepare_optional_oov_vocab(
            vocab, BRICS_OOV_TOKEN, self.use_oov_bucket
        )
        self.vocab_size = len(self.vocab)
        self.frag_to_idx = {frag: idx for idx, frag in enumerate(self.vocab)}

    def __call__(self, graph):
        mol = graph.mol
        node_substructures = [[] for _ in range(graph.num_nodes)]
        fragment_types = []

        fragments = GetMolFrags(BreakBRICSBonds(mol), asMols=True)
        fragments_atom_ids = GetMolFrags(BreakBRICSBonds(mol))

        fragment_id = 0
        for fragment, atom_ids in zip(fragments, fragments_atom_ids):
            smi = Chem.MolToSmiles(fragment)

            fragment_type = self.frag_to_idx.get(smi)
            if fragment_type is None:
                if not self.use_oov_bucket:
                    continue
                fragment_type = self.oov_idx

            atom_ids_filtered = [
                atom_id for atom_id in atom_ids if atom_id < graph.num_nodes
            ]

            for atom_id in atom_ids_filtered:
                node_substructures[atom_id].append((fragment_id, fragment_type))

            # keep existing metadata semantics unchanged
            fragment_types.append([0, len(atom_ids_filtered)])
            fragment_id += 1

        graph.substructures = node_substructures
        graph.fragment_types = (
            torch.tensor(fragment_types, dtype=torch.long)
            if fragment_types else torch.empty((0, 2), dtype=torch.long)
        )
        return graph

class BreakingBridgeBonds(BaseTransform):
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab.vocabulary)

    def __call__(self, graph):
        mol = graph.mol
        node_substructures = [[] for _ in range(graph.num_nodes)]
        for fragment_id, fragment in enumerate(find_motifs_from_vocabulary(mol, self.vocab)):
            fragment_type = self.vocab.vocabulary[fragment.motif_type]
            atoms = [atom.atom_id for atom in fragment.atoms]
            for atom in atoms:
                node_substructures[atom].append((fragment_id, fragment_type))
        graph.substructures = node_substructures
        return graph


class PSM(BaseTransform):
    def __init__(self, vocab: str) -> None:
        self.vocab = vocab
        self.tokenizer = Tokenizer(self.vocab)
        self.vocab_size = len(self.tokenizer.idx2subgraph) - \
            2  # contains two special symbols

    def __call__(self, graph):
        subgraph_mol = self.tokenizer(graph.mol)
        node_substructures = [[] for _ in range(graph.num_nodes)]
        for fragment_id, fragment in enumerate(subgraph_mol.nodes):
            atom_mapping = subgraph_mol.get_node(fragment).get_atom_mapping()
            atom_ids = list(atom_mapping.keys())
            fragment_type = self.tokenizer.subgraph2idx[subgraph_mol.get_node(
                fragment).smiles]
            for atom in atom_ids:
                node_substructures[atom].append((fragment_id, fragment_type))
        graph.substructures = node_substructures
        return graph


class Rings(BaseTransform):
    def __init__(self, vocab_size=15, merge_exo_double_bonds=False) -> None:
        """Initialize ring based fragmentation that finds a set of short Rings
        covering all rings of the molecule.

        Parameters
        ----------
        max_vocab_size, optional
            Maximum vocab size, i.e. size of the longest ring - 2, by default 15
        """
        self.max_vocab_size = vocab_size
        self.max_ring_size = vocab_size + 2
        self.merge_exo_double_bonds = merge_exo_double_bonds

    def __call__(self, graph, vocab_offset=0):
        mol = graph.mol
        rings = Chem.GetSymmSSSR(mol)
        node_substructures = [[] for _ in range(graph.num_nodes)]
        fragment_types = []
        fragment_id = 0
        for i in range(len(rings)):
            ring = list(rings[i])
            fragment_types.append([fragment2type["ring"], len(ring)])
            if len(ring) <= self.max_ring_size:
                for atom in ring:
                    fragment_type = len(ring) - 3 + vocab_offset
                    node_substructures[atom].append(
                        (fragment_id, fragment_type))
                fragment_id += 1
            else:
                for atom in ring:
                    fragment_type = self.max_vocab_size - 1  + vocab_offset # max fragment_type number
                    node_substructures[atom].append(
                        (fragment_id, fragment_type))
                fragment_id += 1
        graph.substructures = node_substructures
        if fragment_types:
            graph.fragment_types = torch.tensor(
                fragment_types, dtype=torch.long)
        else:
            graph.fragment_types = torch.empty((0, 2), dtype=torch.long)
        return graph


class RingsEdges(BaseTransform):
    def __init__(self, vocab_size, cut_leafs=False):
        self.max_ring = vocab_size - 1  # one fragment for edges
        self.rings = Rings(self.max_ring)
        self.cut_leafs = cut_leafs

    def __call__(self, graph):
        self.rings(graph)

        # now find edges not in rings
        max_frag_id = max([frag_id for frag_infos in graph.substructures for (
            frag_id, _) in frag_infos], default=-1)
        fragment_id = max_frag_id + 1

        fragment_types = []

        for bond in graph.mol.GetBonds():
            if not bond.IsInRing():
                # add bond as new fragment
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                if self.cut_leafs and (is_leaf(atom1, graph) or is_leaf(atom2, graph)):
                    continue
                fragment_types.append([fragment2type["path"], 2])
                bond_info = (fragment_id, self.max_ring)
                fragment_id += 1
                graph.substructures[atom1].append(bond_info)
                graph.substructures[atom2].append(bond_info)

        graph.fragment_types = torch.concat(
            [graph.fragment_types, torch.tensor(fragment_types, dtype=torch.long)], dim=0)
        return graph


class RingsPaths(BaseTransform):
    def __init__(self, vocab_size=30, max_ring=15, cut_leafs=False):
        self.max_ring = max_ring
        assert (vocab_size > max_ring)
        self.max_path = vocab_size - max_ring
        self.rings = Rings(max_ring)
        self.cut_leafs = cut_leafs

    def get_frag_type(self, type: Literal["ring", "path"], size):
        if type == "ring":
            return size - 3 if size - 3 < self.max_ring else self.max_ring - 1
        else:  # type == "path"
            offset = self.max_ring
            return offset + size - 2 if size - 2 < self.max_path else offset + self.max_path - 1

    def __call__(self, graph):
        # first find rings
        self.rings(graph)

        # now find paths
        max_frag_id = max([frag_id for frag_infos in graph.substructures for (
            frag_id, _) in frag_infos], default=-1)
        fragment_id = max_frag_id + 1

        fragment_types = []

        # find paths
        visited = set()
        for bond in graph.mol.GetBonds():

            if not bond.IsInRing() and bond.GetIdx() not in visited:
                if self.cut_leafs and is_leaf(bond.GetBeginAtomIdx(), graph) and is_leaf(bond.GetEndAtomIdx(), graph):
                    continue
                visited.add(bond.GetIdx())
                in_path = []
                to_do = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                while to_do:
                    next_node = to_do.pop()
                    in_path.append(next_node)
                    neighbors = [neighbor for neighbor in get_neighbors(
                        next_node, graph) if not is_leaf(neighbor, graph) or not self.cut_leafs]
                    if not graph.mol.GetAtomWithIdx(next_node).IsInRing() and not len(neighbors) > 2:
                        # not in ring and not a junction
                        new_neighbors = [
                            neighbor for neighbor in neighbors if neighbor not in in_path]
                        visited.update([graph.mol.GetBondBetweenAtoms(
                            next_node, neighbor).GetIdx() for neighbor in new_neighbors])
                        to_do.update(new_neighbors)

                path_info = (fragment_id, self.get_frag_type(
                    "path", len(in_path)))
                fragment_types.append([fragment2type["path"], len(in_path)])
                fragment_id += 1
                for node_id in in_path:
                    graph.substructures[node_id].append(path_info)

        graph.fragment_types = torch.concat(
            [graph.fragment_types, torch.tensor(fragment_types, dtype=torch.long)], dim=0)
        # #find junctions
        # for node_id in range(graph.num_nodes):
        #     if not graph.mol.GetAtomWithIdx(node_id).IsInRing():
        #         neighbors = get_neighbors(node_id, graph)
        #         if len(neighbors) > 2:
        #             graph.substructures[node_id].append((fragment_id, self.get_frag_type("junction", len(neighbors))))
        #             fragment_id += 1
        return graph

class RingsEDBs(BaseTransform):
    """
    Create one fragment per ring and merge exocyclic double-bond partners that are
    directly conjugated to the ring atom (SMARTS-like: [#6R,#16R]=[OR0,SR0,CR0,NR0]).
    Assumes graph.mol is an RDKit Mol.
    """
    def __init__(
        self,
        vocab_size: int = 15,
        merge_policy: str = "sp2_or_aromatic",  # "all" | "aromatic_only" | "sp2_or_aromatic"
        allow_exo_C: bool = True,               # include exocyclic CR0 as a partner
        ring_type_id: int = 0,                  # stored in graph.fragment_types as [type_id, size]
    ) -> None:
        self.max_vocab_size = vocab_size
        self.max_ring_size  = vocab_size + 2
        self.merge_policy   = merge_policy
        self.allow_exo_C    = allow_exo_C
        self.ring_type_id   = ring_type_id

    def __call__(self, graph, vocab_offset: int = 0):
        mol: Chem.Mol = graph.mol
        rings = Chem.GetSymmSSSR(mol)

        n = int(graph.num_nodes)
        node_substructures: List[List[List[int]]] = getattr(graph, "substructures", None)
        if not node_substructures or len(node_substructures) != n:
            node_substructures = [[] for _ in range(n)]

        fragment_types: List[List[int]] = []

        # starting fragment id = 1 + max existing id (or 0 if none)
        best = -1
        for infos in getattr(graph, "substructures", []) or []:
            for fid, _ in infos:
                if fid > best:
                    best = fid
        fragment_id = best + 1

        # precompute exocyclic partners per ring atom (DOUBLE bonds not in rings)
        exo_by_ring_atom = {}
        is_ring = [a.IsInRing() for a in mol.GetAtoms()]
        right_ok: Set[int] = {7, 8, 16} | ({6} if self.allow_exo_C else set())  # N,O,S,(C)

        for bond in mol.GetBonds():
            if bond.GetBondType() != rdchem.BondType.DOUBLE or bond.IsInRing():
                continue

            a = bond.GetBeginAtom(); b = bond.GetEndAtom()
            ai = a.GetIdx();        bi = b.GetIdx()

            # merge-policy gate on the ring atom
            if self.merge_policy == "all":
                ok_a = ok_b = True
            elif self.merge_policy == "aromatic_only":
                ok_a = a.GetIsAromatic(); ok_b = b.GetIsAromatic()
            else:  # "sp2_or_aromatic"
                ha = a.GetHybridization(); hb = b.GetHybridization()
                ok_a = a.GetIsAromatic() or ha in (rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP)
                ok_b = b.GetIsAromatic() or hb in (rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP)

            # side A ring, side B non-ring, ring atom is C or S; partner is O/N/S/(C if allowed)
            if is_ring[ai] and not is_ring[bi]:
                if a.GetAtomicNum() in (6, 16) and b.GetAtomicNum() in right_ok and ok_a:
                    exo_by_ring_atom.setdefault(ai, set()).add(bi)

            # side B ring, side A non-ring
            if is_ring[bi] and not is_ring[ai]:
                if b.GetAtomicNum() in (6, 16) and a.GetAtomicNum() in right_ok and ok_b:
                    exo_by_ring_atom.setdefault(bi, set()).add(ai)

        # build fragments: one per ring, then union its EDB partners
        for r in rings:
            ring_atoms = list(r)

            # record (type_id, size)
            fragment_types.append([self.ring_type_id, len(ring_atoms)])

            # embedding/type index from ring size (capped)
            if len(ring_atoms) <= self.max_ring_size:
                fragment_type = len(ring_atoms) - 3 + vocab_offset
            else:
                fragment_type = self.max_vocab_size - 1 + vocab_offset

            # attach ring atoms
            for aidx in ring_atoms:
                node_substructures[aidx].append((fragment_id, fragment_type))

            # merge exocyclic double-bond partners
            added: Set[int] = set()
            for aidx in ring_atoms:
                for other in exo_by_ring_atom.get(aidx, ()):
                    if other not in added:
                        node_substructures[other].append((fragment_id, fragment_type))
                        added.add(other)

            fragment_id += 1

        graph.substructures = node_substructures
        graph.fragment_types = (
            torch.tensor(fragment_types, dtype=torch.long)
            if fragment_types else torch.empty((0, 2), dtype=torch.long)
        )
        return graph


class ErtlEFGs(BaseTransform):
    def __init__(self, vocab: List[str], use_oov_bucket: bool = False):
        self.use_oov_bucket = bool(use_oov_bucket)
        self.vocab, self.oov_idx = _prepare_optional_oov_vocab(
            vocab, ERTL_EFG_OOV_TOKEN, self.use_oov_bucket
        )
        self.vocab_size = len(self.vocab)
        self.frag_to_idx = {frag: i for i, frag in enumerate(self.vocab)}
        print("ErtlEFGs vocab size:", self.vocab_size)

    def __call__(self, graph, vocab_offset: int = 0, min_size: int = 1):
        mol = graph.mol
        node_substructures = [[] for _ in range(graph.num_nodes)]
        fragment_types = []

        fragment_id = 1 + max(
            (fid for infos in getattr(graph, "substructures", []) for (fid, _) in infos),
            default=-1,
        )

        try:
            _, fgs, psmis, _ = ertl_efg.get_dec_fgs(mol)
        except Exception:
            if not hasattr(graph, "fragment_types"):
                graph.fragment_types = torch.empty((0, 2), dtype=torch.long)
            if not hasattr(graph, "substructures"):
                graph.substructures = node_substructures
            return graph

        for atoms, ps in zip(fgs, psmis):
            if len(atoms) < min_size:
                continue

            idx = self.frag_to_idx.get(ps)
            if idx is None:
                if not self.use_oov_bucket:
                    continue
                idx = self.oov_idx

            frag_type = idx + vocab_offset
            for a in atoms:
                node_substructures[a].append((fragment_id, frag_type))

            fragment_types.append([fragment2type["efg"], len(atoms)])
            fragment_id += 1

        if hasattr(graph, "substructures"):
            for i, sub in enumerate(node_substructures):
                graph.substructures[i].extend(sub)
        else:
            graph.substructures = node_substructures

        ft = (
            torch.tensor(fragment_types, dtype=torch.long)
            if fragment_types else torch.empty((0, 2), dtype=torch.long)
        )
        if hasattr(graph, "fragment_types"):
            graph.fragment_types = torch.cat([graph.fragment_types, ft], dim=0)
        else:
            graph.fragment_types = ft

        return graph


class RingsErtlEFGs(BaseTransform):
    def __init__(self, vocab: List[str], max_ring: int = 15, use_oov_bucket: bool = False):
        self.max_ring = max_ring
        self.rings = Rings(self.max_ring)
        self.efgs = ErtlEFGs(vocab, use_oov_bucket=use_oov_bucket)
        vocab_size = self.efgs.vocab_size + max_ring
        assert vocab_size > max_ring

    def __call__(self, graph, min_size: int = 1):
        self.rings(graph, vocab_offset=self.efgs.vocab_size)
        self.efgs(graph, min_size=min_size)
        return graph

class HiFrAMes(BaseTransform):
    """
    HiFrAMes fragmentation integrated into the existing fragment/substructure path.

    Labeling modes
    --------------
    - family:      one label each for chain, linker, ring
    - family_size: separate capped size bins for chain/linker/ring

    Notes
    -----
    - This first integration intentionally reuses the existing HiFrAMes
      fragmentation routine and ignores reduced-graph outputs.
    - graph.fragment_types stays compatible with the existing ordinal/path-like
      metadata convention: rings use fragment2type['ring'], chains/linkers use
      fragment2type['path'].
    """

    FAMILY_LABEL_IDS = {
        'chain': 0,
        'linker': 1,
        'ring': 2,
        'ring_core': 2,
    }

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        label_mode: str = 'family_size',
        size_bins: int = 15,
        chain_size_bins: Optional[int] = None,
        linker_size_bins: Optional[int] = None,
        ring_size_bins: Optional[int] = None,
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
    ) -> None:
        if hiframes.fragment_molecule is None:
            raise ImportError(
                'HiFrAMes fragmentation requested but fragment_molecule could not '
                'be imported from data.fragmentations.hiframes or hiframes.'
            )

        self.label_mode = normalize_hiframes_label_mode(label_mode)
        self.chain_size_bins = int(chain_size_bins if chain_size_bins is not None else size_bins)
        self.linker_size_bins = int(linker_size_bins if linker_size_bins is not None else size_bins)
        self.ring_size_bins = int(ring_size_bins if ring_size_bins is not None else size_bins)
        if min(self.chain_size_bins, self.linker_size_bins, self.ring_size_bins) <= 0:
            raise RuntimeError(
                'HiFrAMes size-bin counts must all be positive. '
                f'Got chain={self.chain_size_bins}, linker={self.linker_size_bins}, ring={self.ring_size_bins}.'
            )

        inferred_vocab_size = (
            3
            if self.label_mode == 'family'
            else self.chain_size_bins + self.linker_size_bins + self.ring_size_bins
        )
        provided_vocab_size = None if vocab_size is None else int(vocab_size)
        # data.py historically subtracts one slot before constructing tree-mode HLGs,
        # because HigherLevelGraph later adds the virtual junction slot back. Accept
        # both shapes here, but always keep the fragment label space at its true size.
        if provided_vocab_size is not None and provided_vocab_size not in {
            inferred_vocab_size,
            inferred_vocab_size - 1,
        }:
            raise RuntimeError(
                'HiFrAMes vocab_size mismatch: config/model expects '
                f'{provided_vocab_size}, but label_mode={self.label_mode!r} implies '
                f'{inferred_vocab_size} (or {inferred_vocab_size - 1} before a tree-mode '
                'junction slot is added back). Use infer_hiframes_vocab_size(...) or '
                'update the config vocab_size to match the chosen label mode/bins.'
            )
        self.vocab_size = inferred_vocab_size

        self.add_explicit_hs = bool(add_explicit_hs)
        self.keep_terminal_exocyclic_db = bool(keep_terminal_exocyclic_db)
        self.keep_chain_exocyclic_db = bool(keep_chain_exocyclic_db)
        self.preserve_ring_linkage_db = bool(preserve_ring_linkage_db)
        self.keep_terminal_linker_exocyclic_db = bool(keep_terminal_linker_exocyclic_db)
        self.keep_chain_linker_exocyclic_db = bool(keep_chain_linker_exocyclic_db)
        self.set_atommap_to_orig_idx = bool(set_atommap_to_orig_idx)
        self.include_attachment_atoms_in_chain_fragments = bool(include_attachment_atoms_in_chain_fragments)
        self.include_attachment_atoms_in_linker_fragments = bool(include_attachment_atoms_in_linker_fragments)
        self.merge_chain_branches_at_carbon_attachments = bool(merge_chain_branches_at_carbon_attachments)
        self.merge_chain_branches_at_hetero_attachments = bool(merge_chain_branches_at_hetero_attachments)
        self.merge_bond_linkers_at_carbon_attachments = bool(merge_bond_linkers_at_carbon_attachments)
        self.merge_bond_linkers_at_hetero_attachments = bool(merge_bond_linkers_at_hetero_attachments)
        self.return_ring_core = bool(return_ring_core)

    @staticmethod
    def _unwrap_fragments(result):
        if isinstance(result, list):
            return result
        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, list):
                    return item
        raise RuntimeError(
            'Unexpected HiFrAMes fragmentation result type: '
            f'{type(result)!r}. Expected a fragment list or tuple containing one.'
        )

    @staticmethod
    def _fragment_family_type(kind: str) -> int:
        if kind in {'ring', 'ring_core'}:
            return fragment2type['ring']
        if kind in {'chain', 'linker'}:
            return fragment2type['path']
        raise RuntimeError(f'Unsupported HiFrAMes fragment kind: {kind!r}')

    @staticmethod
    def _cap_bin(size: int, num_bins: int) -> int:
        size = max(int(size), 1)
        return min(size, num_bins) - 1

    def _fragment_label_id(self, kind: str, size: int) -> int:
        if self.label_mode == 'family':
            return self.FAMILY_LABEL_IDS[kind]

        if kind == 'chain':
            return self._cap_bin(size, self.chain_size_bins)
        if kind == 'linker':
            return self.chain_size_bins + self._cap_bin(size, self.linker_size_bins)
        if kind in {'ring', 'ring_core'}:
            return self.chain_size_bins + self.linker_size_bins + self._cap_bin(size, self.ring_size_bins)
        raise RuntimeError(f'Unsupported HiFrAMes fragment kind: {kind!r}')

    def __call__(self, graph):
        result = hiframes.fragment_molecule(
            graph.mol,
            add_explicit_hs=self.add_explicit_hs,
            keep_terminal_exocyclic_db=self.keep_terminal_exocyclic_db,
            keep_chain_exocyclic_db=self.keep_chain_exocyclic_db,
            preserve_ring_linkage_db=self.preserve_ring_linkage_db,
            keep_terminal_linker_exocyclic_db=self.keep_terminal_linker_exocyclic_db,
            keep_chain_linker_exocyclic_db=self.keep_chain_linker_exocyclic_db,
            set_atommap_to_orig_idx=self.set_atommap_to_orig_idx,
            include_attachment_atoms_in_chain_fragments=self.include_attachment_atoms_in_chain_fragments,
            include_attachment_atoms_in_linker_fragments=self.include_attachment_atoms_in_linker_fragments,
            merge_chain_branches_at_carbon_attachments=self.merge_chain_branches_at_carbon_attachments,
            merge_chain_branches_at_hetero_attachments=self.merge_chain_branches_at_hetero_attachments,
            merge_bond_linkers_at_carbon_attachments=self.merge_bond_linkers_at_carbon_attachments,
            merge_bond_linkers_at_hetero_attachments=self.merge_bond_linkers_at_hetero_attachments,
            return_ring_core=self.return_ring_core,
            return_reduced_graph=False,
        )
        hiframes_frags = self._unwrap_fragments(result)

        node_substructures = [[] for _ in range(graph.num_nodes)]
        fragment_types = []
        fragment_id = 1 + max(
            (fid for infos in getattr(graph, 'substructures', []) for (fid, _) in infos),
            default=-1,
        )

        for frag in hiframes_frags:
            kind = str(frag.kind)
            atom_ids = tuple(sorted({int(a) for a in frag.orig_atom_indices}))
            if not atom_ids:
                continue

            frag_type_id = self._fragment_label_id(kind, len(atom_ids))
            if frag_type_id < 0 or frag_type_id >= self.vocab_size:
                raise RuntimeError(
                    f'Computed HiFrAMes fragment label id {frag_type_id} is out of range '
                    f'for vocab_size={self.vocab_size}.'
                )

            for atom_id in atom_ids:
                node_substructures[atom_id].append((fragment_id, frag_type_id))

            fragment_types.append([self._fragment_family_type(kind), len(atom_ids)])
            fragment_id += 1

        if hasattr(graph, 'substructures'):
            for i, substructure in enumerate(node_substructures):
                graph.substructures[i].extend(substructure)
        else:
            graph.substructures = node_substructures

        ft = (
            torch.tensor(fragment_types, dtype=torch.long)
            if fragment_types else torch.empty((0, 2), dtype=torch.long)
        )
        if hasattr(graph, 'fragment_types'):
            graph.fragment_types = torch.cat([graph.fragment_types, ft], dim=0)
        else:
            graph.fragment_types = ft

        return graph


class NoFragmentation(BaseTransform):
    def __init__(self, vocab_size: int = 0):
        self.vocab_size = int(vocab_size)

    def __call__(self, graph):
        n = int(graph.num_nodes)

        graph.substructures = [[] for _ in range(n)]

        return graph


class EnsureFragmentPlaceholders(BaseTransform):
    def __init__(self, vocab_size: int = 0):
        self.vocab_size = int(vocab_size)

    def __call__(self, graph):
        if not hasattr(graph, "fragments"):
            graph.fragments = torch.zeros((0, self.vocab_size), dtype=torch.float)
        if not hasattr(graph, "low_high_edge_index"):
            graph.low_high_edge_index = torch.zeros((2, 0), dtype=torch.long)
        if not hasattr(graph, "fragments_edge_index"):
            graph.fragments_edge_index = torch.zeros((2, 0), dtype=torch.long)
        return graph


class ConstantOneHotFragmentLabels(BaseTransform):
    """
    Overwrite all fragment-node labels with the same one-hot label.

    This is intended as a strong label-removal ablation for fragment/higher-level
    graph models while preserving the fragment construction, atom-fragment
    incidence, and higher-level graph topology.
    """

    def __init__(self, constant_id: int = 0):
        self.constant_id = int(constant_id)

    def __call__(self, graph):
        if not hasattr(graph, "fragments"):
            raise RuntimeError(
                "ConstantOneHotFragmentLabels requires graph.fragments to exist. "
                "Apply it after FragmentRepresentation or HigherLevelGraph."
            )

        frag_rep = torch.as_tensor(graph.fragments)
        if frag_rep.dim() != 2:
            raise RuntimeError(
                "Expected graph.fragments to be a 2D tensor of shape "
                "(num_fragments, vocab_size)."
            )

        num_frags, vocab_size = frag_rep.size()
        if vocab_size == 0:
            if num_frags == 0:
                return graph
            raise RuntimeError(
                "Cannot apply ConstantOneHotFragmentLabels when graph.fragments "
                "has zero columns."
            )

        if self.constant_id < 0 or self.constant_id >= vocab_size:
            raise RuntimeError(
                f"constant_id={self.constant_id} is out of range for "
                f"fragment vocab width {vocab_size}."
            )

        out = torch.zeros_like(frag_rep)
        if num_frags > 0:
            out[:, self.constant_id] = 1
        graph.fragments = out
        return graph


class NodeFeature(BaseTransform):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, graph):
        node_features = torch.zeros(graph.num_nodes, self.vocab_size)
        for atom_id, fragments in enumerate(graph.substructures):
            for (_, fragment_type) in fragments:
                node_features[atom_id, fragment_type] += 1
        graph.x = torch.cat([graph.x, node_features], dim=1)
        return graph


def _substructure_fragment_type_map(graph):
    return dict(
        frag_info
        for frag_infos in getattr(graph, "substructures", [])
        for frag_info in frag_infos
        if frag_info
    )


def _max_substructure_frag_id(graph):
    return max(
        [frag_id for frag_infos in getattr(graph, "substructures", []) for (frag_id, _) in frag_infos],
        default=-1,
    )


class GraphLevelFeature(BaseTransform):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, graph):
        graph_features = torch.zeros(self.vocab_size)
        frag_type_ids = FragmentRepresentation._normalized_fragment_type_ids(graph)
        for frag_type_id in frag_type_ids.tolist():
            graph_features[frag_type_id] += 1
        node_counts = torch.sum(graph.x, dim=0)
        graph.motif_counts = torch.unsqueeze(torch.concat(
            [graph_features, node_counts], dim=0), dim=0)
        return graph


class FragmentData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'fragments_edge_index':
            return torch.tensor([[self.x.size(0)], [self.fragments.size(0)]])
        elif key == "higher_edge_index":
            return self.fragments.size(0)
        elif key == "low_high_edge_index":
            return torch.tensor([[self.edge_index.size(1)], [self.fragments.size(0)]])
        elif key == "join_node_index":
            return torch.tensor([[self.higher_edge_index.size(1)], [self.x.size(0)]])
        elif key == "join_edge_index":
            return torch.tensor([[self.higher_edge_index.size(1)], [self.edge_index.size(1)]])
        return super().__inc__(key, value, *args, **kwargs)


class FragmentRepresentation(BaseTransform):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    @staticmethod
    def _normalized_fragment_type_ids(graph):
        existing = getattr(graph, "fragment_type_ids", None)
        if existing is None:
            frag_type_ids = torch.empty((0,), dtype=torch.long)
        else:
            frag_type_ids = torch.as_tensor(existing, dtype=torch.long).flatten()

        frag_id_to_type = _substructure_fragment_type_map(graph)
        max_sub_frag_id = _max_substructure_frag_id(graph)
        needed = max(frag_type_ids.numel(), max_sub_frag_id + 1)

        if needed == 0:
            return torch.empty((0,), dtype=torch.long)

        normalized = torch.full((needed,), -1, dtype=torch.long)
        if frag_type_ids.numel() > 0:
            normalized[:frag_type_ids.numel()] = frag_type_ids

        for frag_id, frag_type in frag_id_to_type.items():
            normalized[frag_id] = int(frag_type)

        missing = (normalized < 0).nonzero(as_tuple=False).flatten().tolist()
        if missing:
            raise RuntimeError(
                "Missing fragment vocabulary ids for fragment ids: "
                f"{missing}. Provide graph.fragment_type_ids for virtual fragments."
            )

        return normalized

    def __call__(self, graph):
        frag_type_ids = self._normalized_fragment_type_ids(graph)
        graph.fragment_type_ids = frag_type_ids

        frag_representation = torch.zeros(frag_type_ids.numel(), self.vocab_size)
        if frag_type_ids.numel() > 0:
            frag_representation[
                torch.arange(frag_type_ids.numel(), dtype=torch.long),
                frag_type_ids,
            ] = 1
        graph.fragments = frag_representation

        edges = [
            [node_id, frag_id]
            for node_id, frag_infos in enumerate(getattr(graph, "substructures", []))
            for (frag_id, _) in frag_infos
        ]
        if not edges:
            graph.fragments_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            graph.fragments_edge_index = torch.tensor(
                edges, dtype=torch.long).T.contiguous()

        # get (low level) edges that are part of a fragment
        low_high_edges = []
        for edge_id, (node_a, node_b) in enumerate(graph.edge_index.T):
            overlapping_fragments = set(graph.substructures[node_a]).intersection(
                set(graph.substructures[node_b]))
            for (frag_id, _) in overlapping_fragments:
                low_high_edges.append([edge_id, frag_id])
        if not low_high_edges:
            graph.low_high_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            graph.low_high_edge_index = torch.tensor(
                low_high_edges, dtype=torch.long).T.contiguous()

        return FragmentData(**{k: v for k, v in graph})


class HigherLevelGraph(BaseTransform):
    """
    edge_policy:
        "overlap"          : fragments related by shared atom membership
        "adjacent"         : fragments related across an original-graph bond
        "adjacent_overlap" : union of overlap and adjacent events
        "distance"         : fragments related if atom-set min distance <= max_distance
        "distance_overlap" : union of overlap and distance events
        "complete"         : all fragments related
        "none"             : no higher-level relations

    mode:
        "node" : add direct fragment-fragment edges
        "tree" : if an event involves >2 fragments, insert a junction fragment node

    Notes
    -----
    - overlap+node and overlap+tree preserve the legacy behavior by default.
    - generalized tree mode can introduce virtual junction fragment nodes that
      live in graph.fragments / graph.fragment_type_ids without requiring atom
      memberships in graph.substructures.
    """

    def __init__(
        self,
        vocab_size,
        edge_policy: Literal[
            "overlap",
            "adjacent",
            "distance",
            "complete",
            "adjacent_overlap",
            "distance_overlap",
            "none",
        ] = "overlap",
        mode: Literal["node", "tree"] = "tree",
        higher_edge_features: bool = False,
        max_distance: int = 2,
        neighbor: Literal["node", "tree", None] = None,
        legacy_overlap_tree: bool = True,
    ):
        self.vocab_size = vocab_size
        self.edge_policy = edge_policy
        self.mode = mode
        self.higher_edge_features = higher_edge_features
        self.max_distance = int(max_distance)
        self.legacy_overlap_tree = legacy_overlap_tree

        if self.edge_policy in {"node", "tree"} and neighbor is None:
            # Backward-compatible positional form:
            # HigherLevelGraph(vocab_size, "tree")
            self.mode = self.edge_policy
            self.edge_policy = "overlap"

        if neighbor is not None:
            self.edge_policy = "overlap"
            self.mode = neighbor

        self.edge_policy, self.max_distance = self._normalize_edge_policy_alias(
            self.edge_policy,
            self.max_distance,
        )

        self.frag_rep = FragmentRepresentation(
            vocab_size + 1 if self.mode == "tree" else vocab_size
        )

    @staticmethod
    def _unique_frag_ids(frag_infos):
        return sorted({frag_id for frag_id, _ in frag_infos})

    @staticmethod
    def _empty_fragment_metadata():
        return torch.empty((0, 2), dtype=torch.long)

    @staticmethod

    def _normalize_edge_policy_alias(edge_policy, max_distance):
        """
        Normalize short policy aliases while preserving current semantics.

        Supported aliases:
        ovl        -> overlap
        adj        -> adjacent
        full       -> complete
        dist2      -> distance   + force max_distance=2

        Optional shorthand for new policies:
        adj_ovl    -> adjacent_overlap
        adjovl     -> adjacent_overlap
        dist_ovl   -> distance_overlap
        distovl    -> distance_overlap
        dist2_ovl  -> distance_overlap + force max_distance=2
        dist2ovl   -> distance_overlap + force max_distance=2
        """
        policy = str(edge_policy).strip().lower()

        alias_map = {
            "ovl": "overlap",
            "adj": "adjacent",
            "full": "complete",
            "adj_ovl": "adjacent_overlap",
            "adjovl": "adjacent_overlap",
            "dist_ovl": "distance_overlap",
            "distovl": "distance_overlap",
        }

        if policy == "dist2":
            return "distance", 2

        if policy in {"dist2_ovl", "dist2ovl"}:
            return "distance_overlap", 2

        return alias_map.get(policy, policy), max_distance

    def _ensure_fragment_metadata(self, graph):
        frag_type_ids = FragmentRepresentation._normalized_fragment_type_ids(graph)
        graph.fragment_type_ids = frag_type_ids

        fragment_types = getattr(graph, "fragment_types", None)
        if fragment_types is None:
            fragment_types = self._empty_fragment_metadata()
        else:
            fragment_types = torch.as_tensor(fragment_types, dtype=torch.long)
            if fragment_types.numel() == 0:
                fragment_types = self._empty_fragment_metadata()
            elif fragment_types.dim() == 1:
                fragment_types = fragment_types.unsqueeze(1)
            if fragment_types.size(1) == 1:
                fragment_types = torch.cat(
                    [fragment_types, torch.zeros((fragment_types.size(0), 1), dtype=torch.long)],
                    dim=1,
                )

        needed = graph.fragment_type_ids.numel()
        if fragment_types.size(0) < needed:
            extra = torch.zeros((needed - fragment_types.size(0), 2), dtype=torch.long)
            fragment_types = torch.cat([fragment_types, extra], dim=0)

        graph.fragment_types = fragment_types
        return graph

    def _append_virtual_junction(self, graph, arity: int):
        graph = self._ensure_fragment_metadata(graph)
        junction_id = int(graph.fragment_type_ids.numel())
        graph.fragment_type_ids = torch.cat(
            [graph.fragment_type_ids, torch.tensor([self.vocab_size], dtype=torch.long)],
            dim=0,
        )
        junction_row = torch.tensor(
            [[fragment2type["junction"], arity]], dtype=torch.long
        )
        graph.fragment_types = torch.cat([graph.fragment_types, junction_row], dim=0)
        return graph, junction_id

    def _event_frag_ids_overlap(self, graph):
        events = []
        for frag_infos in graph.substructures:
            frag_ids = self._unique_frag_ids(frag_infos)
            if len(frag_ids) >= 2:
                events.append(frag_ids)
        return events

    def _event_frag_ids_adjacent(self, graph):
        events = []
        for bond in graph.mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()

            frag_ids_a = {frag_id for frag_id, _ in graph.substructures[a]}
            frag_ids_b = {frag_id for frag_id, _ in graph.substructures[b]}

            active = set()
            for fa in frag_ids_a:
                for fb in frag_ids_b:
                    if fa != fb:
                        active.add(fa)
                        active.add(fb)

            if len(active) >= 2:
                events.append(sorted(active))

        return events

    def _event_frag_ids_distance(self, graph):
        if self.max_distance < 1:
            return []

        dist_mat = Chem.GetDistanceMatrix(graph.mol)
        seen = set()
        events = []

        for a in range(graph.num_nodes):
            frag_ids_a = {frag_id for frag_id, _ in graph.substructures[a]}
            if not frag_ids_a:
                continue

            for b in range(a + 1, graph.num_nodes):
                if dist_mat[a, b] <= self.max_distance:
                    frag_ids_b = {frag_id for frag_id, _ in graph.substructures[b]}
                    event = tuple(sorted(frag_ids_a.union(frag_ids_b)))
                    if len(event) >= 2 and event not in seen:
                        seen.add(event)
                        events.append(list(event))

        return events

    @staticmethod
    def _dedupe_events(events):
        seen = set()
        out = []

        for frag_ids in events:
            event = tuple(sorted(set(frag_ids)))
            if len(event) < 2:
                continue
            if event in seen:
                continue
            seen.add(event)
            out.append(list(event))

        return out

    def _event_frag_ids_adjacent_overlap(self, graph):
        return self._dedupe_events(
            self._event_frag_ids_overlap(graph) +
            self._event_frag_ids_adjacent(graph)
        )

    def _event_frag_ids_distance_overlap(self, graph):
        return self._dedupe_events(
            self._event_frag_ids_overlap(graph) +
            self._event_frag_ids_distance(graph)
        )

    def _event_frag_ids_complete(self, graph):
        frag_ids = sorted({
            frag_id
            for frag_infos in graph.substructures
            for frag_id, _ in frag_infos
        })
        return [frag_ids] if len(frag_ids) >= 2 else []

    def _build_events(self, graph):
        if self.edge_policy == "none":
            return []
        if self.edge_policy == "overlap":
            return self._event_frag_ids_overlap(graph)
        if self.edge_policy == "adjacent":
            return self._event_frag_ids_adjacent(graph)
        if self.edge_policy == "adjacent_overlap":
            return self._event_frag_ids_adjacent_overlap(graph)
        if self.edge_policy == "distance":
            return self._event_frag_ids_distance(graph)
        if self.edge_policy == "distance_overlap":
            return self._event_frag_ids_distance_overlap(graph)
        if self.edge_policy == "complete":
            return self._event_frag_ids_complete(graph)
        raise RuntimeError(f"Unsupported edge policy: {self.edge_policy}")

    def _legacy_overlap_node(self, graph):
        higher_edges = []
        for frag_infos in graph.substructures:
            frag_ids = [frag_id for frag_id, _ in frag_infos]
            if len(frag_ids) >= 2:
                for frag1, frag2 in permutations(frag_ids, 2):
                    if [frag1, frag2] not in higher_edges:
                        higher_edges.append([frag1, frag2])
        return graph, higher_edges

    def _legacy_overlap_tree(self, graph):
        higher_edges = []
        fragment_types = []

        max_frag_id = _max_substructure_frag_id(graph)
        fragment_id = max_frag_id + 1

        for node_id, frag_infos in enumerate(graph.substructures):
            frag_ids = [frag_id for frag_id, _ in frag_infos]
            if len(frag_ids) == 2:
                for frag1, frag2 in permutations(frag_ids, 2):
                    if [frag1, frag2] not in higher_edges:
                        higher_edges.append([frag1, frag2])
            elif len(frag_ids) > 2:
                junction_id = fragment_id
                graph.substructures[node_id].append((junction_id, self.vocab_size))
                fragment_types.append([fragment2type["junction"], len(frag_ids)])
                fragment_id += 1

                for frag_id in frag_ids:
                    higher_edges.append([frag_id, junction_id])
                    higher_edges.append([junction_id, frag_id])

        if fragment_types:
            graph = self._ensure_fragment_metadata(graph)
            graph.fragment_types = torch.cat(
                [graph.fragment_types, torch.tensor(fragment_types, dtype=torch.long)],
                dim=0,
            )

        return graph, higher_edges

    def _generalized_from_events(self, graph, events):
        undirected_pairs = set()
        directed_edges = []

        if self.mode == "node":
            for frag_ids in events:
                if len(frag_ids) >= 2:
                    for a, b in combinations(sorted(set(frag_ids)), 2):
                        undirected_pairs.add((a, b))
        elif self.mode == "tree":
            for frag_ids in events:
                frag_ids = sorted(set(frag_ids))
                if len(frag_ids) == 2:
                    undirected_pairs.add((frag_ids[0], frag_ids[1]))
                elif len(frag_ids) > 2:
                    graph, junction_id = self._append_virtual_junction(graph, len(frag_ids))
                    for frag_id in frag_ids:
                        directed_edges.append([frag_id, junction_id])
                        directed_edges.append([junction_id, frag_id])
        else:
            raise RuntimeError(f"Unsupported mode: {self.mode}")

        for a, b in sorted(undirected_pairs):
            directed_edges.append([a, b])
            directed_edges.append([b, a])

        return graph, directed_edges

    def __call__(self, graph):
        if self.edge_policy == "overlap" and self.mode == "node":
            graph, higher_edges = self._legacy_overlap_node(graph)
        elif self.edge_policy == "overlap" and self.mode == "tree" and self.legacy_overlap_tree:
            graph, higher_edges = self._legacy_overlap_tree(graph)
        else:
            events = self._build_events(graph)
            graph, higher_edges = self._generalized_from_events(graph, events)

        graph = self._ensure_fragment_metadata(graph)

        graph = self.frag_rep(graph)

        if not higher_edges:
            graph.higher_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            graph.higher_edge_index = torch.tensor(
                higher_edges, dtype=torch.long
            ).T.contiguous()

        if self.higher_edge_features:
            join_nodes_list = []
            join_edges_list = []
            edge_types_list = []

            for higher_edge_id, higher_edge in enumerate(graph.higher_edge_index.T):
                nodes1 = graph.fragments_edge_index[
                    0, graph.fragments_edge_index[1, :] == higher_edge[0]
                ]
                nodes2 = graph.fragments_edge_index[
                    0, graph.fragments_edge_index[1, :] == higher_edge[1]
                ]

                join_nodes = [node1 for node1 in nodes1 if node1 in nodes2]
                join_edges = [
                    edge_id for edge_id, (node_a, node_b) in enumerate(graph.edge_index.T)
                    if node_a in join_nodes and node_b in join_nodes
                ]

                join_nodes_list += [[higher_edge_id, join_node] for join_node in join_nodes]
                join_edges_list += [[higher_edge_id, join_edge] for join_edge in join_edges]

                edge_types_list.append(1 if join_edges else 0)

            if not join_nodes_list:
                graph.join_node_index = torch.empty((2, 0), dtype=torch.long)
            else:
                graph.join_node_index = torch.tensor(
                    join_nodes_list, dtype=torch.long
                ).T.contiguous()

            if not join_edges_list:
                graph.join_edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                graph.join_edge_index = torch.tensor(
                    join_edges_list, dtype=torch.long
                ).T.contiguous()

            if not edge_types_list:
                graph.higher_edge_types = torch.empty((0), dtype=torch.long)
            else:
                graph.higher_edge_types = torch.tensor(
                    edge_types_list, dtype=torch.long
                )

        return graph
