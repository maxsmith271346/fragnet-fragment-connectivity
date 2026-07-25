# FragNet Fragment Connectivity

This repository contains an independent derivative of the FragNet implementation
developed by Wollschläger et al. for **“Expressivity and Generalization:
Fragment-Biases for Molecular GNNs.”**

The code is based on the `view/publication` branch of
[`KemperNiklas/FragNet`](https://github.com/KemperNiklas/FragNet), at upstream
commit `3f69708ab7f99db74f7e4b7b704a8293d3da83c4`. It is not an official FragNet release.

This derivative adds the molecular fragmentation schemes and fragment-level
higher-level graph (HLG) connectivity policies evaluated in:

> **Context-Dependent Effects of Fragment-Level Connectivity in
> Substructure-Aware Molecular Graph Neural Networks**

## Included extensions

The retained code supports the manuscript's:

- Rings, RingsEDBs, RingsPaths, FGs, RingsFGs, HiFrAMes, BRICS, and
  no-fragmentation variants;
- no-HLG-edge, overlap, adjacency, and 2-hop adjacency policies, where
  applicable;
- ZINC-full training workflow for penalized logP;
- lightweight run metadata and logging used for the HPRC experiments.

Manuscript-facing `FGs` and `RingsFGs` correspond to the internal identifiers
`ErtlEFGs` and `RingsErtlEFGs`, respectively.

## Repository organization

The repository intentionally follows the upstream FragNet layout. It contains
the runtime code required for the reported model variants, one consolidated
paper experiment matrix, two readable example configurations, and one TAMU
HPRC Slurm launcher. Exploratory notebooks, intermediate analysis scripts,
terminal logs, generated reports, checkpoints, and historical experiment files
are not included.

## Environment

```bash
conda env create -f environment.yml
conda activate fragNet
```

## Experiment matrix

`experiment/ZINC_full_all.yaml` contains the 16 retained model variants and
seeds 23, 24, and 25, for 48 runs total. It is generated from the historical
configurations that were present before the publication cleanup.

```bash
python run_hprc_experiment.py --config experiment/ZINC_full_all.yaml --list
python run_hprc_experiment.py --config experiment/ZINC_full_all.yaml --index 0 --print-config
```

## TAMU HPRC

The Slurm array uses the same general HPRC execution pattern and resources as
the original experiments: one GPU, 64 GB memory, four CPU cores, the GPU
partition, a three-day wall time, and at most four simultaneous jobs.

It expects the repository at `$SCRATCH/fragnet-fragment-connectivity` unless
`FRAGNET_PROJECT_ROOT` is set.

```bash
sbatch experiment/run_ZINC_full_all.slurm
```

## Relationship to upstream FragNet

The upstream implementation provides the core fragment-biased molecular GNN
architecture on which this derivative is based. Users should cite the original
FragNet work as well as the accompanying fragmentation/connectivity study.

### Original FragNet work

Wollschläger, T.; Kemper, N.; Hetzel, L.; Sommer, J.; Günnemann, S.
“Expressivity and Generalization: Fragment-Biases for Molecular GNNs.”
*Proceedings of the 41st International Conference on Machine Learning*,
PMLR **235**, 53113–53139 (2024).

- Paper: https://proceedings.mlr.press/v235/wollschlager24a.html
- Upstream code: https://github.com/KemperNiklas/FragNet

### This derivative and accompanying study

Smith, M. A.; Yu, Y.; Liu, J.-C.
“Context-Dependent Effects of Fragment-Level Connectivity in
Substructure-Aware Molecular Graph Neural Networks.”
Publication information will be added when available.

## Citation

When using this repository, cite both the original FragNet publication and the
accompanying study above.
