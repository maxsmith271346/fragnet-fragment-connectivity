import os.path
import fcntl
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch.nn.functional import one_hot
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (BaseTransform, Compose, OneHotDegree,
                                        add_positional_encoding)
from torch_geometric.utils import degree

import data.fragmentations.fragmentations as frag
from config import DATASET_ROOT, VOCAB_ROOT
from data.count_substructures import CountSubstructures
from data.graph_to_mol import (  # I have no idea why but this import has to be in the beginning (I know that this is probably a bad sign...)
    OGB_Graph_Add_Mol_By_Smiles, ZINC_Graph_Add_Mol)
from data.lrgb import lrgb
from data.plogp import FixPlogP, LogP
import hashlib
import json

def _sanitize_path_token(x) -> str:
    return (
        str(x)
        .replace(" ", "")
        .replace("{", "")
        .replace("}", "")
        .replace(":", "-")
        .replace("/", "-")
    )


def _build_config_name(
    fragmentation_method,
    encoding,
    remove_node_features,
    one_hot_edge_features,
    one_hot_degree,
    one_hot_node_features,
    higher_edge_features,
    max_raw_len: int = 120,
) -> str:
    if fragmentation_method:
        raw_frag_method_str = "_".join(
            _sanitize_path_token(x)
            if not isinstance(x, dict)
            else "_".join(
                f"{_sanitize_path_token(k)}-{_sanitize_path_token(v)}"
                for k, v in x.items()
            )
            for x in fragmentation_method
        )
    else:
        raw_frag_method_str = "no_fragmentation"

    raw_name = (
        f"{raw_frag_method_str}_{encoding}_{remove_node_features}_"
        f"{one_hot_edge_features}_{one_hot_degree}_{one_hot_node_features}_"
        f"{higher_edge_features}"
    )
    raw_name = _sanitize_path_token(raw_name)

    # Preserve old readable names when they are already short enough.
    if len(raw_name) <= max_raw_len:
        return raw_name

    payload = {
        "fragmentation_method": fragmentation_method,
        "encoding": encoding,
        "remove_node_features": remove_node_features,
        "one_hot_edge_features": one_hot_edge_features,
        "one_hot_degree": one_hot_degree,
        "one_hot_node_features": one_hot_node_features,
        "higher_edge_features": higher_edge_features,
    }
    digest = hashlib.sha1(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    frag_tag = (
        _sanitize_path_token(fragmentation_method[0])
        if fragmentation_method else "nofrag"
    )
    usage_tag = (
        _sanitize_path_token(fragmentation_method[1])
        if fragmentation_method and len(fragmentation_method) > 1 else "plain"
    )

    return f"{frag_tag}_{usage_tag}_{digest}"

def load_fragmentation(dataset,
                       remove_node_features,
                       one_hot_degree,
                       one_hot_node_features: bool,
                       one_hot_edge_features,
                       fragmentation_method,
                       loader_params,
                       encoding=None,
                       subset_frac=1,
                       higher_edge_features=False,
                       dataset_seed=None,
                       **kwargs):
    """
    Loads the data and computes the fragmentation for a given dataset.

    Args:
        dataset (str): The name of the dataset.
        remove_node_features (bool): Whether to remove node features.
        one_hot_degree (bool): Whether to one-hot encode the node degrees.
        one_hot_node_features (bool): Whether to one-hot encode the node features.
        one_hot_edge_features (bool): Whether to one-hot encode the edge features.
        fragmentation_method (tuple): A tuple containing the fragmentation method, usage, and vocabulary size parameters.
        loader_params (dict): A dictionary containing the loader parameters.
        encoding (list, optional): A list of dictionaries for additional encodings (currently only supports random walk encoding). Defaults to None.
        subset_frac (float, optional): The fraction of the dataset to use for training. Defaults to 1 (full training dataset).
        higher_edge_features (bool, optional): Whether to use higher-level edge features (edge features on the higher-level graph). Defaults to False.
        dataset_seed (int, optional): The seed for the dataset. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        DataLoader: The DataLoader object containing the loaded data including the fragmentation.
    """

    if fragmentation_method:
        frag_type, frag_usage, vocab_size_params = fragmentation_method
        vocab_size_params = vocab_size_params.copy()

        def _coerce_bool(value):
            if isinstance(value, str):
                value = value.strip().lower()
                if value in {"1", "true", "t", "yes", "y", "on"}:
                    return True
                if value in {"0", "false", "f", "no", "n", "off", ""}:
                    return False
            return bool(value)

        use_oov_bucket = _coerce_bool(vocab_size_params.pop("use_oov_bucket", False))
        constant_fragment_labels = _coerce_bool(
            vocab_size_params.pop("constant_fragment_labels", False)
        )
        fragment_label_mode = vocab_size_params.pop(
            "fragment_label_mode",
            "constant_onehot" if constant_fragment_labels else "default",
        )
        fragment_label_mode = str(fragment_label_mode).strip().lower()
        fragment_label_aliases = {
            "": "default",
            "default": "default",
            "none": "default",
            "off": "default",
            "false": "default",
            "constant": "constant_onehot",
            "constant_onehot": "constant_onehot",
            "constant-onehot": "constant_onehot",
        }
        if fragment_label_mode not in fragment_label_aliases:
            raise RuntimeError(
                "Unsupported fragment_label_mode: "
                f"{fragment_label_mode!r}. Supported values are 'default' and "
                "'constant_onehot'."
            )
        fragment_label_mode = fragment_label_aliases[fragment_label_mode]
        constant_fragment_label_id = int(
            vocab_size_params.pop("constant_fragment_label_id", 0)
        )

        # Higher-level graph options are consumed here and must not be passed
        # through to the base fragmenter constructors below.
        higher_graph_edge_policy = vocab_size_params.pop(
            "higher_edge_policy",
            vocab_size_params.pop("edge_policy", "overlap"),
        )
        higher_graph_max_distance = int(vocab_size_params.pop(
            "higher_max_distance",
            vocab_size_params.pop("max_distance", 2),
        ))

        # normalize a few aliases for a no-op scheme
        norm = str(frag_type).strip().lower()
        is_no_frag = norm in {"no-fragmentation", "none", "noop"}

        if frag_type == "HiFrAMes":
            vocab_size_params["label_mode"] = frag.normalize_hiframes_label_mode(
                vocab_size_params.get("label_mode", "family_size")
            )
            vocab_size_params["vocab_size"] = frag.infer_hiframes_vocab_size(vocab_size_params)

        if frag_usage == "higher_level_graph_tree":
            # later on, tree construction introduces an additional junction fragment
            vocab_size_params["vocab_size"] -= 1

        # vocab = get_vocab(dataset, frag_type, vocab_size_params["vocab_size"])

        if is_no_frag:
            vocab = None
            # ensure presence and value of vocab_size for downstream modules
            vocab_size_params["vocab_size"] = 0
        elif frag_type == "RingsErtlEFGs":
            max_ring = vocab_size_params.get("max_ring", 15)
            total_vocab = vocab_size_params["vocab_size"]
            efg_max = total_vocab - max_ring
            if efg_max <= 0:
                raise RuntimeError( f"TOTAL_VOCAB ({total_vocab}) must be > MAX_RING ({max_ring}).")
            vocab = get_vocab(dataset, frag_type, efg_max)
        else:
            vocab = get_vocab(dataset, frag_type, vocab_size_params["vocab_size"])

        frag_constructions = {
            "BBB": frag.BreakingBridgeBonds,
            "PSM": frag.PSM,
            "BRICS": frag.BRICS,
            "Magnet": frag.Magnet,
            "Rings": frag.Rings,
            "RingsEdges": frag.RingsEdges,
            "RingsPaths": frag.RingsPaths,
            "RingsEDBs": frag.RingsEDBs,
            "MagnetWithoutVocab": frag.MagnetWithoutVocab,
            "ErtlEFGs": frag.ErtlEFGs,
            "RingsErtlEFGs": frag.RingsErtlEFGs,
            "HiFrAMes": frag.HiFrAMes,
            "no-fragmentation": frag.NoFragmentation,
            "none": frag.NoFragmentation,
            "noop": frag.NoFragmentation,
        }
        frag_usages = {
            "node_feature":
            frag.NodeFeature(vocab_size_params["vocab_size"]),
            "global":
            frag.GraphLevelFeature(vocab_size_params["vocab_size"]),
            "fragment":
            frag.FragmentRepresentation(vocab_size_params["vocab_size"]),
            "higher_level_graph_tree":
            frag.HigherLevelGraph(
                vocab_size=vocab_size_params["vocab_size"],
                edge_policy=higher_graph_edge_policy,
                mode="tree",
                higher_edge_features=higher_edge_features,
                max_distance=higher_graph_max_distance,
            ),
            "higher_level_graph_node":
            frag.HigherLevelGraph(
                vocab_size=vocab_size_params["vocab_size"],
                edge_policy=higher_graph_edge_policy,
                mode="node",
                higher_edge_features=higher_edge_features,
                max_distance=higher_graph_max_distance,
            )
        }

    # create fragmentation
    transformations = []

    if remove_node_features:
        transformations.append(RemoveNodeFeature())

    if one_hot_node_features:
        one_hot_classes = 30
        transformations.append(
            OneHotEncoding(num_classes=one_hot_classes, feature="x"))

    if one_hot_edge_features:
        num_edge_types = 4
        transformations.append(
            OneHotEncoding(num_classes=num_edge_types, feature="edge_attr"))

    if one_hot_degree:
        transformations.append(OneHotDegree(100))

    if fragmentation_method:
        if is_no_frag:
            # create empty substructures; downstream usage stages will make empty tensors
            frag_construction = frag.NoFragmentation()
        elif frag_type == "RingsErtlEFGs":
            max_ring = vocab_size_params.get("max_ring", 15)
            frag_construction = frag.RingsErtlEFGs(
                vocab,
                max_ring=max_ring,
                use_oov_bucket=use_oov_bucket,
            )
        elif frag_type in {"BRICS", "ErtlEFGs"}:
            frag_construction = frag_constructions[frag_type](
                vocab,
                use_oov_bucket=use_oov_bucket,
            )
        else:
            frag_construction = (
                frag_constructions[frag_type](vocab)
                if vocab else frag_constructions[frag_type](**vocab_size_params)
            )

        frag_representation = frag_usages[frag_usage]
        transformations.append(frag_construction)
        transformations.append(frag_representation)

        if fragment_label_mode == "constant_onehot":
            if frag_usage not in {"fragment", "higher_level_graph_tree", "higher_level_graph_node"}:
                raise RuntimeError(
                    "fragment_label_mode='constant_onehot' requires a fragment "
                    "representation usage ('fragment', 'higher_level_graph_tree', "
                    "or 'higher_level_graph_node')."
                )
            transformations.append(
                frag.ConstantOneHotFragmentLabels(
                    constant_id=constant_fragment_label_id,
                )
            )

        if is_no_frag and frag_usage:
            transformations.append(
                frag.EnsureFragmentPlaceholders(
                    vocab_size=vocab_size_params.get("vocab_size", 0)
                )
            )

    if encoding:
        for encod in encoding:
            if encod["name"] == "random-walk":
                random_walk_encoding = add_positional_encoding.AddRandomWalkPE(
                    encod["walk_length"])
                transformations.append(random_walk_encoding)
            else:
                raise RuntimeError(f"Encoding {encod['name']} not supported.")

    # turn fragmentation method into a valid filename ()
    config_name = _build_config_name(
        fragmentation_method=fragmentation_method,
        encoding=encoding,
        remove_node_features=remove_node_features,
        one_hot_edge_features=one_hot_edge_features,
        one_hot_degree=one_hot_degree,
        one_hot_node_features=one_hot_node_features,
        higher_edge_features=higher_edge_features,
    )
    
    # symlink generation to raw data for any raw dataset
    root = f'{DATASET_ROOT}/{dataset}/{dataset}_{config_name}'
    raw_link = os.path.join(root, "raw")
    real_raw = os.path.abspath(
        os.path.join(DATASET_ROOT, dataset, f"{dataset}_mol", "raw")
    )

    zinc_like_datasets = {
        "ZINC",
        "ZINC-full",
        "ZINC-fixed",
        "ZINC-full-fixed",
        "ZINC-logP",
        "ZINC-full-logP",
        "ZINC-count",
    }

    if dataset in zinc_like_datasets:
        pathlib.Path(root).mkdir(parents=True, exist_ok=True)

        lock_path = os.path.join(root, ".process.lock")
        with open(lock_path, "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)

            try:
                os.symlink(real_raw, raw_link)
            except FileExistsError:
                if not os.path.islink(raw_link):
                    raise RuntimeError(f"{raw_link} exists but is not a symlink")
                if os.path.realpath(raw_link) != os.path.realpath(real_raw):
                    raise RuntimeError(
                        f"{raw_link} points to {os.path.realpath(raw_link)} "
                        f"instead of {os.path.realpath(real_raw)}"
                    )

            if dataset == "ZINC" or dataset == "ZINC-full":
                transformations.insert(0, ZINC_Graph_Add_Mol())
                transform = Compose(transformations)
                subset = True if dataset == "ZINC" else False
                train_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="train")
                val_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="val")
                test_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="test")
                data = train_data
                num_classes = 1

            elif dataset == "ZINC-fixed" or dataset == "ZINC-full-fixed":
                transformations.insert(0, ZINC_Graph_Add_Mol())
                transformations.append(FixPlogP())
                transform = Compose(transformations)
                subset = True if dataset == "ZINC-fixed" else False
                train_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="train")
                val_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="val")
                test_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="test")
                data = train_data
                num_classes = 1

            elif dataset == "ZINC-logP" or dataset == "ZINC-full-logP":
                transformations.insert(0, ZINC_Graph_Add_Mol())
                transformations.append(LogP())
                transform = Compose(transformations)
                subset = True if dataset == "ZINC-logP" else False
                train_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="train")
                val_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="val")
                test_data = ZINC(
                    root=root,
                    pre_transform=transform,
                    subset=subset,
                    split="test")
                data = train_data
                num_classes = 1

            elif dataset == "ZINC-count":
                transformations.insert(0, ZINC_Graph_Add_Mol())
                substructure_idx = kwargs["substructure_idx"] if "substructure_idx" in kwargs else None
                transformations.insert(1, CountSubstructures(
                    substructure_idx=substructure_idx))
                transform = Compose(transformations)
                subset = True
                count_root = f"{root}_{substructure_idx}"

                pathlib.Path(count_root).mkdir(parents=True, exist_ok=True)
                count_raw_link = os.path.join(count_root, "raw")
                count_lock_path = os.path.join(count_root, ".process.lock")

                with open(count_lock_path, "w") as count_lockf:
                    fcntl.flock(count_lockf, fcntl.LOCK_EX)

                    try:
                        os.symlink(real_raw, count_raw_link)
                    except FileExistsError:
                        if not os.path.islink(count_raw_link):
                            raise RuntimeError(f"{count_raw_link} exists but is not a symlink")
                        if os.path.realpath(count_raw_link) != os.path.realpath(real_raw):
                            raise RuntimeError(
                                f"{count_raw_link} points to {os.path.realpath(count_raw_link)} "
                                f"instead of {os.path.realpath(real_raw)}"
                            )

                    train_data = ZINC(
                        root=count_root,
                        pre_transform=transform,
                        subset=subset,
                        split="train")
                    val_data = ZINC(
                        root=count_root,
                        pre_transform=transform,
                        subset=subset,
                        split="val")
                    test_data = ZINC(
                        root=count_root,
                        pre_transform=transform,
                        subset=subset,
                        split="test")

                data = train_data
                num_classes = 15 if substructure_idx is None else 1

    elif dataset == "ogbg-molhiv":
        transformations.insert(0, OGB_Graph_Add_Mol_By_Smiles())
        transform = Compose(transformations)
        data = PygGraphPropPredDataset(
            name=dataset,
            root=f"{DATASET_ROOT}/{dataset}/{dataset}_{config_name}",
            pre_transform=transform)
        split_idx = data.get_idx_split()
        train_data = data[split_idx["train"]]
        val_data = data[split_idx["valid"]]
        test_data = data[split_idx["test"]]
        num_classes = data.num_classes

    elif dataset == "peptides-struct":
        transform = Compose(transformations)
        data = lrgb.PeptidesStructuralDataset(
            root=f"{DATASET_ROOT}/{dataset}/{dataset}_{config_name}",
            pre_transform=transform,
            smiles2graph=lrgb.smiles2graph_add_mol)
        split_idx = data.get_idx_split()
        train_data = data[split_idx["train"]]
        val_data = data[split_idx["val"]]
        test_data = data[split_idx["test"]]
        num_classes = data.num_classes

    elif dataset == "peptides-func":
        transform = Compose(transformations)
        data = lrgb.PeptidesFunctionalDataset(
            root=f"{DATASET_ROOT}/{dataset}/{dataset}_{config_name}",
            pre_transform=transform,
            smiles2graph=lrgb.smiles2graph_add_mol)
        split_idx = data.get_idx_split()
        train_data = data[split_idx["train"]]
        val_data = data[split_idx["val"]]
        test_data = data[split_idx["test"]]
        num_classes = 2  # multilabel binary classification task

    else:
        raise RuntimeError(f"Dataset {dataset} not supported")

    if fragmentation_method and frag_usage in [
            "fragment", "higher_level_graph_tree", "higher_level_graph_node"
    ]:
        follow_batch = ["x", "fragments"]
    else:
        follow_batch = None

    if subset_frac < 1:
        subset_size = int(subset_frac * len(train_data))
        train_data, _ = random_split(
            train_data,
            [subset_size, len(train_data) - subset_size])

    if not dataset_seed:
        train_loader = DataLoader(train_data,
                                  batch_size=loader_params["batch_size"],
                                  num_workers=loader_params["num_workers"],
                                  follow_batch=follow_batch,
                                  shuffle=True)
    else:
        # set seperate seed for dataloaderr
        g = torch.Generator()
        g.manual_seed(dataset_seed)
        train_loader = DataLoader(train_data,
                                  batch_size=loader_params["batch_size"],
                                  num_workers=loader_params["num_workers"],
                                  follow_batch=follow_batch,
                                  shuffle=True,
                                  generator=g)

    val_batch_size = loader_params[
        "val_batch_size"] if "val_batch_size" in loader_params else loader_params[
            "batch_size"]
    val_loader = DataLoader(val_data,
                            batch_size=val_batch_size,
                            num_workers=loader_params["num_workers"],
                            follow_batch=follow_batch)
    test_loader = DataLoader(test_data,
                             batch_size=val_batch_size,
                             num_workers=loader_params["num_workers"],
                             follow_batch=follow_batch)

    num_features = data.num_features
    return train_loader, val_loader, test_loader, num_features, num_classes


def get_vocab(dataset: str,
              frag_type: str,
              max_vocab=100,
              root: str = VOCAB_ROOT):
    """
    Get the vocabulary for a given dataset and fragmentation type.

    Args:
        dataset (str): The name of the dataset.
        frag_type (str): The type of fragmentation.
        max_vocab (int, optional): The maximum size of the vocabulary. Defaults to 100.
        root (str, optional): The root directory for storing the vocabulary. Defaults to VOCAB_ROOT.

    Returns:
        vocab: The vocabulary for the given dataset and fragmentation type, if the fragmentation scheme requires no vocabulary
        (e.g. RingsPaths fragmentation) None is returned.
    """

    vocab_constructions = {
        "BBB": frag.BreakingBridgeBondsVocab(vocab_size=max_vocab),
        "PSM": frag.PrincipalSubgraphVocab(
            vocab_size=max_vocab,
            vocab_path=f'/tmp/{dataset}_PSM_vocab_{max_vocab}.txt'),
        "BRICS": frag.BRICSVocab(vocab_size=max_vocab),
        "Magnet": frag.MagnetVocab(vocab_size=max_vocab),
        "Rings": None,
        "RingsEdges": None,
        "RingsPaths": None,
        "RingsEDBs": None,
        "MagnetWithoutVocab": None,
        "ErtlEFGs": frag.ErtlEFGVocab(vocab_size=max_vocab),
        "RingsErtlEFGs": frag.ErtlEFGVocab(vocab_size=max_vocab),
        "HiFrAMes": None,
        "no-fragmentation": None,
        "none": None,
        "noop": None,
    }

    if frag_type not in vocab_constructions:
        raise RuntimeError("Fragmentation type is not supported")

    if vocab_constructions[frag_type]:
        # check if vocab already exists
        vocab_file_name = f"{root}/{dataset}/{dataset}_{frag_type}_vocab_{max_vocab}"
        if os.path.isfile(vocab_file_name):
            vocab = frag.get_vocab_from_file(vocab_file_name)
        else:
            # create vocab
            if dataset in ["ZINC", "ZINC-full",
               "ZINC-fixed", "ZINC-full-fixed", "ZINC-logP", "ZINC-full-logP"]:
                subset = "full" not in dataset  # subsets are for datasets without "full"
                vocab_data = ZINC(
                    root=f'{DATASET_ROOT}/{dataset}/{dataset}_mol',
                    pre_transform=ZINC_Graph_Add_Mol(),
                    subset=subset,
                    split="train"
                )
            elif dataset == "ogbg-molhiv":
                vocab_data = PygGraphPropPredDataset(
                    name=dataset,
                    root=f'{DATASET_ROOT}/{dataset}/{dataset}_mol',
                    pre_transform=OGB_Graph_Add_Mol_By_Smiles())
                split_idx = vocab_data.get_idx_split()
                vocab_data = vocab_data[split_idx["train"]]
            elif dataset == "peptides-struct":
                vocab_data = lrgb.PeptidesStructuralDataset(
                    root=f'{DATASET_ROOT}/{dataset}/{dataset}_mol',
                    smiles2graph=lrgb.smiles2graph_add_mol)
                split_idx = vocab_data.get_idx_split()
                vocab_data = vocab_data[split_idx["train"]]
            elif dataset == "peptides-func":
                vocab_data = lrgb.PeptidesFunctionalDataset(
                    root=f'{DATASET_ROOT}/{dataset}/{dataset}_mol',
                    smiles2graph=lrgb.smiles2graph_add_mol)
                split_idx = vocab_data.get_idx_split()
                vocab_data = vocab_data[split_idx["train"]]
            else:
                raise RuntimeError(f"Dataset {dataset} not supported")

            vocab_generator = vocab_constructions[frag_type]
            for data in vocab_data:
                vocab_generator(data)
            vocab = vocab_generator.get_vocab()
            frag.vocab_to_file(vocab_generator, vocab_file_name)  # save vocab

        return vocab
    return None


def _get_max_degree(data: Data):
    max_degree = 0
    for graph in data:
        degrees = degree(data.edge_index[0], dtype=torch.long)
        max_degree = max(max_degree, degrees.max())
    return max_degree


class RemoveNodeFeature(BaseTransform):

    def __init__(self):
        pass

    def __call__(self, graph):
        graph.x = torch.zeros((graph.num_nodes, 1))
        return graph


class OneHotEncoding(BaseTransform):

    def __init__(self, num_classes, feature: str):
        self.num_classes = num_classes
        self.feature = feature

    def __call__(self, graph):
        one_hot_encoded = one_hot(
            getattr(graph, self.feature),
            num_classes=self.num_classes).float().squeeze(1)
        setattr(graph, self.feature, one_hot_encoded)
        return graph
