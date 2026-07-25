import json
import math
import os
import socket
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (Callback, EarlyStopping,
                                         LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import CSVLogger
import torch
from torch_geometric.seed import seed_everything

import data.data
import data.data_ood
import data.fragmentations.fragmentations as frag
from config import CHECKPOINT_DIR
from models.fragGNN import FragGNN, FragGNNSmall
from models.gcn import GCN, GCNSubstructure, VerySimpleGCN
from models.lightning_models import *

def _batch_num_graphs(batch):
    if hasattr(batch, "num_graphs"):
        try:
            return int(batch.num_graphs)
        except TypeError:
            return int(batch.num_graphs.item())
    if hasattr(batch, "batch") and batch.batch.numel() > 0:
        return int(torch.max(batch.batch).item()) + 1
    if hasattr(batch, "x_batch") and batch.x_batch.numel() > 0:
        return int(torch.max(batch.x_batch).item()) + 1
    return 1


def _safe_tensor_count(obj, attr_name, dim=None):
    if not hasattr(obj, attr_name):
        return 0
    value = getattr(obj, attr_name)
    if value is None:
        return 0
    if hasattr(value, "size"):
        if dim is None:
            return int(value.numel())
        if value.dim() <= dim:
            return 0
        return int(value.size(dim))
    return 0


def _count_virtual_fragments(batch):
    if not hasattr(batch, "fragments"):
        return 0
    num_fragments = int(batch.fragments.size(0))
    if num_fragments == 0:
        return 0
    if not hasattr(batch, "fragments_edge_index") or batch.fragments_edge_index.numel() == 0:
        return num_fragments
    attached = torch.unique(batch.fragments_edge_index[1])
    return max(0, num_fragments - int(attached.numel()))


def _nested_get(dct, keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class LightweightLoggingCallback(Callback):
    def __init__(
        self,
        checkpoint_directory,
        run_metadata=None,
        grad_norm_interval=50,
        track_batch_graph_stats=True,
        track_cuda_memory=True,
    ):
        super().__init__()
        self.checkpoint_directory = checkpoint_directory
        self.run_metadata = run_metadata or {}
        self.grad_norm_interval = int(grad_norm_interval)
        self.track_batch_graph_stats = bool(track_batch_graph_stats)
        self.track_cuda_memory = bool(track_cuda_memory)
        self._last_grad_log_step = -1
        self._epoch_state = {}
        self._fit_start_time = None
        self._test_start_time = None

    def _reset_phase_state(self, phase):
        self._epoch_state[phase] = {
            "start_time": time.perf_counter(),
            "graphs": 0,
            "nodes": 0,
            "edges": 0,
            "fragments": 0,
            "higher_edges": 0,
            "virtual_fragments": 0,
            "batches": 0,
        }

    def _update_phase_state(self, phase, batch):
        if not self.track_batch_graph_stats:
            return
        state = self._epoch_state.setdefault(phase, {})
        state["graphs"] += _batch_num_graphs(batch)
        state["nodes"] += _safe_tensor_count(batch, "x", dim=0)
        state["edges"] += _safe_tensor_count(batch, "edge_index", dim=1)
        state["fragments"] += _safe_tensor_count(batch, "fragments", dim=0)
        state["higher_edges"] += _safe_tensor_count(batch, "higher_edge_index", dim=1)
        state["virtual_fragments"] += _count_virtual_fragments(batch)
        state["batches"] += 1

    def _maybe_reset_cuda_peak(self, trainer):
        if not self.track_cuda_memory:
            return
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)
            except Exception:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

    def _peak_cuda_metrics(self, trainer):
        metrics = {}
        if not self.track_cuda_memory:
            return metrics
        if torch.cuda.is_available():
            try:
                device = trainer.strategy.root_device
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
            except Exception:
                allocated = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
            metrics["cuda_max_memory_allocated_mb"] = allocated / (1024 ** 2)
            metrics["cuda_max_memory_reserved_mb"] = reserved / (1024 ** 2)
        return metrics

    def _log_metrics(self, trainer, metrics):
        if trainer.logger is not None and metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def _finalize_phase(self, trainer, phase):
        state = self._epoch_state.get(phase)
        if not state:
            return

        elapsed = max(time.perf_counter() - state["start_time"], 1e-12)
        metrics = {
            f"timing/{phase}_epoch_sec": elapsed,
        }

        if self.track_batch_graph_stats:
            graphs = max(state["graphs"], 1)
            metrics.update({
                f"throughput/{phase}_graphs_per_sec": state["graphs"] / elapsed,
                f"throughput/{phase}_batches_per_sec": state["batches"] / elapsed,
                f"graph_stats/{phase}_mean_nodes_per_graph": state["nodes"] / graphs,
                f"graph_stats/{phase}_mean_edges_per_graph": state["edges"] / graphs,
                f"graph_stats/{phase}_mean_fragments_per_graph": state["fragments"] / graphs,
                f"graph_stats/{phase}_mean_higher_edges_per_graph": state["higher_edges"] / graphs,
                f"graph_stats/{phase}_mean_virtual_fragments_per_graph": state["virtual_fragments"] / graphs,
                f"graph_stats/{phase}_total_graphs": state["graphs"],
            })

        peak_metrics = self._peak_cuda_metrics(trainer)
        metrics.update({f"memory/{phase}_{k}": v for k, v in peak_metrics.items()})
        self._log_metrics(trainer, metrics)

    def setup(self, trainer, pl_module, stage=None):
        os.makedirs(self.checkpoint_directory, exist_ok=True)

    def on_fit_start(self, trainer, pl_module):
        self._fit_start_time = time.perf_counter()
        metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "git_commit": _git_commit_hash(),
            **self.run_metadata,
        }
        metadata_path = os.path.join(self.checkpoint_directory, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, default=str)
        if trainer.logger is not None:
            try:
                trainer.logger.log_hyperparams({
                    "experiment_name": metadata.get("experiment_name"),
                    "dataset": metadata.get("dataset"),
                    "model_type": metadata.get("model_type"),
                    "batch_size": metadata.get("batch_size"),
                    "seed": metadata.get("seed"),
                    "dataset_seed": metadata.get("dataset_seed"),
                    "fragmentation_name": metadata.get("fragmentation_name"),
                    "hlg_mode": metadata.get("hlg_mode"),
                    "higher_edge_policy": metadata.get("higher_edge_policy"),
                    "higher_max_distance": metadata.get("higher_max_distance"),
                    "lr": metadata.get("lr"),
                    "weight_decay": metadata.get("weight_decay"),
                    "gradient_clip_val": metadata.get("gradient_clip_val"),
                })
            except Exception:
                pass

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset_phase_state("train")
        self._maybe_reset_cuda_peak(trainer)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset_phase_state("val")
        self._maybe_reset_cuda_peak(trainer)

    def on_test_epoch_start(self, trainer, pl_module):
        self._reset_phase_state("test")
        self._maybe_reset_cuda_peak(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._update_phase_state("train", batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_phase_state("val", batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_phase_state("test", batch)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, *args, **kwargs):
        if self.grad_norm_interval <= 0:
            return
        if trainer.global_step == self._last_grad_log_step:
            return
        if trainer.global_step % self.grad_norm_interval != 0:
            return

        sq_norm = 0.0
        found_grad = False
        for param in pl_module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            sq_norm += float(torch.sum(grad * grad).item())
            found_grad = True

        if found_grad:
            grad_norm = math.sqrt(sq_norm)
            self._log_metrics(trainer, {"optimization/grad_global_norm": grad_norm})
            self._last_grad_log_step = trainer.global_step

    def on_train_epoch_end(self, trainer, pl_module):
        self._finalize_phase(trainer, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._finalize_phase(trainer, "val")

    def on_test_start(self, trainer, pl_module):
        self._test_start_time = time.perf_counter()

    def on_test_epoch_end(self, trainer, pl_module):
        self._finalize_phase(trainer, "test")

    def on_test_end(self, trainer, pl_module):
        if self._test_start_time is not None:
            self._log_metrics(trainer, {"timing/test_total_sec": time.perf_counter() - self._test_start_time})

    def on_fit_end(self, trainer, pl_module):
        metrics = {}
        if self._fit_start_time is not None:
            metrics["timing/fit_total_sec"] = time.perf_counter() - self._fit_start_time
        metrics["training/epochs_completed"] = trainer.current_epoch + 1
        checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_cb is not None:
            if getattr(checkpoint_cb, "best_model_path", None):
                metrics["checkpoint/best_model_path"] = checkpoint_cb.best_model_path
            if getattr(checkpoint_cb, "best_model_score", None) is not None:
                best_score = checkpoint_cb.best_model_score
                try:
                    best_score = float(best_score.item())
                except Exception:
                    best_score = float(best_score)
                metrics["checkpoint/best_model_score"] = best_score
        self._log_metrics(trainer, metrics)
        summary_path = os.path.join(self.checkpoint_directory, "run_summary.json")
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True, default=str)


class ExperimentWrapper:

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()
    def init_dataset(self,
                     _config,
                     dataset: str,
                     seed: Optional[int] = None,
                     remove_node_features: bool = False,
                     one_hot_degree: bool = False,
                     one_hot_node_features: bool = False,
                     one_hot_edge_features: bool = False,
                     fragmentation_method: Optional[Tuple[str, str,
                                                          Dict]] = None,
                     loader_params: Optional[Dict] = None,
                     encoding: List = [],
                     dataset_params={}):
        """Initialize train, validation and test loader.

        Parameters
        ----------
        dataset
            Name of the dataset
        seed
            Seed for everything
        remove_node_features, optional
            Boolean indicating whether node_labels should be removed, by default False
        one_hot_degree, optional
            Boolean indicating whether to concatinate the node features with a one hot encoded node degree, by default False.
        fragmentation_method, optional
            Tuple ``(name_of_fragmentation, type_of_fragmentation, vocab_size)``.
        loader_params, optional
            Dictionary containing train_fraction, val_fraction and batch_size, not needed for Planetoid datasets, by default None.
        encoding, optional
            List of encodings that should be used.
        dataset_params, optional
            If subset_frac in dataset_params: Only subset_frac of the dataset will be used for training.
            If filter in dataset_params: Only molecules containing no ring of size filter will be used for training.
            If higher_edge_features in dataset_params: Information about the edges in the higher level graph will be computed.
            If dataset_seed in dataset_params: Seperate seed for the dataset split.
        """
        print(f"Dataset received config: {_config}")
        if seed is not None:
            # torch.manual_seed(seed)
            seed_everything(seed)

        if fragmentation_method and len(fragmentation_method) > 2:
            frag_name = fragmentation_method[0]
            frag_params = fragmentation_method[2]
            if frag_name == "HiFrAMes" and isinstance(frag_params, dict):
                self.num_substructures = frag.infer_hiframes_vocab_size(frag_params)
            else:
                self.num_substructures = frag_params["vocab_size"]
        else:
            self.num_substructures = None

        if "filter" in dataset_params:
            # only used in the ood experiment
            self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes = data.data_ood.load_fragmentation(
                dataset,
                remove_node_features=remove_node_features,
                one_hot_degree=one_hot_degree,
                one_hot_node_features=one_hot_node_features,
                one_hot_edge_features=one_hot_edge_features,
                fragmentation_method=fragmentation_method,
                loader_params=loader_params,
                **dataset_params)
        else:
            self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes = data.data.load_fragmentation(
                dataset,
                remove_node_features=remove_node_features,
                one_hot_degree=one_hot_degree,
                one_hot_node_features=one_hot_node_features,
                one_hot_edge_features=one_hot_edge_features,
                fragmentation_method=fragmentation_method,
                loader_params=loader_params,
                encoding=encoding,
                **dataset_params)
    def init_model(self,
                   model_type: str,
                   model_params: dict,
                   classification: bool = True):
        self.classification = classification
        model_params = model_params.copy()  # allows us to add fields to it
        if not "out_channels" in model_params:
            if classification:
                model_params[
                    "out_channels"] = self.num_classes if self.num_classes > 2 else 1
            else:
                model_params["out_channels"] = self.num_classes
        model_params["in_channels"] = self.num_features
        if model_type == "GCN":
            self.model = GCN(**model_params)
        elif model_type == "VerySimpleGCN":
            self.model = VerySimpleGCN(**model_params)
        elif model_type == "GCNSubstructure":
            model_params["in_channels_substructure"] = self.num_substructures
            self.model = GCNSubstructure(**model_params)
        elif model_type == "FragGNNSmall":
            model_params["in_channels_substructure"] = (self.num_substructures or 0)
            model_params[
                "in_channels_edge"] = 4  # TODO: could be different for other datasets
            self.model = FragGNNSmall(**model_params)
        elif model_type == "FragGNN":
            model_params["in_channels_substructure"] = (self.num_substructures or 0)
            model_params[
                "in_channels_edge"] = 4  # TODO: could be different for other datasets
            self.model = FragGNN(**model_params)
        else:
            raise RuntimeError(f"Model {model_type} not supported")
        print("Setup model:")
        print(self.model)
    def init_optimizer(self,
                   optimization_params,
                   scheduler_parameters=None,
                   loss: Optional[str] = None,
                   additional_metric: Optional[str] = None,
                   ema_decay=None,
                   train_loss_on_step: bool = True,
                   train_loss_on_epoch: bool = True):
        loss_func = None
        if self.classification and self.num_classes > 2:
            loss_func = ce_loss
            acc = classification_accuracy
        elif self.classification:
            loss_func = bce_loss
            acc = binary_classification_accuracy
        else:
            if loss and loss == "mae":
                loss_func = mae_loss
            else:
                loss_func = mse_loss
            acc = regression_acc

        additional_metric_func = None
        if additional_metric:
            if additional_metric == "mae":
                additional_metric_func = mae_loss
            elif additional_metric == "mse":
                additional_metric_func = mse_loss
            elif additional_metric == "auroc":
                additional_metric_func = auroc
            elif additional_metric == "ap":
                additional_metric_func = average_multilabel_precision
            elif additional_metric == "counting_experiment":
                additional_metric_func = [
                    regression_acc, regression_precision, regression_recall, num_true_positives]

        self.lightning_model = LightningModel(
            model=self.model,
            loss=loss_func,
            acc=acc,
            optimizer_parameters=optimization_params,
            scheduler_parameters=scheduler_parameters,
            additional_metric=additional_metric_func,
            ema_decay=ema_decay,
            train_loss_on_step=train_loss_on_step,
            train_loss_on_epoch=train_loss_on_epoch)

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.init_optimizer()
    def train(self, trainer_params, project_name, _config, notes="", ckpt_path=None, use_wandb=True, use_profiler=False):

        db_collection = str(_config.get("db_collection") or "local")
        run_id = _config.get(
            "run_no",
            _config.get("overwrite", _config.get("_id", "local")),
        )
        checkpoint_directory = os.path.join(
            CHECKPOINT_DIR,
            db_collection,
            f"run-{run_id}",
        )
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)

        profiler = "simple" if use_profiler else None

        if "gradient_clip_val" in trainer_params:
            additional_params = {
                "gradient_clip_val": trainer_params["gradient_clip_val"]
            }
        else:
            additional_params = {}

        log_every_n_steps = int(trainer_params.get("log_every_n_steps", 15))
        grad_norm_interval = int(trainer_params.get("grad_norm_interval", 50))
        track_batch_graph_stats = bool(
            trainer_params.get("track_batch_graph_stats", True)
        )
        track_cuda_memory = bool(
            trainer_params.get("track_cuda_memory", True)
        )

        monitor = trainer_params["monitor"] if "monitor" in trainer_params else "val_loss"
        mode = "min" if monitor == "val_loss" else "max"
        patience = trainer_params["patience_early_stopping"] if "patience_early_stopping" in trainer_params else 50

        csv_logger = CSVLogger(save_dir=checkpoint_directory, name="csv_logs")
        run_metadata = {
            "experiment_name": _config.get("experiment_name") or _config.get("overwrite") or f"{_config.get('db_collection')}_{_config.get('run_no')}",
            "project_name": project_name,
            "notes": notes,
            "db_collection": _config.get("db_collection"),
            "run_no": _config.get("run_no"),
            "dataset": _config.get("data", {}).get("dataset"),
            "seed": _config.get("data", {}).get("seed"),
            "dataset_seed": _nested_get(_config, ["data", "dataset_params", "dataset_seed"]),
            "batch_size": _nested_get(_config, ["data", "loader_params", "batch_size"]),
            "model_type": _config.get("model", {}).get("model_type"),
            "lr": _nested_get(_config, ["optimization", "optimization_params", "lr"]),
            "weight_decay": _nested_get(_config, ["optimization", "optimization_params", "weight_decay"]),
            "gradient_clip_val": trainer_params.get("gradient_clip_val"),
            "fragmentation_name": None,
            "hlg_mode": None,
            "higher_edge_policy": None,
            "higher_max_distance": None,
        }
        fragmentation_method = _nested_get(_config, ["data", "fragmentation_method"])
        if isinstance(fragmentation_method, (list, tuple)):
            if len(fragmentation_method) > 0:
                run_metadata["fragmentation_name"] = fragmentation_method[0]
            if len(fragmentation_method) > 1:
                run_metadata["hlg_mode"] = fragmentation_method[1]
            if len(fragmentation_method) > 2 and isinstance(fragmentation_method[2], dict):
                run_metadata["higher_edge_policy"] = fragmentation_method[2].get("higher_edge_policy")
                run_metadata["higher_max_distance"] = fragmentation_method[2].get("higher_max_distance")

        callbacks = [
            EarlyStopping(monitor=monitor,
                        mode=mode,
                        patience=patience,
                        verbose=True),
            ModelCheckpoint(monitor=monitor, mode=mode),
            LearningRateMonitor(logging_interval="epoch"),
            LightweightLoggingCallback(
                checkpoint_directory=checkpoint_directory,
                run_metadata=run_metadata,
                grad_norm_interval=grad_norm_interval,
                track_batch_graph_stats=track_batch_graph_stats,
                track_cuda_memory=track_cuda_memory,
            ),
        ]

        trainer = Trainer(
            max_epochs=trainer_params["max_epochs"],
            logger=csv_logger,
            log_every_n_steps=log_every_n_steps,
            default_root_dir=checkpoint_directory,
            detect_anomaly=False,
            callbacks=callbacks,
            enable_progress_bar=False,
            profiler=profiler,
            **additional_params)

        trainer.fit(self.lightning_model,
                    train_dataloaders=self.train_loader,
                    val_dataloaders=self.val_loader,
                    ckpt_path=ckpt_path)
        if trainer_params["testing"] == True:
            result = trainer.test(self.lightning_model, self.test_loader)
            print(f"Test result: {result}")
            return result
        else:
            pass
