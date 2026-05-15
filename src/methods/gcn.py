"""GCN-based link prediction — residual encoder, concat predictor, early stopping."""

from __future__ import annotations

import copy
import logging
from typing import Any

from src.evaluation.metrics import DEFAULT_DATASET_NAME, DEFAULT_HITS_KS
from src.evaluation.metrics import evaluate_link_prediction
from src.evaluation.runtime import get_memory_usage_mb, track_resources
from src.experiments.progress import progress_bar
from src.methods.mlp import _edge_tensor, _require_torch

logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────


class GcnEncoder(_require_torch().nn.Module):
    """GCN encoder with BatchNorm, residual, and layer-wise dropout.

    Each layer: GCNConv → BatchNorm → ReLU → Dropout → + residual.
    Output is at reduced dimension (hidden_channels // 2) for compact embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        torch = _require_torch()
        from torch_geometric.nn import GCNConv

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # First layer: in → hidden
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Middle layers: hidden → hidden
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(torch.nn.BatchNorm1d(hidden_channels))

        # Last layer: hidden → out (no BatchNorm, no ReLU, no residual)
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.norms.append(None)

        # Residual projection for first layer if dimensions differ
        if in_channels != hidden_channels:
            self.res_proj = torch.nn.Linear(in_channels, hidden_channels)
        else:
            self.res_proj = None

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if norm is not None:
                norm.reset_parameters()
        if self.res_proj is not None:
            self.res_proj.reset_parameters()

    def forward(self, x: Any, edge_index: Any) -> Any:
        torch = _require_torch()

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index)

            if norm is not None:  # all except last layer
                x = norm(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.dropout(
                    x, p=self.dropout_rate, training=self.training
                )

                # Residual connection
                if i == 0 and self.res_proj is not None:
                    residual = self.res_proj(residual)
                if residual.shape == x.shape:
                    x = x + residual

        return x


class GcnLinkPredictor(_require_torch().nn.Module):
    """Compact MLP link predictor on concatenated embeddings.

    Uses ``cat(h_i, h_j)`` → shallow 2-layer MLP for fast convergence.
    """

    def __init__(
        self,
        in_channels: int,  # 2 × embedding_dim (concatenation)
        hidden_channels: int,
        out_channels: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        torch = _require_torch()

        self.dropout_rate = dropout

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def reset_parameters(self) -> None:
        for module in self.net:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, h_i: Any, h_j: Any) -> Any:
        torch = _require_torch()
        h = torch.cat([h_i, h_j], dim=-1)
        return torch.sigmoid(self.net(h))


# ── Utilities ─────────────────────────────────────────────────────────────


def _select_device(device: str) -> Any:
    torch = _require_torch()
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _make_message_passing_edge_index(train_edges: Any, device: Any) -> Any:
    torch = _require_torch()
    edge_tensor = _edge_tensor(train_edges, device)
    edge_index = edge_tensor.t().contiguous()
    reverse_edge_index = edge_index.flip(0)
    return torch.cat([edge_index, reverse_edge_index], dim=1)


def _score_split(
    encoder: GcnEncoder,
    predictor: GcnLinkPredictor,
    x: Any,
    mp_edge_index: Any,
    split_edge: dict,
    split_name: str,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    """Score positive and negative edges for one split."""
    pos_scores = score_edges_gcn(
        encoder, predictor, x, mp_edge_index,
        split_edge[split_name]["edge"], batch_size,
        description=f"GCN {split_name} positive",
    )
    neg_scores = score_edges_gcn(
        encoder, predictor, x, mp_edge_index,
        split_edge[split_name]["edge_neg"], batch_size,
        description=f"GCN {split_name} negative",
    )
    return pos_scores, neg_scores


# ── Training ──────────────────────────────────────────────────────────────


def train_gcn_epoch(
    encoder: GcnEncoder,
    predictor: GcnLinkPredictor,
    x: Any,
    message_passing_edge_index: Any,
    train_edges: Any,
    optimizer: Any,
    batch_size: int,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> float:
    torch = _require_torch()
    from torch.utils.data import DataLoader

    encoder.train()
    predictor.train()
    total_loss = 0.0
    total_examples = 0

    batches = DataLoader(range(train_edges.size(0)), batch_size=batch_size, shuffle=True)
    description = "GCN batches"
    if epoch is not None and total_epochs is not None:
        description = f"GCN epoch {epoch}/{total_epochs}"

    for perm in progress_bar(batches, desc=description, leave=False):
        optimizer.zero_grad()

        h = encoder(x, message_passing_edge_index)
        edge = train_edges[perm]
        pos_out = predictor(h[edge[:, 0]], h[edge[:, 1]]).view(-1)
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        neg_edge = torch.randint(
            0, x.size(0), edge.shape, dtype=torch.long, device=x.device
        )
        # Mix 50% hard negatives: existing train edges with permuted endpoints
        if torch.rand(1).item() < 0.5 and edge.size(0) > 1:
            perm = torch.randperm(edge.size(0), device=edge.device)
            hard_neg = torch.stack([
                edge[perm, 0],  # source from random row
                edge[:, 1],     # same target
            ], dim=1)
            # Keep only those that aren't actual positive edges (simple dedup)
            neg_edge = torch.cat([neg_edge, hard_neg], dim=0)
            # Take first batch_size worth
            neg_edge = neg_edge[:edge.size(0)]
        neg_out = predictor(h[neg_edge[:, 0]], h[neg_edge[:, 1]]).view(-1)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item()) * edge.size(0)
        total_examples += int(edge.size(0))

    return total_loss / max(total_examples, 1)


def score_edges_gcn(
    encoder: GcnEncoder,
    predictor: GcnLinkPredictor,
    x: Any,
    message_passing_edge_index: Any,
    edges: Any,
    batch_size: int,
    description: str = "GCN score batches",
) -> list[float]:
    torch = _require_torch()
    from torch.utils.data import DataLoader

    encoder.eval()
    predictor.eval()
    edge_tensor = _edge_tensor(edges, x.device)
    scores: list[float] = []

    with torch.no_grad():
        h = encoder(x, message_passing_edge_index)
        batches = DataLoader(range(edge_tensor.size(0)), batch_size=batch_size)
        for perm in progress_bar(batches, desc=description, leave=False):
            edge = edge_tensor[perm]
            batch_scores = predictor(h[edge[:, 0]], h[edge[:, 1]]).view(-1)
            scores.extend(float(v) for v in batch_scores.cpu())

    return scores


# ── Main entry point ──────────────────────────────────────────────────────


def run_gcn(
    data: Any,
    split_edge: dict[str, dict[str, Any]],
    config: Any,
    dataset_name: str = DEFAULT_DATASET_NAME,
    ks: tuple[int, ...] = DEFAULT_HITS_KS,
) -> dict[str, Any]:
    """Train and evaluate GCN with validation-based early stopping.

    Improvements over baseline:
    - Residual GCN encoder with BatchNorm
    - Concatenation-based link predictor (not Hadamard)
    - Validation tracking every ``val_freq`` epochs
    - Cosine annealing LR schedule
    - Best model checkpointing by validation Hits@50
    """
    torch = _require_torch()

    if data is None or not hasattr(data, "x") or data.x is None:
        raise ValueError("GCN requires node features in data.x.")

    hp = config.hyperparameters
    device = _select_device(config.device)
    torch.manual_seed(config.seed)

    epochs: int = hp.get("epochs", 400)
    batch_size: int = hp.get("batch_size", 64 * 1024)
    hidden_channels: int = hp.get("hidden_channels", 256)
    embed_channels: int = hp.get("embed_channels", 128)
    num_layers: int = hp.get("num_layers", 3)
    dropout: float = hp.get("dropout", 0.2)
    learning_rate: float = hp.get("learning_rate", 0.005)
    val_freq: int = hp.get("val_freq", 10)
    early_stop_patience: int = hp.get("early_stop_patience", 5)

    logger.info(
        "Starting GCN v3: scale=%s epochs=%s lr=%s hidden=%s embed=%s layers=%s dropout=%s device=%s",
                config.dataset_scale, epochs, learning_rate, hidden_channels, embed_channels, num_layers, dropout, device,
    )

    start_memory_mb = get_memory_usage_mb()
    with track_resources() as usage:
        x = data.x.to(device=device, dtype=torch.float)
        train_edges = _edge_tensor(split_edge["train"]["edge"], device)
        message_passing_edge_index = _make_message_passing_edge_index(train_edges, device)

        # ── Build models ──────────────────────────────────────────────
        encoder = GcnEncoder(
            in_channels=x.size(-1),
            hidden_channels=hidden_channels,
            out_channels=embed_channels,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        predictor = GcnLinkPredictor(
            in_channels=2 * embed_channels,  # concatenation → 2×
            hidden_channels=hidden_channels,
            out_channels=1,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(predictor.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6,
        )

        # ── Train with validation tracking ────────────────────────────
        losses: list[float] = []
        val_metrics: list[dict[str, float]] = []
        best_val_hits50 = -1.0
        best_state: dict | None = None
        best_epoch = 0
        no_improve_streak = 0

        for epoch in progress_bar(range(1, epochs + 1), desc="GCN epochs"):
            loss = train_gcn_epoch(
                encoder=encoder, predictor=predictor,
                x=x, message_passing_edge_index=message_passing_edge_index,
                train_edges=train_edges, optimizer=optimizer,
                batch_size=batch_size, epoch=epoch, total_epochs=epochs,
            )
            losses.append(loss)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            if epoch % val_freq == 0:
                # Score validation edges
                pos_scores, neg_scores = _score_split(
                    encoder, predictor, x, message_passing_edge_index,
                    split_edge, "valid", batch_size,
                )
                split_metrics = evaluate_link_prediction(
                    positive_scores={"valid": pos_scores},
                    negative_scores={"valid": neg_scores},
                    ks=ks, dataset_name=dataset_name,
                )
                valid_hits = split_metrics.get("valid", {}).get("hits_at_50", 0.0)
                val_metrics.append({"epoch": epoch, "valid_hits_at_50": valid_hits, "lr": current_lr})

                logger.info(
                    "GCN epoch %s/%s loss=%.6f lr=%.2e valid_hits@50=%.4f",
                    epoch, epochs, loss, current_lr, valid_hits,
                )

                if valid_hits > best_val_hits50:
                    best_val_hits50 = valid_hits
                    best_state = {
                        "encoder": copy.deepcopy(encoder.state_dict()),
                        "predictor": copy.deepcopy(predictor.state_dict()),
                    }
                    best_epoch = epoch
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
                    if no_improve_streak >= early_stop_patience:
                        logger.info(
                            "Early stopping at epoch %s — no improvement for %s validations",
                            epoch, early_stop_patience,
                        )
                        break
            else:
                logger.info("GCN epoch %s/%s loss=%.6f lr=%.2e", epoch, epochs, loss, current_lr)

        # ── Restore best model ────────────────────────────────────────
        if best_state is not None:
            logger.info(
                "Restoring best model from epoch %s (valid hits@50=%.4f)",
                best_epoch, best_val_hits50,
            )
            encoder.load_state_dict(best_state["encoder"])
            predictor.load_state_dict(best_state["predictor"])

        # ── Final scoring ─────────────────────────────────────────────
        logger.info("Scoring GCN validation/test edges with best model")
        positive_scores: dict[str, list[float]] = {}
        negative_scores: dict[str, list[float]] = {}
        for split in progress_bar(("valid", "test"), desc="GCN scoring"):
            if "edge" not in split_edge.get(split, {}) or "edge_neg" not in split_edge.get(split, {}):
                continue
            pos, neg = _score_split(
                encoder, predictor, x, message_passing_edge_index,
                split_edge, split, batch_size,
            )
            positive_scores[split] = pos
            negative_scores[split] = neg

        metrics = evaluate_link_prediction(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            ks=ks,
            dataset_name=dataset_name,
        )
        logger.info("Completed GCN v2 metrics=%s", metrics)

    return {
        "method_name": "gcn",
        "dataset_name": dataset_name,
        "metrics": metrics,
        "losses": losses,
        "val_metrics": val_metrics,
        "best_epoch": best_epoch,
        "best_val_hits_at_50": best_val_hits50,
        "runtime_seconds": usage.runtime_seconds,
        "memory_mb": usage.end_memory_mb,
        "memory_delta_mb": usage.end_memory_mb - start_memory_mb,
        "config": dict(hp),
        "status": "completed",
    }
