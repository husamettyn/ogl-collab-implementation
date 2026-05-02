"""Dataset loading helpers for the ogbl-collab link prediction task."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.metrics import DEFAULT_DATASET_NAME
from src.experiments.paths import DATASET_DIR


@dataclass(slots=True)
class CollabDataBundle:
    """Project-level container for OGB collab data and split information."""

    dataset_name: str
    dataset_root: Path
    data: Any
    split_edge: dict[str, dict[str, Any]]


def load_collab_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    root: Path | str = DATASET_DIR,
) -> Any:
    """Load the OGB PyG link prediction dataset."""
    from ogb.linkproppred import PygLinkPropPredDataset

    return PygLinkPropPredDataset(name=dataset_name, root=str(root))


def load_edge_split(dataset: Any) -> dict[str, dict[str, Any]]:
    """Return the official train/validation/test edge split."""
    return dataset.get_edge_split()


def load_collab_data_bundle(
    dataset_name: str = DEFAULT_DATASET_NAME,
    root: Path | str = DATASET_DIR,
) -> CollabDataBundle:
    """Load dataset, graph data, and official edge splits together."""
    dataset_root = Path(root)
    dataset = load_collab_dataset(dataset_name=dataset_name, root=dataset_root)
    data = dataset[0]
    split_edge = load_edge_split(dataset)

    return CollabDataBundle(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        data=data,
        split_edge=split_edge,
    )
