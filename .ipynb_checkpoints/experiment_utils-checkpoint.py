import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import swanlab  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None


def create_run_dirs(dataset: str, base_dir: str = "runs") -> Dict[str, str]:
    """Create a timestamped folder structure for one training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{timestamp}_{dataset}")
    artifacts_dir = os.path.join(run_dir, "artifacts")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "artifacts": artifacts_dir,
        "plots": plots_dir,
        "timestamp": timestamp,
    }


def _json_default(o: Any):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False, default=_json_default)


def init_swanlab_run(args, run_meta: Dict[str, str]):
    """Initialize SwanLab experiment if the package is available."""
    if swanlab is None:
        print("SwanLab is not installed. Install it with `pip install swanlab` to enable logging.")
        return None
    return swanlab.init(
        project="KAN-MCP",
        experiment_name=f"{args.dataset}-{run_meta['timestamp']}",
        config=vars(args),
    )


def log_swanlab_metrics(run, metrics: Dict[str, float], step: int) -> None:
    if run is None or swanlab is None:
        return
    swanlab.log(metrics, step=step)


def save_model_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_npz(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **data)


def plot_kan_contribution(
    model: torch.nn.Module,
    batch,
    compressed_dim: int,
    device: torch.device,
    save_path: str,
) -> Dict[str, Any]:
    """Plot input contribution for KAN head using one batch of data."""
    model.eval()
    with torch.no_grad():
        input_ids, visual, acoustic, label_ids = (t.to(device) for t in batch)
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

        outputs = model(
            input_ids,
            visual_norm,
            acoustic_norm,
            label_ids,
        )
        kan_inputs = outputs["concat_feature"].detach()

    if not hasattr(model, "KAN"):
        raise RuntimeError("KAN classifier is not attached to this model.")

    model.KAN.save_act = True
    model.KAN.cache_data = kan_inputs
    model.KAN.get_act(kan_inputs)
    scores = model.KAN.feature_score
    if isinstance(scores, torch.Tensor) and scores.ndim > 1:
        scores = torch.mean(scores, dim=0)

    score_np = scores.detach().cpu().flatten().numpy()
    labels = []
    for prefix in ("text", "audio", "visual"):
        for idx in range(compressed_dim):
            labels.append(f"{prefix}_{idx + 1}")

    modality_totals = {
        "text": float(np.sum(score_np[:compressed_dim])),
        "audio": float(np.sum(score_np[compressed_dim:2 * compressed_dim])),
        "visual": float(np.sum(score_np[2 * compressed_dim:])),
    }

    plt.figure(figsize=(max(8, len(score_np) * 0.6), 4))
    plt.bar(range(len(score_np)), score_np)
    plt.xticks(range(len(score_np)), labels, rotation=45, ha="right")
    plt.title("KAN contribution to output")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "per_dim": {label: float(score_np[idx]) for idx, label in enumerate(labels)},
        "per_modality": modality_totals,
        "plot_path": save_path,
    }


def plot_kan_tree(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    save_path: str,
) -> str:
    """Render a structural tree plot for KAN using the same batch as contribution plot."""
    from kan.hypothesis import plot_tree  # local import to avoid heavy load when unused

    if not hasattr(model, "KAN"):
        raise RuntimeError("KAN classifier is not attached to this model.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    with torch.no_grad():
        input_ids, visual, acoustic, label_ids = (t.to(device) for t in batch)
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)
        outputs = model(
            input_ids,
            visual_norm,
            acoustic_norm,
            label_ids,
        )
        kan_inputs = outputs["concat_feature"].detach()

    old_show = plt.show
    try:
        plt.show = lambda *args, **kwargs: None
        plot_tree(model.KAN, kan_inputs, style="tree")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    except Exception as exc:  # pragma: no cover - visualization may fail in headless
        print(f"KAN tree plot failed: {exc}")
    finally:
        plt.show = old_show
        plt.close()

    return save_path


def plot_kan_model_diagram(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    save_path: str,
    metric: str = "backward",
    scale: float = 0.8,
) -> str:
    """Use the built-in `model.KAN.plot()` to save the structural diagram to `save_path`.

    This runs a forward pass on ``batch`` to populate cached activations, calls ``model.KAN.plot``,
    then copies the resulting PNG from the default ``kan_plot`` directory into ``save_path``.
    """
    if not hasattr(model, "KAN"):
        raise RuntimeError("KAN classifier is not attached to this model.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_folder = os.path.join(os.path.dirname(save_path), "_kan_plot_tmp")

    model.eval()
    with torch.no_grad():
        input_ids, visual, acoustic, label_ids = (t.to(device) for t in batch)
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)
        outputs = model(
            input_ids,
            visual_norm,
            acoustic_norm,
            label_ids,
        )
        kan_inputs = outputs["concat_feature"].detach()

    model.KAN.save_act = True
    model.KAN.cache_data = kan_inputs

    file_name = os.path.basename(save_path)
    try:
        os.makedirs(tmp_folder, exist_ok=True)
        model.KAN.plot(file_name=file_name, folder=tmp_folder, metric=metric, scale=scale)
        src_path = os.path.join("kan_plot", file_name if file_name.endswith(".png") else f"{file_name}.png")
        if os.path.exists(src_path):
            dest_path = save_path if save_path.endswith(".png") else f"{save_path}.png"
            shutil.copy(src_path, dest_path)
            try:
                os.remove(src_path)
            except OSError:
                pass
        else:  # pragma: no cover - diagnostic only
            print(f"KAN model plot not found at {src_path}")
    except Exception as exc:  # pragma: no cover - visualization may fail in headless
        print(f"KAN model plot failed: {exc}")
    finally:
        shutil.rmtree(tmp_folder, ignore_errors=True)
        if os.path.isdir("kan_plot") and not os.listdir("kan_plot"):
            shutil.rmtree("kan_plot", ignore_errors=True)

    return save_path if save_path.endswith(".png") else f"{save_path}.png"
