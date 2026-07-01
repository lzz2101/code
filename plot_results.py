from __future__ import annotations

from pathlib import Path


def _save_line_plot(values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(range(len(values)), values)
    ax.set_title(title)
    ax.set_xlabel("time slot")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_histories(histories: dict[str, list[float]], output_dir: str | Path) -> list[Path]:
    """Save the three Version 1 diagnostic figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_specs = [
        (
            "weighted_sum_aoi",
            "Weighted-sum AoI vs Time Slot",
            "weighted-sum AoI",
            "weighted_sum_aoi.png",
        ),
        (
            "handover_count",
            "Handover Count vs Time Slot",
            "handover count",
            "handover_count.png",
        ),
        (
            "edge_group_count",
            "Edge Group Count vs Time Slot",
            "edge group count",
            "edge_group_count.png",
        ),
    ]

    saved_paths: list[Path] = []
    for key, title, ylabel, filename in figure_specs:
        path = output_dir / filename
        _save_line_plot(histories[key], title, ylabel, path)
        saved_paths.append(path)

    return saved_paths
