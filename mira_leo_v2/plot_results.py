from __future__ import annotations

from pathlib import Path


def _plot_line(values, title: str, ylabel: str, output_path: Path) -> None:
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


def _plot_histogram(values, title: str, xlabel: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=sorted(set(values)) + [max(values) + 1], align="left", rwidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_histories(histories: dict[str, list[float]], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    specs = [
        ("action", "Selected Regrouping Action vs Time", "a(t)", "action.png"),
        ("reward", "Observed Reward vs Time", "reward", "reward.png"),
        (
            "weighted_sum_aoi",
            "Weighted-sum AoI vs Time",
            "weighted-sum AoI",
            "weighted_sum_aoi.png",
        ),
    ]
    for key, title, ylabel, filename in specs:
        path = output_dir / filename
        _plot_line(histories[key], title, ylabel, path)
        saved_paths.append(path)

    hist_path = output_dir / "action_distribution.png"
    _plot_histogram(histories["action"], "Action Distribution", "action", hist_path)
    saved_paths.append(hist_path)
    return saved_paths
