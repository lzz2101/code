from __future__ import annotations

from pathlib import Path


def _scale_points(values, x0, y0, width, height):
    values = [float(value) for value in values]
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-12)
    denom = max(len(values) - 1, 1)
    points = []
    for idx, value in enumerate(values):
        x = x0 + int(width * idx / denom)
        y = y0 + height - int(height * (value - min_value) / span)
        points.append((x, y))
    return points


def _fallback_line_plot(
    series, title: str, ylabel: str, output_path: Path
) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 480), "white")
    draw = ImageDraw.Draw(image)
    x0, y0, width, height = 70, 70, 760, 320
    draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(40, 40, 40))
    draw.text((30, 20), title, fill=(0, 0, 0))
    draw.text((30, 430), f"x: time slot    y: {ylabel}", fill=(0, 0, 0))

    for values, color, label in series:
        points = _scale_points(values, x0, y0, width, height)
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        elif points:
            draw.line(points, fill=color, width=2)

    legend_x = 650
    legend_y = 20
    for _, color, label in series:
        draw.line((legend_x, legend_y + 6, legend_x + 25, legend_y + 6), fill=color, width=3)
        draw.text((legend_x + 32, legend_y), label, fill=(0, 0, 0))
        legend_y += 20

    image.save(output_path)


def _fallback_histogram(values, title: str, xlabel: str, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    values = [int(value) for value in values]
    counts = {value: values.count(value) for value in sorted(set(values))}
    image = Image.new("RGB", (700, 450), "white")
    draw = ImageDraw.Draw(image)
    x0, y0, width, height = 60, 60, 560, 300
    draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(40, 40, 40))
    draw.text((30, 20), title, fill=(0, 0, 0))
    draw.text((30, 400), f"x: {xlabel}    y: count", fill=(0, 0, 0))
    if counts:
        bar_width = max(12, int(width / max(1, len(counts) * 1.5)))
        max_count = max(counts.values())
        gap = int(width / max(1, len(counts)))
        for idx, (value, count) in enumerate(counts.items()):
            x = x0 + idx * gap + gap // 4
            bar_height = int(height * count / max_count)
            draw.rectangle(
                (x, y0 + height - bar_height, x + bar_width, y0 + height),
                fill=(72, 111, 175),
            )
            draw.text((x, y0 + height + 5), str(value), fill=(0, 0, 0))
    image.save(output_path)


def _plot_line(values, title: str, ylabel: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_line_plot(
            [(values, (72, 111, 175), ylabel)],
            title,
            ylabel,
            output_path,
        )
        return

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
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_histogram(values, title, xlabel, output_path)
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=sorted(set(values)) + [max(values) + 1], align="left", rwidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_power_allocation(
    histories: dict[str, list[float]], output_path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _fallback_line_plot(
            [
                (histories["edge_power"], (72, 111, 175), "edge"),
                (histories["nonedge_power"], (221, 132, 82), "non-edge"),
                (histories["total_allocated_power"], (85, 168, 104), "total"),
            ],
            "Algorithm 3 Power Allocation, Constellation Total",
            "normalized power across all satellites",
            output_path,
        )
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(histories["total_allocated_power"]))
    ax.plot(x, histories["edge_power"], label="edge allocated power")
    ax.plot(x, histories["nonedge_power"], label="non-edge allocated power")
    ax.plot(x, histories["total_allocated_power"], label="constellation total")
    ax.set_title("Algorithm 3 Power Allocation, Constellation Total")
    ax.set_xlabel("time slot")
    ax.set_ylabel("normalized power across all satellites")
    ax.legend()
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

    if {
        "edge_power",
        "nonedge_power",
        "total_allocated_power",
    }.issubset(histories):
        power_path = output_dir / "power_allocation.png"
        _plot_power_allocation(histories, power_path)
        saved_paths.append(power_path)
    return saved_paths
