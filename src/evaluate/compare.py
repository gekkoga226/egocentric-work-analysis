from src.schemas import SegmentList
from src.evaluate.metrics import compute_all


def compare_systems(
    ground_truth: SegmentList,
    predictions: dict[str, SegmentList],
    fps: float = 1.0,
) -> dict[str, dict[str, float]]:
    return {
        name: compute_all(pred, ground_truth, fps)
        for name, pred in predictions.items()
    }


def comparison_report(results: dict[str, dict[str, float]]) -> str:
    metrics = ["f1@10", "f1@25", "f1@50", "edit", "acc"]
    header = "| System | " + " | ".join(m.upper() for m in metrics) + " |"
    sep = "|--------|" + "|".join("-------" for _ in metrics) + "|"
    rows = [header, sep]
    for sys in sorted(results):
        vals = results[sys]
        row = f"| {sys} | " + " | ".join(f"{vals[m]:.3f}" for m in metrics) + " |"
        rows.append(row)
    return "\n".join(rows)
