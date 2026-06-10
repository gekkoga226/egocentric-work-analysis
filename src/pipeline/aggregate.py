# src/pipeline/aggregate.py
from src.schemas import SegmentList


def aggregate(seg_list: SegmentList) -> dict:
    """Compute category-level and label-level statistics.

    Returns:
        {
            "by_category": {cat: {"total_sec", "count", "ratio"}},
            "by_label":    {lbl: {"total_sec", "count", "mean_sec"}},
            "total_sec":   float,
        }
    """
    by_category: dict = {}
    by_label: dict = {}
    total_sec = 0.0

    for seg in seg_list.segments:
        dur = seg.end_sec - seg.start_sec
        total_sec += dur

        cat = seg.category or "unknown"
        entry = by_category.setdefault(cat, {"total_sec": 0.0, "count": 0, "ratio": 0.0})
        entry["total_sec"] += dur
        entry["count"] += 1

        lentry = by_label.setdefault(seg.label, {"total_sec": 0.0, "count": 0, "mean_sec": 0.0})
        lentry["total_sec"] += dur
        lentry["count"] += 1

    for data in by_category.values():
        data["ratio"] = data["total_sec"] / total_sec if total_sec > 0 else 0.0

    for data in by_label.values():
        data["mean_sec"] = data["total_sec"] / data["count"] if data["count"] > 0 else 0.0

    return {"by_category": by_category, "by_label": by_label, "total_sec": total_sec}
