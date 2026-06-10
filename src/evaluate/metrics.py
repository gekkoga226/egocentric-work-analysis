from src.schemas import SegmentList, Segment


def f1_at_k(pred: SegmentList, gt: SegmentList, k: float) -> float:
    pred_segs, gt_segs = pred.segments, gt.segments
    pred_matched = [False] * len(pred_segs)
    gt_matched = [False] * len(gt_segs)
    tp = 0

    for i, p in enumerate(pred_segs):
        for j, g in enumerate(gt_segs):
            if gt_matched[j] or p.label != g.label:
                continue
            overlap = max(0.0, min(p.end_sec, g.end_sec) - max(p.start_sec, g.start_sec))
            union = max(p.end_sec, g.end_sec) - min(p.start_sec, g.start_sec)
            if union > 0 and overlap / union >= k:
                tp += 1
                pred_matched[i] = True
                gt_matched[j] = True
                break

    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def edit_score(pred: SegmentList, gt: SegmentList) -> float:
    a = [s.label for s in pred.segments]
    b = [s.label for s in gt.segments]
    dist = _levenshtein(a, b)
    max_len = max(len(a), len(b))
    return (1.0 - dist / max_len) * 100.0 if max_len > 0 else 100.0


def frame_accuracy(pred: SegmentList, gt: SegmentList, fps: float = 1.0) -> float:
    total = max(
        (max(s.end_sec for s in gt.segments) if gt.segments else 0.0),
        (max(s.end_sec for s in pred.segments) if pred.segments else 0.0),
    )
    if total == 0:
        return 1.0
    pred_f = pred.to_frame_labels(total, fps)
    gt_f = gt.to_frame_labels(total, fps)
    n = min(len(pred_f), len(gt_f))
    return sum(p == g for p, g in zip(pred_f[:n], gt_f[:n])) / n if n > 0 else 0.0


def compute_all(pred: SegmentList, gt: SegmentList, fps: float = 1.0) -> dict[str, float]:
    return {
        "f1@10": f1_at_k(pred, gt, 0.10),
        "f1@25": f1_at_k(pred, gt, 0.25),
        "f1@50": f1_at_k(pred, gt, 0.50),
        "edit":  edit_score(pred, gt),
        "acc":   frame_accuracy(pred, gt, fps),
    }


def boundary_deviation_log(
    pred: SegmentList,
    gt: SegmentList,
) -> dict[str, list[float]]:
    """Return per-boundary deviation between pred and gt boundary timestamps.

    NOTE: track_std's Gemini boundary refinement may move boundaries 1-2s from the
    annotator's physical timestamps to IE-logical positions (e.g., 'tool touch = start').
    A lower F1@10 vs track_b does NOT necessarily mean worse quality — sample-review
    the video to verify logical correctness before drawing conclusions.
    """
    pred_bounds = sorted({s.start_sec for s in pred.segments} | {s.end_sec for s in pred.segments})
    gt_bounds   = sorted({s.start_sec for s in gt.segments}   | {s.end_sec for s in gt.segments})

    deviations: dict[str, list[float]] = {
        "matched_deviations": [],   # |pred_b - nearest_gt_b| for matched pairs
        "unmatched_pred": [],       # pred boundaries with no gt within 50s
        "unmatched_gt": [],         # gt boundaries with no pred within 50s
    }

    gt_used = set()
    for pb in pred_bounds:
        candidates = [(abs(pb - gb), i) for i, gb in enumerate(gt_bounds) if i not in gt_used]
        if not candidates:
            deviations["unmatched_pred"].append(pb)
            continue
        dist, idx = min(candidates)
        if dist <= 50.0:
            deviations["matched_deviations"].append(dist)
            gt_used.add(idx)
        else:
            deviations["unmatched_pred"].append(pb)

    for i, gb in enumerate(gt_bounds):
        if i not in gt_used:
            deviations["unmatched_gt"].append(gb)

    return deviations


def _levenshtein(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
