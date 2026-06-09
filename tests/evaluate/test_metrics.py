import pytest
from src.schemas import Segment, SegmentList
from src.evaluate.metrics import f1_at_k, edit_score, frame_accuracy, compute_all


def _make_sl(segments, source="pred"):
    return SegmentList("v", 1.0, ["A", "B", "C"], segments, source)


def test_f1_perfect_match():
    gt = _make_sl([Segment(0.0, 10.0, "A", 1.0), Segment(10.0, 20.0, "B", 1.0)], "gt")
    pred = _make_sl([Segment(0.0, 10.0, "A", 1.0), Segment(10.0, 20.0, "B", 1.0)])
    assert f1_at_k(pred, gt, 0.5) == pytest.approx(1.0)


def test_f1_no_match():
    gt = _make_sl([Segment(0.0, 10.0, "A", 1.0)], "gt")
    pred = _make_sl([Segment(0.0, 10.0, "B", 1.0)])
    assert f1_at_k(pred, gt, 0.5) == pytest.approx(0.0)


def test_f1_partial_overlap_below_threshold():
    gt = _make_sl([Segment(0.0, 10.0, "A", 1.0)], "gt")
    pred = _make_sl([Segment(0.0, 6.0, "A", 1.0)])
    assert f1_at_k(pred, gt, 0.5) == pytest.approx(1.0)

    pred2 = _make_sl([Segment(0.0, 4.0, "A", 1.0)])
    assert f1_at_k(pred2, gt, 0.5) == pytest.approx(0.0)


def test_edit_score_perfect():
    gt = _make_sl([Segment(0.0, 10.0, "A"), Segment(10.0, 20.0, "B")], "gt")
    pred = _make_sl([Segment(0.0, 10.0, "A"), Segment(10.0, 20.0, "B")])
    assert edit_score(pred, gt) == pytest.approx(100.0)


def test_edit_score_all_wrong():
    gt = _make_sl([Segment(0.0, 10.0, "A"), Segment(10.0, 20.0, "B")], "gt")
    pred = _make_sl([Segment(0.0, 10.0, "C"), Segment(10.0, 20.0, "C")])
    assert edit_score(pred, gt) < 100.0


def test_frame_accuracy_perfect(ground_truth_segments, perfect_prediction):
    acc = frame_accuracy(perfect_prediction, ground_truth_segments, fps=1.0)
    assert acc == pytest.approx(1.0)


def test_compute_all_returns_all_keys(ground_truth_segments, perfect_prediction):
    result = compute_all(perfect_prediction, ground_truth_segments, fps=1.0)
    assert set(result.keys()) == {"f1@10", "f1@25", "f1@50", "edit", "acc"}
    assert all(0.0 <= v <= 100.0 or v <= 1.0 for v in result.values())
