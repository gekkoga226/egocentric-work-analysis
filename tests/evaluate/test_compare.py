import pytest
from src.evaluate.compare import compare_systems, comparison_report


def test_compare_systems_keys(ground_truth_segments, perfect_prediction):
    import copy
    pred_b = perfect_prediction
    pred_a = copy.deepcopy(perfect_prediction)
    pred_a.source = "track_a"

    results = compare_systems(
        ground_truth_segments,
        {"track_a": pred_a, "track_b": pred_b},
        fps=1.0,
    )
    assert set(results.keys()) == {"track_a", "track_b"}
    assert set(results["track_a"].keys()) == {"f1@10", "f1@25", "f1@50", "edit", "acc"}


def test_compare_perfect_scores(ground_truth_segments, perfect_prediction):
    results = compare_systems(
        ground_truth_segments,
        {"track_b": perfect_prediction},
        fps=1.0,
    )
    assert results["track_b"]["f1@50"] == pytest.approx(1.0)
    assert results["track_b"]["acc"] == pytest.approx(1.0)


def test_comparison_report_is_markdown_table(ground_truth_segments, perfect_prediction):
    results = compare_systems(
        ground_truth_segments,
        {"track_b": perfect_prediction},
        fps=1.0,
    )
    report = comparison_report(results)
    assert "| track_b |" in report
    assert "F1@10" in report
    assert "EDIT" in report
