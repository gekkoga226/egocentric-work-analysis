# src/pipeline/report.py
from pathlib import Path
from src.schemas import SegmentList

_CAT_NAMES = {
    "seimi": "正味作業",
    "fuzui": "付随作業",
    "muda":  "ムダ作業",
}


def to_timeline_markdown(seg_list: SegmentList) -> str:
    lines = [
        f"# 作業タイムライン: {seg_list.video_id}",
        "",
        "| # | 開始 | 終了 | 要素作業 | 分類 | 信頼度 |",
        "|---|------|------|----------|------|--------|",
    ]
    for i, seg in enumerate(seg_list.segments, 1):
        cat = _CAT_NAMES.get(seg.category or "", "—")
        lines.append(
            f"| {i} | {_fmt(seg.start_sec)} | {_fmt(seg.end_sec)} "
            f"| {seg.label} | {cat} | {seg.confidence:.2f} |"
        )
    return "\n".join(lines)


def to_procedure_markdown(seg_list: SegmentList) -> str:
    lines = [
        f"# 標準作業手順書（ドラフト）: {seg_list.video_id}",
        "",
        "> 生成AI分析による自動ドラフト。内容を確認・編集してください。",
        "",
    ]
    for i, seg in enumerate(seg_list.segments, 1):
        duration = seg.end_sec - seg.start_sec
        lines += [
            f"## Step {i}: {seg.label}",
            "",
            f"- **所要時間**: {duration:.1f}秒 ({_fmt(seg.start_sec)} ～ {_fmt(seg.end_sec)})",
            f"- **信頼度**: {seg.confidence:.2f}",
        ]
        if seg.category and seg.category in _CAT_NAMES:
            lines.append(f"- **分類**: {_CAT_NAMES[seg.category]}")
        if seg.description:
            lines.append(f"- **内容**: {seg.description}")
        if seg.improvement:
            lines.append(f"- **改善ヒント**: {seg.improvement}")
        lines.append("")
    return "\n".join(lines)


def save_segments(seg_list: SegmentList, output_dir: str) -> str:
    out = Path(output_dir) / f"{seg_list.video_id}_{seg_list.source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(seg_list.to_json(), encoding="utf-8")
    return str(out)


def _fmt(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"
