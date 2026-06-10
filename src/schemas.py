# src/schemas.py
from dataclasses import dataclass, asdict
import json
from typing import Optional


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    label: str
    confidence: float = 1.0
    category: Optional[str] = None      # "seimi" | "fuzui" | "muda"
    description: Optional[str] = None   # what is observed (factual)
    improvement: Optional[str] = None   # kaizen hint (muda/fuzui only)


@dataclass
class Hint:
    label: str
    frame_sec: float
    bbox: Optional[tuple] = None        # normalized (x, y, w, h); None = whole frame
    note: Optional[str] = None


@dataclass
class SegmentList:
    video_id: str
    fps_sampled: float
    label_vocabulary: list[str]
    segments: list[Segment]
    source: str  # "track_std" | "track_a" | "track_b" | "ground_truth" | "ollo"

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SegmentList":
        d = json.loads(s)
        known = set(Segment.__dataclass_fields__)
        d["segments"] = [
            Segment(**{k: v for k, v in seg.items() if k in known})
            for seg in d["segments"]
        ]
        return cls(**d)

    def to_frame_labels(self, total_duration: float, fps: float = 1.0) -> list[str]:
        n = int(total_duration * fps)
        labels = ["background"] * n
        for seg in self.segments:
            start_i = int(seg.start_sec * fps)
            end_i = int(seg.end_sec * fps)
            for i in range(start_i, min(end_i, n)):
                labels[i] = seg.label
        return labels
