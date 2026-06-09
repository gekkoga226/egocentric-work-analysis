# src/schemas.py
from dataclasses import dataclass, asdict
import json


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    label: str
    confidence: float = 1.0


@dataclass
class SegmentList:
    video_id: str
    fps_sampled: float
    label_vocabulary: list[str]
    segments: list[Segment]
    source: str  # "track_a" | "track_b" | "ground_truth" | "ollo"

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SegmentList":
        d = json.loads(s)
        d["segments"] = [Segment(**seg) for seg in d["segments"]]
        return cls(**d)

    def to_frame_labels(self, total_duration: float, fps: float = 1.0) -> list[str]:
        """Convert to frame-level label list at given fps."""
        n = int(total_duration * fps)
        labels = ["background"] * n
        for seg in self.segments:
            start_i = int(seg.start_sec * fps)
            end_i = int(seg.end_sec * fps)
            for i in range(start_i, min(end_i, n)):
                labels[i] = seg.label
        return labels
