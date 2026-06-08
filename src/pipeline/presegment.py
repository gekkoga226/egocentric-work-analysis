import numpy as np
import ruptures as rpt
import logging

logger = logging.getLogger(__name__)


def detect_boundaries(
    timestamps: list[float],
    embeddings: np.ndarray,
    penalty: float = 10.0,
    min_segment_sec: float = 5.0,
) -> list[float]:
    if len(timestamps) < 2 or embeddings.shape[0] < 2:
        return []

    algo = rpt.Pelt(model="rbf", min_size=2).fit(embeddings)
    breakpoints = algo.predict(pen=penalty)
    raw_indices = breakpoints[:-1]

    boundaries: list[float] = []
    prev_ts = timestamps[0]
    for idx in raw_indices:
        clamped = min(idx, len(timestamps) - 1)
        ts = float(timestamps[clamped])
        if ts - prev_ts >= min_segment_sec:
            boundaries.append(ts)
            prev_ts = ts

    return boundaries
