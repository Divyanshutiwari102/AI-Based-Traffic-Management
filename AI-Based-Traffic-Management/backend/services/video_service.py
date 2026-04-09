from __future__ import annotations

from pathlib import Path

from config import DEFAULT_LANE_CAPACITY
from detector import detect_video, weighted_density


def process_videos(video_paths: list[str]) -> dict:
    lane_names = ["north", "south", "west", "east"]
    lane_counts = {}
    lane_density = {}

    for lane_name, path in zip(lane_names, video_paths):
        counts = detect_video(str(Path(path)))
        lane_counts[lane_name] = counts
        lane_density[lane_name] = weighted_density(counts, DEFAULT_LANE_CAPACITY)

    return {
        "lane_counts": lane_counts,
        "lane_density": lane_density,
    }
