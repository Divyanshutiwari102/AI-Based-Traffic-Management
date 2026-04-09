from __future__ import annotations

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from config import MAX_TASK_WORKERS
from optimizer.ga_optimizer import optimize_traffic
from services.video_service import process_videos

_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_TASK_WORKERS)
_TASKS: dict[str, Future] = {}


@dataclass
class TaskHandle:
    id: str


class ProcessVideosTask:
    def delay(self, video_paths: list[str]) -> TaskHandle:
        job_id = str(uuid.uuid4())
        _TASKS[job_id] = _EXECUTOR.submit(self._run, video_paths)
        return TaskHandle(id=job_id)

    @staticmethod
    def _run(video_paths: list[str]) -> dict:
        video_result = process_videos(video_paths)
        densities = [
            video_result["lane_density"]["north"],
            video_result["lane_density"]["south"],
            video_result["lane_density"]["west"],
            video_result["lane_density"]["east"],
        ]
        timings = optimize_traffic(densities)
        return {
            "timings": timings,
            "lane_density": video_result["lane_density"],
            "lane_counts": video_result["lane_counts"],
        }


process_videos_task = ProcessVideosTask()


def get_task_status(job_id: str) -> dict:
    future = _TASKS.get(job_id)
    if future is None:
        return {"state": "NOT_FOUND", "error": "Job ID not found"}

    if not future.done():
        return {"state": "PENDING"}

    exc = future.exception()
    if exc is not None:
        return {"state": "FAILURE", "error": str(exc)}

    return {"state": "SUCCESS", "result": future.result()}
