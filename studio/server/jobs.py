"""A single background worker that runs GPU jobs one at a time and reports progress."""
import queue
import threading
import time
import traceback
import uuid
from typing import Callable


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, kind: str, title: str, fn: Callable[["Job"], dict]):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.fn = fn
        self.status = "queued"  # queued, running, done, error, cancelled
        self.progress = 0.0
        self.message = "Na fila"
        self.result: dict | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self._cancel = threading.Event()

    def update(self, progress: float | None = None, message: str | None = None):
        if self._cancel.is_set():
            raise Cancelled()
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "progress": round(self.progress, 3), "message": self.message,
                "result": self.result, "error": self.error, "created_at": self.created_at}


_jobs: dict[str, Job] = {}
_queue: "queue.Queue[Job]" = queue.Queue()


def _worker():
    while True:
        job = _queue.get()
        if job._cancel.is_set():
            continue
        job.status, job.message = "running", "Iniciando"
        try:
            job.result = job.fn(job)
            job.status, job.progress, job.message = "done", 1.0, "Concluído"
        except Cancelled:
            job.status, job.message = "cancelled", "Cancelado"
        except Exception as e:  # surfaced to the UI
            traceback.print_exc()
            job.status, job.error, job.message = "error", str(e) or e.__class__.__name__, "Falhou"


threading.Thread(target=_worker, name="job-worker", daemon=True).start()


def submit(kind: str, title: str, fn: Callable[[Job], dict]) -> Job:
    job = Job(kind, title, fn)
    _jobs[job.id] = job
    # Keep memory bounded: drop the oldest finished jobs.
    finished = [j for j in _jobs.values() if j.status in ("done", "error", "cancelled")]
    for old in sorted(finished, key=lambda j: j.created_at)[:-100]:
        _jobs.pop(old.id, None)
    _queue.put(job)
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def active() -> list[dict]:
    return [j.as_dict() for j in _jobs.values() if j.status in ("queued", "running")]


def cancel(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if not job or job.status not in ("queued", "running"):
        return False
    job._cancel.set()
    if job.status == "queued":
        job.status, job.message = "cancelled", "Cancelado"
    return True
