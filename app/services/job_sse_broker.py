"""
In-memory SSE event broker (replay + fan-out). One worker / process only.
For multiple server instances use Redis pub/sub or a queue service.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

MAX_REPLAY = 150

_replay: Dict[str, List[dict]] = {}
_subscribers: DefaultDict[str, List[asyncio.Queue]] = defaultdict(list)
_lock = asyncio.Lock()


async def publish(job_id: str, event: dict[str, Any]) -> None:
    """Append to replay buffer and push to all live SSE subscribers."""
    async with _lock:
        if job_id not in _replay:
            _replay[job_id] = []
        _replay[job_id].append(event)
        if len(_replay[job_id]) > MAX_REPLAY:
            _replay[job_id] = _replay[job_id][-MAX_REPLAY:]
        queues = list(_subscribers.get(job_id, []))

    for q in queues:
        try:
            await q.put(event)
        except Exception:
            pass


def replay_snapshot(job_id: str) -> List[dict]:
    return list(_replay.get(job_id, []))


async def attach_subscriber(job_id: str) -> asyncio.Queue:
    async with _lock:
        q: asyncio.Queue = asyncio.Queue()
        _subscribers[job_id].append(q)
        return q


async def detach_subscriber(job_id: str, q: asyncio.Queue) -> None:
    async with _lock:
        lst = _subscribers.get(job_id)
        if lst and q in lst:
            lst.remove(q)
