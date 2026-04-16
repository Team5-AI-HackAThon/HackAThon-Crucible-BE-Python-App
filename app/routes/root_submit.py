"""
Top-level submit + SSE (matches clients calling POST /submit-async with JSON).
"""

from __future__ import annotations

import asyncio
import json as _json
import uuid as uuid_stdlib

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.models import JsonSubmitAsyncAccepted, SubmitAsyncJsonRequest
from app.services.async_indexer_job import run_sentiment_indexer_job_sse
from app.services.job_sse_broker import (
    attach_subscriber,
    detach_subscriber,
    publish as sse_publish,
    replay_snapshot,
)
from app.services.supabase_service import (
    get_media_download_url,
    get_row_status,
    insert_sentiment_pending,
)

import os

router = APIRouter(tags=["submit"])


@router.post(
    "/submit-async",
    response_model=JsonSubmitAsyncAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_async_json(
    body: SubmitAsyncJsonRequest,
    background_tasks: BackgroundTasks,
):
    """
    Accept JSON describing an existing Storage object; create `sentiment_outputs` with
    `is_processed=false`, return **202** immediately, then run Video Indexer in the background.

    Subscribe to **GET /submit-async/events/{sentiment_output_id}** (SSE) for stage updates.
    """
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase is not configured (SUPABASE_URL, SUPABASE_KEY).",
        )

    ma = body.media_asset
    try:
        uuid_stdlib.UUID(ma.id.strip())
        uuid_stdlib.UUID(ma.owner_id.strip())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="media_asset.id and media_asset.owner_id must be valid UUIDs",
        )
    if ma.project_id:
        try:
            uuid_stdlib.UUID(ma.project_id.strip())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="media_asset.project_id must be a valid UUID when provided",
            )

    try:
        media_url = get_media_download_url(ma.storage_bucket.strip(), ma.storage_path.strip())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not build media URL (check bucket/path and Supabase key): {e}",
        ) from e

    try:
        sentiment_output_id = insert_sentiment_pending(ma.id.strip(), media_url)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create sentiment_outputs row: {e}",
        ) from e

    await sse_publish(
        sentiment_output_id,
        {
            "stage": "queued",
            "message": "sentiment_outputs row created; starting background indexer",
            "progress": 2,
            "sentiment_output_id": sentiment_output_id,
            "media_asset_id": ma.id.strip(),
            "media_url": media_url,
        },
    )

    background_tasks.add_task(run_sentiment_indexer_job_sse, sentiment_output_id)

    return JsonSubmitAsyncAccepted(
        sentiment_output_id=sentiment_output_id,
        media_asset_id=ma.id.strip(),
        media_url=media_url,
        sse_events_path=f"/submit-async/events/{sentiment_output_id}",
        status_poll_path=f"/api/v1/sentiment/status/{sentiment_output_id}",
    )


@router.get("/submit-async/events/{sentiment_output_id}")
async def submit_async_events_sse(sentiment_output_id: str, request: Request):
    """
    Server-Sent Events stream for one job. Replays past events first (if any), then streams    new ones until `stage` is `done` or `error`.
    """
    row = get_row_status(sentiment_output_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Row not found")

    async def event_gen():
        q = None
        try:
            for ev in replay_snapshot(sentiment_output_id):
                yield f"data: {_json.dumps(ev)}\n\n"
            q = await attach_subscriber(sentiment_output_id)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {_json.dumps(ev)}\n\n"
                    if ev.get("stage") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            if q is not None:
                await detach_subscriber(sentiment_output_id, q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
