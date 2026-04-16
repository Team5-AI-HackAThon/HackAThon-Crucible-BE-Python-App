import uuid as uuid_stdlib
from typing import Optional

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Request, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import os
import tempfile
import asyncio
import json as _json
import time

from app.models import AsyncVideoSubmitResponse, VideoSentimentResponse
from app.services.azure_video_indexer_service import AzureVideoIndexerService
from app.services.llm_service import LLMService
from app.services.scorer import run_b_layer


router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])
azure_video_indexer_service = AzureVideoIndexerService()
llm_service = LLMService()

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".wmv", ".webm"}


def background_index_sentiment_output(sentiment_output_id: str) -> None:
    """Runs after HTTP 202: Video Indexer URL job + persist scores (same logic as process-queue row)."""
    from app.services.supabase_service import (
        fetch_sentiment_output_for_job,
        set_job_meta,
        store_raw_output,
        update_azure_video_indexer_id,
        update_processed,
    )

    try:
        set_job_meta(sentiment_output_id, "running", None)
    except Exception as e:
        print(f"[async-job] could not mark running id={sentiment_output_id}: {e}", flush=True)
        return

    try:
        row = fetch_sentiment_output_for_job(sentiment_output_id)
        media_url = row.get("media_url")
        if not media_url:
            set_job_meta(sentiment_output_id, "failed", "No media_url on sentiment_outputs row")
            return

        if not azure_video_indexer_service.is_configured():
            set_job_meta(
                sentiment_output_id,
                "failed",
                "Azure Video Indexer is not configured on the server.",
            )
            return

        asset_id = row.get("media_asset_id") or sentiment_output_id
        print(f"[async-job] Video Indexer start id={sentiment_output_id}", flush=True)
        indexer_result = azure_video_indexer_service.analyze_video_url(
            video_url=media_url,
            name=f"asset_{asset_id}",
        )
        vi_id = indexer_result.get("video_id")
        if vi_id:
            update_azure_video_indexer_id(sentiment_output_id, str(vi_id))
        store_raw_output(sentiment_output_id, indexer_result["raw_index_data"])
        pitch = run_b_layer(indexer_result["raw_index_data"])
        scores = {
            "team_strength": pitch["team_strength"]["score"],
            "technical_strength": pitch["technical_strength"]["score"],
            "innovation": pitch["innovation"]["score"],
            "credibility": pitch["credibility"]["score"],
            "confidence": pitch["confidence"]["score"],
        }
        gpt_sentiment = {}
        try:
            gpt_sentiment = llm_service.analyze_sentiment_with_gpt(
                indexer_result["transcript"], video_data=indexer_result
            )
        except Exception as e:
            print(f"[async-job] GPT sentiment failed (non-fatal): {e}", flush=True)
        transcript = (indexer_result.get("transcript") or "").strip()
        summary = transcript[:16000] if transcript else None
        update_processed(sentiment_output_id, scores, gpt_sentiment, video_transcript_summary=summary)
        print(f"[async-job] complete id={sentiment_output_id}", flush=True)
    except Exception as e:
        print(f"[async-job] ERROR id={sentiment_output_id}: {e}", flush=True)
        try:
            set_job_meta(sentiment_output_id, "failed", str(e))
        except Exception:
            pass


def _gpt_as_dict(gpt_sentiment) -> dict:
    if gpt_sentiment is None:
        return {}
    if isinstance(gpt_sentiment, dict):
        return gpt_sentiment
    if hasattr(gpt_sentiment, "model_dump"):
        return gpt_sentiment.model_dump()
    return dict(gpt_sentiment)


@router.post("/video", response_model=VideoSentimentResponse)
async def analyze_video_sentiment(
    file: UploadFile = File(...),
    supabase_row_id: Optional[str] = Form(
        default=None,
        description=(
            "Optional sentiment_outputs.id: after analysis, raw VI JSON and scores "
            "are written to this row (same as process-queue)."
        ),
    ),
):
    """Upload a video file, extract transcript via Azure Video Indexer, then analyze sentiment."""
    temp_path = None
    try:
        print("[video] /api/v1/sentiment/video request received", flush=True)
        filename = file.filename or "uploaded_video"
        extension = os.path.splitext(filename.lower())[1]
        print(f"[video] incoming file name={filename}, ext={extension}, content_type={file.content_type}", flush=True)

        if extension not in SUPPORTED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported video extension: {extension or 'unknown'}. "
                    "Supported formats are .mp4, .mov, .avi, .wmv, .webm"
                ),
            )

        if not azure_video_indexer_service.is_configured():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Azure Video Indexer is not configured. "
                    "Set AZURE_VIDEO_INDEXER_ACCOUNT_ID, AZURE_VIDEO_INDEXER_LOCATION, "
                    "and AZURE_VIDEO_INDEXER_API_KEY."
                ),
            )

        file_bytes = await file.read()
        print(f"[video] bytes_read={len(file_bytes)}", flush=True)
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video file is empty",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        indexer_result = azure_video_indexer_service.analyze_video_file(
            file_path=temp_path,
            filename=filename,
        )

        transcript = indexer_result["transcript"]

        gpt_sentiment = None
        if transcript:
            try:
                print("[video] GPT sentiment analysis starting", flush=True)
                gpt_result = llm_service.analyze_sentiment_with_gpt(transcript, video_data=indexer_result)
                gpt_sentiment = gpt_result
                print(f"[video] GPT sentiment={gpt_result.get('sentiment')} confidence={gpt_result.get('confidence')}", flush=True)
            except Exception as e:
                print(f"[video] GPT sentiment failed (non-fatal): {str(e)}", flush=True)

        pitch_scores = None
        try:
            print("[video] pitch scoring starting", flush=True)
            pitch_scores = run_b_layer(indexer_result["raw_index_data"])
            print("[video] pitch scoring done", flush=True)
        except Exception as e:
            print(f"[video] pitch scoring failed (non-fatal): {str(e)}", flush=True)

        persisted = False
        persistence_error = None
        row_for_response: Optional[str] = None
        transcript_summary = (transcript or "").strip()[:16000] if transcript else None
        rid = (supabase_row_id or "").strip() or None
        if rid:
            from app.services.supabase_service import persist_upload_analysis

            if not pitch_scores:
                persistence_error = "Cannot persist without pitch scores (scoring failed)."
            else:
                ok, persistence_error = persist_upload_analysis(
                    rid,
                    indexer_result["raw_index_data"],
                    pitch_scores,
                    _gpt_as_dict(gpt_sentiment),
                    video_transcript_summary=transcript_summary,
                )
                persisted = ok
                if ok:
                    row_for_response = rid
                    print(f"[video] persisted to Supabase row id={rid}", flush=True)

        return VideoSentimentResponse(
            transcript=transcript,
            overall_sentiment="neutral",
            confidence_scores={},
            sentences=[],
            video_sentiments=indexer_result["video_sentiments"],
            emotions=indexer_result["emotions"],
            gpt_sentiment=gpt_sentiment,
            insights=indexer_result["insights"],
            pitch_scores=pitch_scores,
            raw_index_data=indexer_result["raw_index_data"],
            video_id=indexer_result["video_id"],
            response_time_seconds=indexer_result["response_time_seconds"],
            persisted_to_supabase=persisted,
            supabase_row_id=row_for_response,
            persistence_error=persistence_error if rid and not persisted else None,
            video_transcript_summary=transcript_summary,
        )

    except HTTPException:
        raise
    except ValueError as e:
        print(f"[video] value error: {str(e)}", flush=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print(f"[video] unexpected error: {str(e)}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video sentiment analysis failed: {str(e)}",
        )
    finally:
        print("[video] cleanup temp files", flush=True)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/video/stream")
async def analyze_video_stream(
    file: UploadFile = File(...),
    supabase_row_id: Optional[str] = Form(
        default=None,
        description=(
            "Optional sentiment_outputs.id: on success, raw VI JSON and scores are "
            "saved to this row; final SSE event includes persisted_to_supabase."
        ),
    ),
):
    """SSE endpoint — streams real Azure Video Indexer progress to the frontend."""

    filename = file.filename or "video"
    extension = os.path.splitext(filename.lower())[1]

    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported extension: {extension}. Supported: .mp4, .mov, .avi, .wmv, .webm",
        )

    if not azure_video_indexer_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Azure Video Indexer is not configured.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as f:
        f.write(file_bytes)
        temp_path = f.name

    row_to_persist = (supabase_row_id or "").strip() or None

    async def event_stream():
        loop = asyncio.get_running_loop()
        svc = azure_video_indexer_service
        start = time.monotonic()

        def sse(obj: dict) -> str:
            return f"data: {_json.dumps(obj)}\n\n"

        try:
            # Step 1 — auth
            yield sse({"stage": "auth", "progress": 3, "message": "Authenticating..."})
            access_token = await loop.run_in_executor(None, svc._get_access_token)

            # Step 2 — upload
            yield sse({"stage": "uploading", "progress": 8, "message": "Uploading video to Azure..."})
            video_id = await loop.run_in_executor(
                None, svc._upload_video_file, temp_path, filename, access_token
            )
            yield sse({"stage": "indexing", "progress": 12, "message": f"Uploaded (ID: {video_id}). Indexing started..."})

            # Step 3 — poll with real VI progress
            poll_url = f"{svc.BASE_URL}/{svc.location}/Accounts/{svc.account_id}/Videos/{video_id}/Index"
            import httpx as _httpx
            deadline = time.monotonic() + 600
            index_data = None

            while time.monotonic() < deadline:
                def do_poll():
                    with _httpx.Client(timeout=_httpx.Timeout(connect=10, read=30, write=5, pool=5)) as c:
                        return c.get(poll_url, params={"accessToken": access_token})

                resp = await loop.run_in_executor(None, do_poll)
                data = resp.json()
                state = data.get("state", "")
                progress_str = data.get("videos", [{}])[0].get("processingProgress", "0%")
                vi_pct = int(progress_str.replace("%", "").strip() or 0)
                # Map VI 0-100% to our 12-82% range
                mapped = 12 + int(vi_pct * 0.70)
                yield sse({"stage": "indexing", "progress": mapped, "message": f"Indexing... {vi_pct}%"})

                if state == "Processed":
                    index_data = data
                    break
                elif state == "Failed":
                    yield sse({"stage": "error", "progress": 0, "message": "Video Indexer processing failed"})
                    return

                await asyncio.sleep(10)

            if not index_data:
                yield sse({"stage": "error", "progress": 0, "message": "Processing timed out after 600s"})
                return

            # Step 4 — extract
            yield sse({"stage": "extracting", "progress": 85, "message": "Extracting transcript and insights..."})
            transcript = svc._extract_transcript(index_data)
            video_sentiments = svc._extract_sentiments(index_data)
            emotions = svc._extract_emotions(index_data)
            insights = svc._extract_insights(index_data)

            # Step 5 — score
            yield sse({"stage": "scoring", "progress": 90, "message": "Running pitch scorer..."})
            pitch = run_b_layer(index_data)
            scores = {
                "team_strength":      pitch["team_strength"]["score"],
                "technical_strength": pitch["technical_strength"]["score"],
                "innovation":         pitch["innovation"]["score"],
                "credibility":        pitch["credibility"]["score"],
                "confidence":         pitch["confidence"]["score"],
            }

            # Step 6 — GPT sentiment
            yield sse({"stage": "sentiment", "progress": 95, "message": "Analyzing sentiment with GPT..."})
            gpt_sentiment = None
            try:
                gpt_sentiment = llm_service.analyze_sentiment_with_gpt(
                    transcript,
                    video_data={"video_sentiments": video_sentiments, "emotions": emotions, "insights": insights},
                )
            except Exception as e:
                print(f"[stream] GPT sentiment failed (non-fatal): {e}", flush=True)

            total = round(time.monotonic() - start, 1)

            persisted = False
            persistence_error = None
            supabase_row_out: Optional[str] = None
            if row_to_persist:
                from app.services.supabase_service import persist_upload_analysis

                tsum = (transcript or "").strip()[:16000] if transcript else None
                ok, persistence_error = persist_upload_analysis(
                    row_to_persist,
                    index_data,
                    pitch,
                    _gpt_as_dict(gpt_sentiment),
                    video_transcript_summary=tsum,
                )
                persisted = ok
                if ok:
                    supabase_row_out = row_to_persist
                    print(f"[stream] persisted to Supabase row id={row_to_persist}", flush=True)

            # Final result
            yield sse({
                "stage": "done",
                "progress": 100,
                "message": f"Complete in {total}s",
                "result": {
                    "video_id": video_id,
                    "transcript": transcript,
                    "video_sentiments": video_sentiments,
                    "emotions": emotions,
                    "insights": insights,
                    "pitch_scores": pitch,
                    "scores": scores,
                    "gpt_sentiment": gpt_sentiment,
                    "response_time_seconds": total,
                    "persisted_to_supabase": persisted,
                    "supabase_row_id": supabase_row_out,
                    "persistence_error": persistence_error if row_to_persist and not persisted else None,
                },
            })

        except Exception as e:
            print(f"[stream] unexpected error: {e}", flush=True)
            yield sse({"stage": "error", "progress": 0, "message": str(e)})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/video/submit-async",
    response_model=AsyncVideoSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_video_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    owner_id: str = Form(
        ...,
        description="profiles.id (auth.users id) — owner of the new media_assets row",
    ),
    project_id: Optional[str] = Form(
        default=None,
        description="Optional projects.id for media_assets.project_id",
    ),
    media_kind: str = Form(
        default="video",
        description="media_assets.kind — must match your Postgres enum / check constraint",
    ),
):
    """
    Store the file in Supabase Storage, insert media_assets + sentiment_outputs, return
    IDs and media_url immediately (HTTP 202). Video Indexer + scoring run in a background task.
    Poll GET /api/v1/sentiment/status/{sentiment_output_id} until job_status is done or failed.
    """
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase is not configured (SUPABASE_URL, SUPABASE_KEY).",
        )
    try:
        uuid_stdlib.UUID(owner_id.strip())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="owner_id must be a valid UUID")
    pid = (project_id or "").strip() or None
    if pid:
        try:
            uuid_stdlib.UUID(pid)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="project_id must be a valid UUID")

    filename = file.filename or "uploaded_video.mp4"
    extension = os.path.splitext(filename.lower())[1]
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported extension: {extension}. "
                f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
            ),
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    from app.services.supabase_service import upload_video_create_job

    try:
        created = upload_video_create_job(
            owner_id=owner_id.strip(),
            project_id=pid,
            media_kind=(media_kind or "video").strip() or "video",
            file_bytes=file_bytes,
            filename=filename,
            mime_type=file.content_type,
        )
    except Exception as e:
        print(f"[submit-async] persist failed: {e}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not store media in Supabase: {str(e)}",
        ) from e

    sid = created["sentiment_output_id"]
    background_tasks.add_task(background_index_sentiment_output, sid)
    return AsyncVideoSubmitResponse(
        sentiment_output_id=sid,
        media_asset_id=created["media_asset_id"],
        media_url=created["media_url"],
        storage_bucket=created["storage_bucket"],
        storage_path=created["storage_path"],
        job_status="queued",
        status_poll_path=f"/api/v1/sentiment/status/{sid}",
    )


@router.post("/process-queue")
async def process_queue():
    """Fetch unprocessed rows from Supabase, run Video Indexer + scorer, write results back."""
    from app.services.supabase_service import (
        fetch_unprocessed,
        store_raw_output,
        update_azure_video_indexer_id,
        update_processed,
    )

    print("[queue] checking for unprocessed videos", flush=True)
    rows = fetch_unprocessed()
    if not rows:
        print("[queue] no unprocessed rows found", flush=True)
        return {"processed": 0, "message": "No unprocessed videos found"}

    print(f"[queue] found {len(rows)} unprocessed row(s)", flush=True)
    results = []

    for row in rows:
        row_id = row["id"]
        media_url = row["media_url"]
        asset_id = row.get("media_asset_id", row_id)
        print(f"[queue] processing row id={row_id} media_url={media_url}", flush=True)

        try:
            indexer_result = azure_video_indexer_service.analyze_video_url(
                video_url=media_url,
                name=f"asset_{asset_id}",
            )
            vi_id = indexer_result.get("video_id")
            if vi_id:
                update_azure_video_indexer_id(row_id, str(vi_id))

            store_raw_output(row_id, indexer_result["raw_index_data"])

            pitch = run_b_layer(indexer_result["raw_index_data"])
            scores = {
                "team_strength":      pitch["team_strength"]["score"],
                "technical_strength": pitch["technical_strength"]["score"],
                "innovation":         pitch["innovation"]["score"],
                "credibility":        pitch["credibility"]["score"],
                "confidence":         pitch["confidence"]["score"],
            }
            print(f"[queue] scores={scores}", flush=True)

            gpt_sentiment = {}
            try:
                gpt_sentiment = llm_service.analyze_sentiment_with_gpt(
                    indexer_result["transcript"], video_data=indexer_result
                )
            except Exception as e:
                print(f"[queue] GPT sentiment failed (non-fatal): {str(e)}", flush=True)

            transcript = (indexer_result.get("transcript") or "").strip()
            summary = transcript[:16000] if transcript else None
            update_processed(
                row_id=row_id,
                scores=scores,
                sentiment_analysis_score=gpt_sentiment,
                video_transcript_summary=summary,
            )
            print(f"[queue] row id={row_id} updated in Supabase", flush=True)
            results.append({"id": row_id, "status": "ok", "scores": scores})

        except Exception as e:
            print(f"[queue] ERROR row id={row_id}: {str(e)}", flush=True)
            results.append({"id": row_id, "status": "error", "detail": str(e)})

    return {"processed": len(rows), "results": results}


@router.post("/callback")
async def vi_callback(id: str, state: str):
    """Azure Video Indexer POSTs here when processing completes or fails."""
    from app.services.supabase_service import store_callback_result, store_callback_error

    print(f"[callback] received id={id} state={state}", flush=True)

    if state == "Failed":
        store_callback_error(id, "Video Indexer processing failed")
        return {"ok": True}

    if state != "Processed":
        print(f"[callback] ignoring state={state}", flush=True)
        return {"ok": True}

    try:
        index_data = azure_video_indexer_service.fetch_index_data(id)
        transcript = azure_video_indexer_service._extract_transcript(index_data)
        video_sentiments = azure_video_indexer_service._extract_sentiments(index_data)
        emotions = azure_video_indexer_service._extract_emotions(index_data)
        insights = azure_video_indexer_service._extract_insights(index_data)

        pitch = run_b_layer(index_data)
        scores = {
            "team_strength":      pitch["team_strength"]["score"],
            "technical_strength": pitch["technical_strength"]["score"],
            "innovation":         pitch["innovation"]["score"],
            "credibility":        pitch["credibility"]["score"],
            "confidence":         pitch["confidence"]["score"],
        }
        print(f"[callback] scores={scores}", flush=True)

        gpt_sentiment = {}
        try:
            gpt_sentiment = llm_service.analyze_sentiment_with_gpt(
                transcript,
                video_data={"video_sentiments": video_sentiments, "emotions": emotions, "insights": insights},
            )
        except Exception as e:
            print(f"[callback] GPT failed (non-fatal): {e}", flush=True)

        tsum = (transcript or "").strip()[:16000] if transcript else None
        store_callback_result(id, index_data, scores, gpt_sentiment, video_transcript_summary=tsum)
        print(f"[callback] done azure_video_indexer_id={id}", flush=True)

    except Exception as e:
        print(f"[callback] ERROR azure_video_indexer_id={id}: {e}", flush=True)
        store_callback_error(id, str(e))

    return {"ok": True}


@router.get("/status/{row_id}")
async def get_status(row_id: str):
    """Poll this to check processing status for a queue row."""
    from app.services.supabase_service import get_row_status
    row = get_row_status(row_id)
    if not row:
        raise HTTPException(status_code=404, detail="Row not found")
    return row
