from fastapi import APIRouter, HTTPException, status, UploadFile, File, Request
from fastapi.responses import StreamingResponse
import os
import tempfile
import asyncio
import json as _json
import time

from app.models import VideoSentimentResponse
from app.services.azure_video_indexer_service import AzureVideoIndexerService
from app.services.llm_service import LLMService
from app.services.scorer import run_b_layer


router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])
azure_video_indexer_service = AzureVideoIndexerService()
llm_service = LLMService()

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".wmv", ".webm"}


@router.post("/video", response_model=VideoSentimentResponse)
async def analyze_video_sentiment(
    file: UploadFile = File(...),
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
async def analyze_video_stream(file: UploadFile = File(...)):
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


@router.post("/process-queue")
async def process_queue():
    """Fetch unprocessed rows from Supabase, run Video Indexer + scorer, write results back."""
    from app.services.supabase_service import fetch_unprocessed, store_raw_output, update_processed

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

            update_processed(row_id=row_id, scores=scores, sentiment_analysis_score=gpt_sentiment)
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

        store_callback_result(id, index_data, scores, gpt_sentiment)
        print(f"[callback] done vi_video_id={id}", flush=True)

    except Exception as e:
        print(f"[callback] ERROR vi_video_id={id}: {e}", flush=True)
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
