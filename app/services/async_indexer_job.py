"""Async Video Indexer pipeline with SSE stage events (used by POST /submit-async JSON)."""

from __future__ import annotations

import asyncio
import traceback

from app.services.azure_video_indexer_service import AzureVideoIndexerService
from app.services.job_sse_broker import publish as sse_publish
from app.services.llm_service import LLMService
from app.services.scorer import run_b_layer
from app.services.supabase_service import (
    fetch_sentiment_output_for_job,
    set_job_meta,
    store_raw_output,
    update_azure_video_indexer_id,
    update_processed,
)

_vi = AzureVideoIndexerService()
_llm = LLMService()


async def run_sentiment_indexer_job_sse(sentiment_output_id: str) -> None:
    try:
        await sse_publish(
            sentiment_output_id,
            {
                "stage": "running",
                "message": "Job started; calling Azure Video Indexer",
                "progress": 8,
                "sentiment_output_id": sentiment_output_id,
            },
        )
        try:
            set_job_meta(sentiment_output_id, "running", None)
        except Exception:
            pass

        row = fetch_sentiment_output_for_job(sentiment_output_id)
        media_url = row.get("media_url")
        if not media_url:
            await sse_publish(
                sentiment_output_id,
                {"stage": "error", "message": "No media_url on sentiment_outputs row", "progress": 0},
            )
            set_job_meta(sentiment_output_id, "failed", "No media_url")
            return

        if not _vi.is_configured():
            msg = "Azure Video Indexer is not configured on the server."
            await sse_publish(sentiment_output_id, {"stage": "error", "message": msg, "progress": 0})
            set_job_meta(sentiment_output_id, "failed", msg)
            return

        asset_id = row.get("media_asset_id") or sentiment_output_id
        await sse_publish(
            sentiment_output_id,
            {
                "stage": "indexing",
                "message": "Video Indexer is processing the URL (often 1–5+ minutes)",
                "progress": 20,
            },
        )

        indexer_result = await asyncio.to_thread(
            _vi.analyze_video_url,
            media_url,
            f"asset_{asset_id}",
        )
        vi_id = indexer_result.get("video_id")
        if vi_id:
            update_azure_video_indexer_id(sentiment_output_id, str(vi_id))

        await sse_publish(
            sentiment_output_id,
            {
                "stage": "indexed",
                "message": "Indexer finished; persisting raw output and scoring",
                "progress": 65,
                "video_id": indexer_result.get("video_id"),
            },
        )

        store_raw_output(sentiment_output_id, indexer_result["raw_index_data"])

        await sse_publish(
            sentiment_output_id,
            {"stage": "scoring", "message": "Running pitch scorer", "progress": 78},
        )
        pitch = run_b_layer(indexer_result["raw_index_data"])
        scores = {
            "team_strength": pitch["team_strength"]["score"],
            "technical_strength": pitch["technical_strength"]["score"],
            "innovation": pitch["innovation"]["score"],
            "credibility": pitch["credibility"]["score"],
            "confidence": pitch["confidence"]["score"],
        }

        await sse_publish(
            sentiment_output_id,
            {"stage": "sentiment", "message": "LLM sentiment analysis", "progress": 88},
        )
        gpt_sentiment: dict = {}
        try:
            gpt_sentiment = await asyncio.to_thread(
                _llm.analyze_sentiment_with_gpt,
                indexer_result["transcript"],
                indexer_result,
            )
        except Exception as e:
            print(f"[indexer-sse] GPT sentiment failed (non-fatal): {e}", flush=True)

        transcript = (indexer_result.get("transcript") or "").strip()
        summary = transcript[:16000] if transcript else None
        update_processed(sentiment_output_id, scores, gpt_sentiment, video_transcript_summary=summary)

        await sse_publish(
            sentiment_output_id,
            {
                "stage": "done",
                "message": "Complete; sentiment_outputs updated",
                "progress": 100,
                "is_processed": True,
                "sentiment_output_id": sentiment_output_id,
                "scores": scores,
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[indexer-sse] ERROR id={sentiment_output_id}: {e}\n{tb}", flush=True)
        await sse_publish(
            sentiment_output_id,
            {"stage": "error", "message": str(e), "progress": 0},
        )
        try:
            set_job_meta(sentiment_output_id, "failed", str(e))
        except Exception:
            pass
