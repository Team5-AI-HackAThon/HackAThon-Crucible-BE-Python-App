import os
from supabase import create_client


def get_client():
    url = os.environ["SUPABASE_URL"].strip()
    key = os.environ["SUPABASE_KEY"].strip()
    return create_client(url, key)


def fetch_unprocessed() -> list[dict]:
    """Return all rows where is_processed=false and media_url is set."""
    res = (
        get_client()
        .table("sentiment_outputs")
        .select("id, media_url, media_asset_id")
        .eq("is_processed", False)
        .not_.is_("media_url", "null")
        .execute()
    )
    return res.data or []


def store_raw_output(row_id: str, video_analysis_output: dict):
    """Store raw VI JSON immediately after indexing — safety net before scoring."""
    get_client().table("sentiment_outputs").update(
        {"video_analysis_output": video_analysis_output}
    ).eq("id", row_id).execute()


def fetch_raw_output(row_id: str) -> dict:
    """Fallback: fetch raw VI JSON from Supabase if in-memory dict is unavailable."""
    res = (
        get_client()
        .table("sentiment_outputs")
        .select("video_analysis_output")
        .eq("id", row_id)
        .single()
        .execute()
    )
    return (res.data or {}).get("video_analysis_output") or {}


def update_processed(row_id: str, scores: dict, sentiment_analysis_score: dict):
    """Store scores and mark row as processed. Raw output already stored separately."""
    get_client().table("sentiment_outputs").update({
        "scores": scores,
        "sentiment_analysis_score": sentiment_analysis_score,
        "is_processed": True,
        "raw_model_version": "v1",
    }).eq("id", row_id).execute()


def submit_row(row_id: str, vi_video_id: str):
    """Mark row as submitted to VI with its video_id."""
    get_client().table("sentiment_outputs").update({
        "vi_video_id": vi_video_id,
        "processing_status": "submitted",
    }).eq("id", row_id).execute()


def store_callback_result(vi_video_id: str, raw_output: dict, scores: dict, gpt_sentiment: dict):
    """Called from callback — store results and mark done."""
    get_client().table("sentiment_outputs").update({
        "video_analysis_output": raw_output,
        "scores": scores,
        "sentiment_analysis_score": gpt_sentiment,
        "is_processed": True,
        "processing_status": "done",
        "raw_model_version": "v1",
    }).eq("vi_video_id", vi_video_id).execute()


def store_callback_error(vi_video_id: str, error: str):
    """Called from callback on failure."""
    get_client().table("sentiment_outputs").update({
        "processing_status": "error",
        "processing_error": error,
    }).eq("vi_video_id", vi_video_id).execute()


def get_row_status(row_id: str) -> dict:
    """Return current processing status for a row."""
    res = (
        get_client()
        .table("sentiment_outputs")
        .select("processing_status, processing_error, scores, is_processed")
        .eq("id", row_id)
        .single()
        .execute()
    )
    return res.data or {}
