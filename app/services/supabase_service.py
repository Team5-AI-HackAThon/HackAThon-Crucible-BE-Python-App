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


def update_processed(
    row_id: str,
    video_analysis_output: dict,
    scores: dict,
    sentiment_analysis_score: dict,
):
    get_client().table("sentiment_outputs").update(
        {
            "video_analysis_output": video_analysis_output,
            "scores": scores,
            "sentiment_analysis_score": sentiment_analysis_score,
            "is_processed": True,
            "raw_model_version": "v1",
        }
    ).eq("id", row_id).execute()
