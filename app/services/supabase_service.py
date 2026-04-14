import os
import re
import uuid
from typing import Any, Dict, Optional, Tuple

from supabase import create_client


def get_client():
    url = os.environ["SUPABASE_URL"].strip()
    key = os.environ["SUPABASE_KEY"].strip()
    return create_client(url, key)


def _sanitize_storage_filename(name: str) -> str:
    base = os.path.basename(name) or "video.mp4"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", base)[:180]


def _derive_job_status(row: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """queued | running | done | failed — uses sentiment_analysis_score._job_* when present."""
    if row.get("is_processed"):
        return "done", None
    sas = row.get("sentiment_analysis_score") or {}
    if isinstance(sas, dict):
        if sas.get("_job_status") == "failed":
            return "failed", sas.get("_job_error")
        if sas.get("_job_status") == "running":
            return "running", None
    return "queued", None


def set_job_meta(row_id: str, job_status: str, job_error: Optional[str] = None) -> None:
    meta: Dict[str, Any] = {"_job_status": job_status}
    if job_error is not None:
        meta["_job_error"] = job_error
    get_client().table("sentiment_outputs").update({"sentiment_analysis_score": meta}).eq("id", row_id).execute()


def fetch_sentiment_output_for_job(sentiment_output_id: str) -> dict:
    try:
        res = (
            get_client()
            .table("sentiment_outputs")
            .select("id, media_url, media_asset_id")
            .eq("id", sentiment_output_id)
            .single()
            .execute()
        )
    except Exception:
        return {}
    return res.data or {}


def upload_video_create_job(
    owner_id: str,
    project_id: Optional[str],
    media_kind: str,
    file_bytes: bytes,
    filename: str,
    mime_type: Optional[str],
) -> Dict[str, Any]:
    """
    Insert media_assets, upload bytes to Storage, insert sentiment_outputs with media_url.
    Returns media_asset_id, sentiment_output_id, media_url, storage_bucket, storage_path.
    """
    client = get_client()
    media_id = str(uuid.uuid4())
    bucket = os.getenv("SUPABASE_MEDIA_BUCKET", "crucible-media").strip()
    safe_name = _sanitize_storage_filename(filename)
    storage_path = f"pitch_uploads/{media_id}/{safe_name}"
    content_type = mime_type or "application/octet-stream"

    asset_row: Dict[str, Any] = {
        "id": media_id,
        "owner_id": owner_id,
        "kind": media_kind,
        "storage_bucket": bucket,
        "storage_path": storage_path,
        "mime_type": content_type,
    }
    if project_id:
        asset_row["project_id"] = project_id

    ins_asset = client.table("media_assets").insert(asset_row).execute()
    if not ins_asset.data:
        raise RuntimeError("media_assets insert returned no row")

    try:
        client.storage.from_(bucket).upload(
            storage_path,
            file_bytes,
            {"content-type": content_type, "upsert": "true"},
        )
    except Exception:
        client.table("media_assets").delete().eq("id", media_id).execute()
        raise

    use_signed = os.getenv("SUPABASE_MEDIA_USE_SIGNED_URL", "").lower() in ("1", "true", "yes")
    if use_signed:
        ttl = int(os.getenv("SUPABASE_SIGNED_URL_TTL_SEC", "604800"))
        signed = client.storage.from_(bucket).create_signed_url(storage_path, ttl)
        if isinstance(signed, dict):
            media_url = signed.get("signedUrl") or signed.get("signedURL")
        else:
            media_url = getattr(signed, "signed_url", None) or str(signed)
        if not media_url:
            raise RuntimeError("Could not build signed URL for uploaded video")
    else:
        media_url = client.storage.from_(bucket).get_public_url(storage_path)

    out_row = {
        "media_asset_id": media_id,
        "media_url": media_url,
        "is_processed": False,
    }
    try:
        ins_out = client.table("sentiment_outputs").insert(out_row).execute()
    except Exception:
        try:
            client.storage.from_(bucket).remove([storage_path])
        except Exception:
            pass
        client.table("media_assets").delete().eq("id", media_id).execute()
        raise

    if not ins_out.data:
        raise RuntimeError("sentiment_outputs insert returned no row")
    sentiment_output_id = ins_out.data[0]["id"]

    return {
        "media_asset_id": media_id,
        "sentiment_output_id": sentiment_output_id,
        "media_url": media_url,
        "storage_bucket": bucket,
        "storage_path": storage_path,
    }


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


def persist_upload_analysis(
    row_id: str,
    raw_index_data: dict,
    pitch: dict,
    gpt_sentiment: Optional[dict],
) -> Tuple[bool, Optional[str]]:
    """
    Store VI raw JSON and final scores for an existing sentiment_outputs row
    (same pattern as process-queue). Returns (success, error_message).
    """
    try:
        store_raw_output(row_id, raw_index_data)
        scores = {
            "team_strength": pitch["team_strength"]["score"],
            "technical_strength": pitch["technical_strength"]["score"],
            "innovation": pitch["innovation"]["score"],
            "credibility": pitch["credibility"]["score"],
            "confidence": pitch["confidence"]["score"],
        }
        update_processed(row_id, scores, gpt_sentiment or {})
        return True, None
    except Exception as e:
        return False, str(e)


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
    """Return sentiment_outputs row plus computed job_status for async upload flow."""
    try:
        res = (
            get_client()
            .table("sentiment_outputs")
            .select(
                "id, media_asset_id, media_url, is_processed, scores, "
                "sentiment_analysis_score, video_analysis_output"
            )
            .eq("id", row_id)
            .single()
            .execute()
        )
    except Exception:
        return {}
    data = res.data or {}
    if not data:
        return {}
    job_status, job_error = _derive_job_status(data)
    data["job_status"] = job_status
    if job_error:
        data["job_error"] = job_error
    return data
