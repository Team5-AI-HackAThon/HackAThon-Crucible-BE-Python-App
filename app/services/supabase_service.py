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

    # Private buckets: public URLs are not readable by Azure Video Indexer → URL_UNREACHABLE.
    # Default to signed URLs; set SUPABASE_MEDIA_USE_PUBLIC_URL=true only if the bucket is anon-readable.
    use_public = os.getenv("SUPABASE_MEDIA_USE_PUBLIC_URL", "").lower() in ("1", "true", "yes")
    if use_public:
        media_url = client.storage.from_(bucket).get_public_url(storage_path)
    else:
        ttl = int(os.getenv("SUPABASE_SIGNED_URL_TTL_SEC", "2592000"))  # 30 days — VI can be slow
        signed = client.storage.from_(bucket).create_signed_url(storage_path, ttl)
        if isinstance(signed, dict):
            media_url = signed.get("signedUrl") or signed.get("signedURL")
        else:
            media_url = getattr(signed, "signed_url", None) or str(signed)
        if not media_url:
            raise RuntimeError("Could not build signed URL for uploaded video")

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


def get_media_download_url(storage_bucket: str, storage_path: str) -> str:
    """
    URL Azure Video Indexer can GET. Defaults to signed URL (private buckets).
    Set SUPABASE_MEDIA_USE_PUBLIC_URL=true if the bucket allows anonymous read.
    """
    client = get_client()
    use_public = os.getenv("SUPABASE_MEDIA_USE_PUBLIC_URL", "").lower() in ("1", "true", "yes")
    if use_public:
        return client.storage.from_(storage_bucket).get_public_url(storage_path)
    ttl = int(os.getenv("SUPABASE_SIGNED_URL_TTL_SEC", "2592000"))
    signed = client.storage.from_(storage_bucket).create_signed_url(storage_path, ttl)
    if isinstance(signed, dict):
        media_url = signed.get("signedUrl") or signed.get("signedURL")
    else:
        media_url = getattr(signed, "signed_url", None) or str(signed)
    if not media_url:
        raise RuntimeError("Could not build signed URL for storage object")
    return media_url


def insert_sentiment_pending(media_asset_id: str, media_url: str) -> str:
    """Create sentiment_outputs row; returns new row id."""
    res = (
        get_client()
        .table("sentiment_outputs")
        .insert(
            {
                "media_asset_id": media_asset_id,
                "media_url": media_url,
                "is_processed": False,
            }
        )
        .execute()
    )
    if not res.data:
        raise RuntimeError("sentiment_outputs insert returned no row")
    return res.data[0]["id"]


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


def update_processed(
    row_id: str,
    scores: dict,
    sentiment_analysis_score: dict,
    video_transcript_summary: Optional[str] = None,
):
    """Store scores and mark row as processed. Raw output already stored separately."""
    payload = {
        "scores": scores,
        "sentiment_analysis_score": sentiment_analysis_score,
        "is_processed": True,
        "raw_model_version": "v1",
    }
    if video_transcript_summary is not None:
        payload["video_transcript_summary"] = video_transcript_summary[:16000]
    get_client().table("sentiment_outputs").update(payload).eq("id", row_id).execute()


def update_azure_video_indexer_id(row_id: str, vi_external_id: str) -> None:
    """Persist Azure VI video id for webhook callback matching (column must exist — see sql/add_azure_video_indexer_id.sql)."""
    if not vi_external_id:
        return
    try:
        get_client().table("sentiment_outputs").update(
            {"azure_video_indexer_id": vi_external_id}
        ).eq("id", row_id).execute()
    except Exception as e:
        print(
            f"[supabase] azure_video_indexer_id update skipped (run sql/add_azure_video_indexer_id.sql if needed): {e}",
            flush=True,
        )


def persist_upload_analysis(
    row_id: str,
    raw_index_data: dict,
    pitch: dict,
    gpt_sentiment: Optional[dict],
    video_transcript_summary: Optional[str] = None,
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
        update_processed(
            row_id, scores, gpt_sentiment or {}, video_transcript_summary=video_transcript_summary
        )
        return True, None
    except Exception as e:
        return False, str(e)


def store_callback_result(
    vi_video_id: str,
    raw_output: dict,
    scores: dict,
    gpt_sentiment: dict,
    video_transcript_summary: Optional[str] = None,
):
    """Called from VI webhook — matches sentiment_outputs.azure_video_indexer_id (see sql/add_azure_video_indexer_id.sql)."""
    payload = {
        "video_analysis_output": raw_output,
        "scores": scores,
        "sentiment_analysis_score": gpt_sentiment,
        "is_processed": True,
        "raw_model_version": "v1",
    }
    if video_transcript_summary:
        payload["video_transcript_summary"] = video_transcript_summary[:16000]
    get_client().table("sentiment_outputs").update(payload).eq("azure_video_indexer_id", vi_video_id).execute()


def store_callback_error(vi_video_id: str, error: str):
    """Mark job failed for the row matched by azure_video_indexer_id; clears GPT fields for visibility."""
    get_client().table("sentiment_outputs").update({
        "sentiment_analysis_score": {"_job_status": "failed", "_job_error": error},
        "is_processed": False,
    }).eq("azure_video_indexer_id", vi_video_id).execute()


def get_row_status(row_id: str) -> dict:
    """Return sentiment_outputs row plus computed job_status for async upload flow."""
    client = get_client()
    cols_base = (
        "id, media_asset_id, media_url, is_processed, scores, "
        "sentiment_analysis_score, video_analysis_output, "
        "video_transcript_summary, raw_model_version, created_at"
    )
    cols_extended = cols_base + ", azure_video_indexer_id"
    try:
        res = (
            client.table("sentiment_outputs")
            .select(cols_extended)
            .eq("id", row_id)
            .single()
            .execute()
        )
    except Exception:
        try:
            res = (
                client.table("sentiment_outputs")
                .select(cols_base)
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
