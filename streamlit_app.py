"""
Streamlit UI for the Video Intelligence demo (replaces demo_chat.html).
Run: streamlit run streamlit_app.py
Requires the FastAPI backend (e.g. uvicorn app.main:app --port 8000).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

import httpx
import streamlit as st

DIMS = ["team_strength", "technical_strength", "innovation", "credibility", "confidence"]
DIM_LABELS = {
    "team_strength": "Team Strength",
    "technical_strength": "Technical Strength",
    "innovation": "Innovation",
    "credibility": "Credibility",
    "confidence": "Confidence",
}


def _base(base_url: str) -> str:
    return base_url.rstrip("/")


def format_pitch_summary(final_result: dict[str, Any]) -> str:
    ps = final_result.get("pitch_scores")
    vid = final_result.get("video_id", "")
    rt = final_result.get("response_time_seconds", "")
    lines = ["=== PITCH SCORES ===", f"Video: {vid}  |  Response Time: {rt}s"]
    if not ps:
        lines.append("  (not available)")
    else:
        lines.append("")
        for d in DIMS:
            dim = ps.get(d) or {}
            score = dim.get("score")
            ev = dim.get("evidence_strength", "")
            if score is not None:
                lines.append(f"{DIM_LABELS[d]:20}  {float(score):.1f} / 10  [{ev}]")
            else:
                lines.append(f"{DIM_LABELS[d]:20}  (n/a)")
        lines.append("")
        for d in DIMS:
            dim = ps.get(d) or {}
            reasoning = dim.get("reasoning") or []
            lines.append(f"  {DIM_LABELS[d]} reasoning:")
            for r in reasoning:
                lines.append(f"    - {r}")
    if final_result.get("persisted_to_supabase"):
        lines.extend(["", f"Supabase: saved to row id={final_result.get('supabase_row_id')}"])
    elif final_result.get("persistence_error"):
        lines.extend(["", f"Supabase save failed: {final_result['persistence_error']}"])
    return "\n".join(lines)


def format_queue_results(data: dict[str, Any]) -> str:
    processed = data.get("processed", 0)
    if processed == 0:
        return data.get("message") or "No unprocessed videos found in queue."
    lines = [f"Processed: {processed} video(s)", ""]
    for r in data.get("results") or []:
        rid = r.get("id")
        status = r.get("status")
        lines.append(f"ID: {rid}  →  {status}")
        if status == "ok":
            s = r.get("scores") or {}
            for k in DIMS:
                v = s.get(k)
                label = DIM_LABELS[k]
                lines.append(f"  {label:22}  {v} / 10" if v is not None else f"  {label:22}  (n/a)")
        else:
            lines.append(f"  Error: {r.get('detail', '')}")
        lines.append("")
    return "\n".join(lines)


def stream_video_analysis(
    base_url: str,
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str],
    on_event: Callable[[dict[str, Any]], None],
    supabase_row_id: Optional[str] = None,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """POST /video/stream, parse SSE. on_event receives each parsed SSE payload."""
    url = f"{_base(base_url)}/api/v1/sentiment/video/stream"
    mime = content_type or "application/octet-stream"
    files = {"file": (filename, file_bytes, mime)}
    form: dict[str, str] = {}
    if supabase_row_id and supabase_row_id.strip():
        form["supabase_row_id"] = supabase_row_id.strip()
    final_result: Optional[dict[str, Any]] = None
    timeout = httpx.Timeout(600.0, connect=30.0, read=600.0, write=600.0)

    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, files=files, data=form or None) as r:
            if r.status_code != 200:
                body = r.read().decode("utf-8", errors="replace")
                try:
                    err = json.loads(body)
                    detail = err.get("detail", body)
                    if isinstance(detail, list):
                        detail = json.dumps(detail)
                except json.JSONDecodeError:
                    detail = body or f"HTTP {r.status_code}"
                return None, str(detail)

            buffer = ""
            for chunk in r.iter_text():
                buffer += chunk
                parts = buffer.split("\n")
                buffer = parts.pop() if parts else ""
                for line in parts:
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    on_event(event)
                    stage = event.get("stage")
                    if stage == "error":
                        return None, str(event.get("message") or "Unknown error")
                    if stage == "done":
                        final_result = event.get("result")

            line = buffer.strip()
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    on_event(event)
                    if event.get("stage") == "done":
                        final_result = event.get("result")
                except json.JSONDecodeError:
                    pass

    if not final_result:
        return None, "No result received from stream"
    return final_result, None


def post_process_queue(base_url: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    url = f"{_base(base_url)}/api/v1/sentiment/process-queue"
    try:
        r = httpx.post(url, timeout=httpx.Timeout(600.0, connect=30.0, read=600.0))
    except httpx.RequestError as e:
        return None, str(e)
    try:
        data = r.json()
    except json.JSONDecodeError:
        return None, r.text or f"HTTP {r.status_code}"
    if r.status_code != 200:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        if isinstance(detail, list):
            detail = json.dumps(detail)
        return None, str(detail)
    return data, None


def main() -> None:
    st.set_page_config(page_title="Video Intelligence Demo", layout="wide")
    st.title("Video Intelligence Demo")
    st.caption("Analyze pitch videos using Azure Video Indexer and AI scoring.")

    base_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")

    st.subheader("Video Analysis (Video Indexer → Pitch Scores)")
    uploaded = st.file_uploader(
        "Upload video (.mp4, .mov, .avi, .wmv, .webm)",
        type=["mp4", "mov", "avi", "wmv", "webm"],
    )
    supabase_row_id = st.text_input(
        "Optional Supabase row id (sentiment_outputs.id — saves VI output + scores after analysis)",
        value="",
        placeholder="Leave empty to skip database save",
    )

    st.session_state.setdefault("last_result_text", "(waiting for request)")
    st.session_state.setdefault("last_status", "Ready.")

    progress = st.progress(0, text="Idle")
    status_line = st.empty()
    status_line.info(st.session_state["last_status"])

    c1, c2 = st.columns(2)
    with c1:
        run_analyze = st.button("Analyze Video", disabled=uploaded is None)
    with c2:
        run_queue = st.button("Process Queue (Supabase)")

    if run_analyze and uploaded is not None:
        data = uploaded.getvalue()
        fname = uploaded.name or "uploaded_video.mp4"
        ctype = uploaded.type

        def on_event(event: dict[str, Any]) -> None:
            pct = min(100, int(event.get("progress") or 0))
            msg = str(event.get("message") or "")
            progress.progress(pct / 100.0, text=msg[:80] if msg else "…")
            status_line.info(msg or "…")

        status_line.info("Uploading / processing…")
        final_result, err = stream_video_analysis(
            base_url, data, fname, ctype, on_event, supabase_row_id=supabase_row_id or None
        )

        if err:
            progress.progress(0, text="Error")
            status_line.error(err)
            st.session_state["last_result_text"] = "(analysis failed)"
            st.session_state["last_status"] = f"Error: {err}"
        else:
            progress.progress(1.0, text="Analysis complete")
            status_line.success("Video analysis complete")
            st.session_state["last_result_text"] = format_pitch_summary(final_result or {})
            st.session_state["last_status"] = "Video analysis complete"

    if run_queue:
        def q_progress(p: float, msg: str) -> None:
            progress.progress(p, text=msg)

        q_progress(0.05, "Processing queue…")
        status_line.info("Processing queue… this may take several minutes.")
        data, err = post_process_queue(base_url)

        if err:
            q_progress(0, "Error")
            status_line.error(err)
            st.session_state["last_result_text"] = "(queue processing failed)"
            st.session_state["last_status"] = f"Error: {err}"
        else:
            q_progress(1.0, "Queue complete")
            status_line.success("Queue processing complete")
            st.session_state["last_result_text"] = format_queue_results(data or {})
            st.session_state["last_status"] = "Queue processing complete"

    st.subheader("Result")
    st.code(st.session_state["last_result_text"], language=None)

    st.subheader("Status")
    st.text(st.session_state.get("last_status", "Ready."))


if __name__ == "__main__":
    main()
