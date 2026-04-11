from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
import os
import tempfile
import shutil
import subprocess
import time

from app.models import SentimentRequest, SentimentResponse, AudioSentimentResponse, VideoSentimentResponse
from app.services.azure_language_service import AzureLanguageService
from app.services.document_intelligence_service import DocumentIntelligenceService
from app.services.azure_speech_service import AzureSpeechService
from app.services.azure_video_indexer_service import AzureVideoIndexerService
from app.services.llm_service import LLMService
from app.services.scorer import run_b_layer


router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])
azure_language_service = AzureLanguageService()
document_intelligence_service = DocumentIntelligenceService()
azure_speech_service = AzureSpeechService()
azure_video_indexer_service = AzureVideoIndexerService()
llm_service = LLMService()


SUPPORTED_DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain",
}

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".ogg"}

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".wmv"}


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert input audio to WAV PCM 16kHz mono using ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg is required for mp3/mp4 conversion but is not installed")

    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path,
    ]

    try:
        start = time.monotonic()
        print(f"[audio] ffmpeg start input={input_path} output={output_path}", flush=True)
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        elapsed = time.monotonic() - start
        print(f"[audio] ffmpeg completed in {elapsed:.2f}s", flush=True)
    except subprocess.TimeoutExpired as e:
        raise ValueError("Audio conversion timed out after 120s") from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "unknown ffmpeg error").strip()
        raise ValueError(f"Audio conversion to WAV failed: {err}") from e


@router.post("/text", response_model=SentimentResponse)
async def analyze_text_sentiment(request: SentimentRequest):
    """Analyze sentiment for plain text using Azure AI Language."""
    try:
        result = azure_language_service.analyze_sentiment(
            text=request.text,
            language=request.language,
            include_opinion_mining=request.include_opinion_mining,
        )
        return SentimentResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


@router.post("/simple", response_model=SentimentResponse)
async def analyze_text_sentiment_simple(text: str, language: str = "en"):
    """Simple query-param endpoint for quick sentiment tests."""
    try:
        result = azure_language_service.analyze_sentiment(
            text=text,
            language=language,
            include_opinion_mining=False,
        )
        return SentimentResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}",
        )


@router.post("/document", response_model=SentimentResponse)
async def analyze_document_sentiment(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    """Analyze sentiment for uploaded document (PDF/DOCX/DOC/TXT)."""
    try:
        filename = file.filename or "uploaded_file"
        extension = os.path.splitext(filename.lower())[1]
        content_type = file.content_type or "application/octet-stream"

        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported file extension: {extension or 'unknown'}. "
                    "Supported formats are .pdf, .docx, .txt"
                ),
            )

        if content_type not in SUPPORTED_DOCUMENT_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported MIME type: {content_type}. "
                    "Supported MIME types are application/pdf, "
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document, text/plain"
                ),
            )

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )

        if content_type == "text/plain":
            extracted_text = file_bytes.decode("utf-8", errors="ignore").strip()
            if not extracted_text:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No readable text found in uploaded text file",
                )
        else:
            extracted_text = document_intelligence_service.extract_text(
                file_bytes=file_bytes,
            )

        result = azure_language_service.analyze_sentiment(
            text=extracted_text,
            language=language,
            include_opinion_mining=False,
        )
        return SentimentResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Document sentiment analysis failed for filename={filename}, "
                f"mime={content_type}, size_bytes={len(file_bytes) if 'file_bytes' in locals() else 0}. "
                f"Error: {str(e)}"
            ),
        )


@router.post("/audio", response_model=AudioSentimentResponse)
async def analyze_audio_sentiment(
    file: UploadFile = File(...),
    speech_language: str = Form("en-US"),
    sentiment_language: str = Form("en"),
):
    """Transcribe uploaded audio using Azure Speech, then analyze sentiment using Azure Language."""
    temp_path = None
    converted_wav_path = None
    try:
        print("[audio] /api/v1/sentiment/audio request received", flush=True)
        filename = file.filename or "uploaded_audio"
        extension = os.path.splitext(filename.lower())[1]
        print(
            f"[audio] incoming file name={filename}, ext={extension}, content_type={file.content_type}",
            flush=True,
        )
        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported audio extension: {extension or 'unknown'}. "
                    "Supported formats are .wav, .mp3, .mp4, .m4a, .ogg"
                ),
            )

        file_bytes = await file.read()
        print(f"[audio] bytes_read={len(file_bytes)}", flush=True)
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        transcription_input_path = temp_path
        if extension != ".wav":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                converted_wav_path = wav_file.name
            print(f"[audio] converting to wav via ffmpeg: {temp_path} -> {converted_wav_path}", flush=True)
            _convert_to_wav(temp_path, converted_wav_path)
            transcription_input_path = converted_wav_path

        print(f"[audio] transcription starting, speech_language={speech_language}", flush=True)
        transcript = azure_speech_service.transcribe_file(
            file_path=transcription_input_path,
            language=speech_language,
        )
        print(f"[audio] transcription success, transcript_len={len(transcript)}", flush=True)

        print(f"[audio] sentiment analysis starting, sentiment_language={sentiment_language}", flush=True)
        sentiment = azure_language_service.analyze_sentiment(
            text=transcript,
            language=sentiment_language,
            include_opinion_mining=False,
        )
        print(f"[audio] sentiment analysis success, sentiment={sentiment.get('sentiment')}", flush=True)

        return AudioSentimentResponse(
            transcript=transcript,
            sentiment=sentiment["sentiment"],
            confidence_scores=sentiment["confidence_scores"],
            sentences=sentiment.get("sentences", []),
        )

    except HTTPException:
        raise
    except ValueError as e:
        print(f"[audio] value error: {str(e)}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        print(f"[audio] unexpected error: {str(e)}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio sentiment analysis failed: {str(e)}",
        )
    finally:
        print("[audio] cleanup temp files", flush=True)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if converted_wav_path and os.path.exists(converted_wav_path):
            os.remove(converted_wav_path)


@router.post("/video", response_model=VideoSentimentResponse)
async def analyze_video_sentiment(
    file: UploadFile = File(...),
    sentiment_language: str = Form("en"),
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
                    "Supported formats are .mp4, .mov, .avi, .wmv"
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
        analysis_text = transcript if transcript else "No speech detected"

        print(f"[video] Azure Language sentiment starting, language={sentiment_language}", flush=True)
        sentiment = azure_language_service.analyze_sentiment(
            text=analysis_text,
            language=sentiment_language,
            include_opinion_mining=False,
        )
        print(f"[video] Azure sentiment={sentiment.get('sentiment')}", flush=True)

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
            print(f"[video] pitch scoring done", flush=True)
        except Exception as e:
            print(f"[video] pitch scoring failed (non-fatal): {str(e)}", flush=True)

        return VideoSentimentResponse(
            transcript=transcript,
            overall_sentiment=sentiment["sentiment"],
            confidence_scores=sentiment["confidence_scores"],
            sentences=sentiment.get("sentences", []),
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


@router.post("/process-queue")
async def process_queue():
    """Fetch unprocessed rows from Supabase, run Video Indexer + scorer, write results back."""
    from app.services.supabase_service import fetch_unprocessed, update_processed

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
            # Submit URL directly to Video Indexer
            indexer_result = azure_video_indexer_service.analyze_video_url(
                video_url=media_url,
                name=f"asset_{asset_id}",
            )

            # Flat scores dict (C section)
            pitch = run_b_layer(indexer_result["raw_index_data"])
            scores = {
                "team_strength":      pitch["team_strength"]["score"],
                "technical_strength": pitch["technical_strength"]["score"],
                "innovation":         pitch["innovation"]["score"],
                "credibility":        pitch["credibility"]["score"],
                "confidence":         pitch["confidence"]["score"],
            }
            print(f"[queue] scores={scores}", flush=True)

            # GPT sentiment (non-fatal)
            gpt_sentiment = {}
            try:
                gpt_sentiment = llm_service.analyze_sentiment_with_gpt(
                    indexer_result["transcript"], video_data=indexer_result
                )
            except Exception as e:
                print(f"[queue] GPT sentiment failed (non-fatal): {str(e)}", flush=True)

            # Write back to Supabase
            update_processed(
                row_id=row_id,
                video_analysis_output=indexer_result["raw_index_data"],
                scores=scores,
                sentiment_analysis_score=gpt_sentiment,
            )
            print(f"[queue] row id={row_id} updated in Supabase", flush=True)
            results.append({"id": row_id, "status": "ok", "scores": scores})

        except Exception as e:
            print(f"[queue] ERROR row id={row_id}: {str(e)}", flush=True)
            results.append({"id": row_id, "status": "error", "detail": str(e)})

    return {"processed": len(rows), "results": results}
