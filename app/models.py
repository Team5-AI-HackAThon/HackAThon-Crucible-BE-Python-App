from pydantic import BaseModel
from typing import List, Optional, Literal


class VideoSentimentAppearance(BaseModel):
    start_time: str
    end_time: str


class VideoSentimentSegment(BaseModel):
    sentiment_type: str
    average_score: float
    appearances: List[VideoSentimentAppearance]


class VideoEmotionSegment(BaseModel):
    emotion_type: str
    confidence: float
    appearances: List[VideoSentimentAppearance]


class VideoSpeaker(BaseModel):
    id: int
    name: str
    word_count: int
    talk_ratio: float


class VideoInsights(BaseModel):
    keywords: List[str]
    topics: List[str]
    speakers: List[VideoSpeaker]
    duration_seconds: float


class DimensionScore(BaseModel):
    score: float
    out_of: int = 10
    evidence_strength: str
    reasoning: List[str]


class PitchScoreResponse(BaseModel):
    video_id: str
    video_name: str
    team_strength: DimensionScore
    technical_strength: DimensionScore
    innovation: DimensionScore
    credibility: DimensionScore
    confidence: DimensionScore


class GPTSentimentResult(BaseModel):
    sentiment: str
    confidence: float
    reason: str
    key_phrases: List[str] = []


class SentenceSentiment(BaseModel):
    text: str
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    confidence_scores: dict


class MediaAssetPayload(BaseModel):
    """Client-provided media row; file must already exist at storage_bucket / storage_path."""

    id: str
    owner_id: str
    project_id: Optional[str] = None
    kind: str
    storage_bucket: str
    storage_path: str
    duration_seconds: Optional[int] = None
    mime_type: Optional[str] = None
    thumbnail_path: Optional[str] = None
    published_at: Optional[str] = None
    quiz_template_slug: Optional[str] = None
    metadata: dict = {}
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SubmitAsyncJsonRequest(BaseModel):
    media_asset: MediaAssetPayload


class JsonSubmitAsyncAccepted(BaseModel):
    """Immediate response for POST /submit-async (JSON body)."""

    sentiment_output_id: str
    media_asset_id: str
    media_url: str
    sse_events_path: str
    status_poll_path: str
    job_status: Literal["queued"] = "queued"
    message: str = (
        "Processing started. Open an SSE connection to sse_events_path for live stage updates."
    )


class AsyncVideoSubmitResponse(BaseModel):
    """Immediate response after upload to Storage + DB; Video Indexer runs in background."""

    sentiment_output_id: str
    media_asset_id: str
    media_url: str
    storage_bucket: str
    storage_path: str
    job_status: Literal["queued"] = "queued"
    status_poll_path: str


class VideoSentimentResponse(BaseModel):
    transcript: str
    overall_sentiment: Literal["positive", "neutral", "negative", "mixed"]
    confidence_scores: dict
    sentences: List[SentenceSentiment] = []
    video_sentiments: List[VideoSentimentSegment] = []
    emotions: List[VideoEmotionSegment] = []
    gpt_sentiment: Optional[GPTSentimentResult] = None
    insights: Optional[VideoInsights] = None
    pitch_scores: Optional[PitchScoreResponse] = None
    raw_index_data: Optional[dict] = None
    video_id: str
    response_time_seconds: float
    persisted_to_supabase: bool = False
    supabase_row_id: Optional[str] = None
    persistence_error: Optional[str] = None
    video_transcript_summary: Optional[str] = None
