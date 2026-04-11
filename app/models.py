from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    provider: Optional[Literal["openai", "anthropic"]] = None  # Override default provider
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
                "provider": "openai"
            }
        }

class PerformanceMetrics(BaseModel):
    total_time_seconds: float
    first_token_time_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None
    input_tokens: int
    output_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    response: str
    model: str
    provider: str
    usage: dict
    performance: PerformanceMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str
    version: str
    providers_available: List[str]


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for sentiment analysis")
    language: str = Field(default="en", min_length=2, max_length=10)
    include_opinion_mining: bool = False


class SentenceSentiment(BaseModel):
    text: str
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    confidence_scores: dict


class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    confidence_scores: dict
    sentences: List[SentenceSentiment] = []


class AudioSentimentResponse(BaseModel):
    transcript: str
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    confidence_scores: dict
    sentences: List[SentenceSentiment] = []


class VideoSentimentAppearance(BaseModel):
    start_time: str
    end_time: str


class VideoSentimentSegment(BaseModel):
    sentiment_type: str        # "Neutral", "Positive"
    average_score: float       # from averageScore in API response
    appearances: List[VideoSentimentAppearance]


class VideoEmotionSegment(BaseModel):
    emotion_type: str          # "Joy", "Sadness", etc.
    confidence: float          # from instances[0].confidence
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
