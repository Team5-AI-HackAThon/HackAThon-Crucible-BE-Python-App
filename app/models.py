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
