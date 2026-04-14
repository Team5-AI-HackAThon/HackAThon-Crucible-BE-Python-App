"""
Pitch Scorer — Rule-based B layer.
Input : raw Azure Video Indexer JSON (dict)
Output: C-layer dict with 5 dimension scores
"""

import re
import statistics
from typing import Any

# ---------------------------------------------------------------------------
# B2 — Investor profile config (swappable)
# ---------------------------------------------------------------------------
INVESTOR_PROFILE = {
    "reward": ["conviction", "problem statement", "building", "traction", "pain point", "mission"],
    "penalty": ["vague", "no traction", "crowded", "no differentiation"],
}

# ---------------------------------------------------------------------------
# B1 — Structured extraction
# ---------------------------------------------------------------------------

def b1_extract(data: dict) -> dict:
    videos = data.get("videos", [{}])
    insights = videos[0].get("insights", {}) if videos else {}
    stats = insights.get("statistics", {})

    # Transcript
    segments = insights.get("transcript", [])
    transcript = " ".join(s.get("text", "") for s in segments if s.get("text", "")).strip()
    transcript_segments = [
        {"text": s.get("text", ""), "confidence": s.get("confidence", 0.0)}
        for s in segments
    ]

    # Speakers
    talk_ratios = stats.get("speakerTalkToListenRatio", {})
    word_counts = stats.get("speakerWordCount", {})
    fragment_counts = stats.get("speakerNumberOfFragments", {})
    longest_monologs = stats.get("speakerLongestMonolog", {})
    speakers = []
    for spk in insights.get("speakers", []):
        sid = spk["id"]
        speakers.append({
            "id": sid,
            "name": spk.get("name", f"Speaker #{sid}"),
            "talk_ratio": talk_ratios.get(str(sid), 0.0),
            "word_count": word_counts.get(str(sid), 0),
            "fragment_count": fragment_counts.get(str(sid), 0),
            "longest_monolog": longest_monologs.get(str(sid), 0),
        })

    # Sentiments
    sentiments = [
        {"type": s.get("sentimentType", ""), "average_score": s.get("averageScore", 0.0)}
        for s in insights.get("sentiments", [])
    ]

    # Emotions — confidence from first instance
    emotions = []
    for e in insights.get("emotions", []):
        instances = e.get("instances", [])
        confidence = instances[0].get("confidence", 0.0) if instances else 0.0
        emotions.append({"type": e.get("type", ""), "confidence": confidence})

    # Audio effects — confidence from first instance
    audio_effects = []
    for ae in insights.get("audioEffects", []):
        instances = ae.get("instances", [])
        confidence = instances[0].get("confidence", 0.0) if instances else 0.0
        audio_effects.append({"type": ae.get("type", ""), "confidence": confidence})

    # Keywords & topics
    keywords = [k.get("text", "") for k in insights.get("keywords", [])]
    topics = [t.get("name", "") for t in insights.get("topics", [])]

    return {
        "video_id": data.get("id", ""),
        "video_name": data.get("name", ""),
        "transcript": transcript,
        "transcript_segments": transcript_segments,
        "speakers": speakers,
        "sentiments": sentiments,
        "emotions": emotions,
        "audio_effects": audio_effects,
        "keywords": keywords,
        "topics": topics,
    }


# ---------------------------------------------------------------------------
# B3 — Feature computation
# ---------------------------------------------------------------------------

def _count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def b3_features(extracted: dict) -> dict:
    transcript = extracted["transcript"].lower()
    speakers = extracted["speakers"]
    sentiments = extracted["sentiments"]
    emotions = extracted["emotions"]
    audio_effects = extracted["audio_effects"]
    segments = extracted["transcript_segments"]

    # Speaker balance: how close to 50/50
    if len(speakers) >= 2:
        ratio1 = speakers[0]["talk_ratio"]
        speaker_balance = max(0.0, 1.0 - abs(ratio1 - 0.5) * 2) * 10
    elif len(speakers) == 1:
        speaker_balance = 3.0
    else:
        speaker_balance = 1.0

    # Both contribute (>20 words each)
    both_contribute = len(speakers) >= 2 and all(s["word_count"] > 20 for s in speakers)

    # Keyword hits
    tech_kws = ["engineer", "software", "hacked", "built", "stack", "prototype", "architect", "code"]
    diff_kws = ["only", "unique", "first", "unlike", "actually solves", "different", "better than"]
    commit_kws = ["decade", "mission", "years of our lives", "dedicated", "life's work"]
    assertive_kws = ["will", "solves", "proven", "works", "already", "achieved"]

    tech_hits = _count_keywords(transcript, tech_kws)
    diff_hits = _count_keywords(transcript, diff_kws)
    commit_hits = _count_keywords(transcript, commit_kws)
    assertive_hits = _count_keywords(transcript, assertive_kws)

    # Traction specificity: numbers >= 1000
    numbers = [int(n.replace(",", "")) for n in re.findall(r"\b[\d,]+\b", transcript)
               if int(n.replace(",", "")) >= 1000]
    traction_specificity = min(len(numbers) * 3, 9)

    # Sentiment peak (Positive entries)
    positive_scores = [s["average_score"] for s in sentiments if s["type"] == "Positive"]
    sentiment_peak = max(positive_scores) if positive_scores else 0.0

    # Joy confidence
    joy_confidences = [e["confidence"] for e in emotions if e["type"] == "Joy"]
    joy_confidence = max(joy_confidences) if joy_confidences else 0.0

    # Avg transcript confidence
    confidences = [s["confidence"] for s in segments if s.get("confidence") is not None]
    avg_transcript_confidence = statistics.mean(confidences) if confidences else 0.0

    # Has close silence
    has_close_silence = any(
        ae["confidence"] > 0.7 for ae in audio_effects if ae["type"] == "Silence"
    )

    return {
        "speaker_balance": speaker_balance,
        "both_contribute": both_contribute,
        "tech_hits": tech_hits,
        "diff_hits": diff_hits,
        "commit_hits": commit_hits,
        "assertive_hits": assertive_hits,
        "traction_specificity": traction_specificity,
        "sentiment_peak": sentiment_peak,
        "joy_confidence": joy_confidence,
        "avg_transcript_confidence": avg_transcript_confidence,
        "has_close_silence": has_close_silence,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 1.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def _evidence_strength(score: float) -> str:
    if score >= 8.0:
        return "high"
    if score >= 6.5:
        return "medium-high"
    if score >= 5.0:
        return "medium"
    if score >= 3.5:
        return "medium-low"
    return "low-medium"


def _dimension(raw_score: float, reasoning: list[str]) -> dict:
    score = _clamp(raw_score)
    return {
        "score": round(score, 1),
        "out_of": 10,
        "evidence_strength": _evidence_strength(score),
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# B4+B5 — Dimension scorers
# ---------------------------------------------------------------------------

def score_team_strength(extracted: dict, features: dict) -> dict:
    f = features
    speakers = extracted["speakers"]

    sb = f["speaker_balance"]
    bc = 10.0 if f["both_contribute"] else 3.0
    ch = min(f["commit_hits"] * 3, 10.0)
    collab = 5.0
    ts = f["traction_specificity"]

    raw = sb * 0.25 + bc * 0.25 + ch * 0.20 + collab * 0.15 + ts * 0.15

    reasoning = [
        f"Speaker balance score: {sb:.1f}/10"
        + (f" ({speakers[0]['talk_ratio']*100:.1f}% vs {speakers[1]['talk_ratio']*100:.1f}%)" if len(speakers) >= 2 else ""),
        f"Both founders contribute >20 words: {f['both_contribute']}",
        f"Commitment language hits: {f['commit_hits']} (decade, mission, life's work...)",
        f"Traction specificity score: {ts:.1f}/10",
    ]
    return _dimension(raw, reasoning)


def score_technical_strength(extracted: dict, features: dict) -> dict:
    f = features

    tech = min(f["tech_hits"] * 2, 10.0)
    building = 8.0 if f["tech_hits"] > 2 else 4.0
    complexity = 7.0 if f["tech_hits"] > 3 else 4.0
    ownership = 7.0 if f["both_contribute"] else 4.0

    raw = tech * 0.25 + building * 0.30 + complexity * 0.25 + ownership * 0.20

    reasoning = [
        f"Technical keyword hits: {f['tech_hits']} (engineer, software, hacked, built...)",
        f"Building evidence score: {building}/10 (threshold: >2 tech hits)",
        f"Complexity/defensibility score: {complexity}/10 (threshold: >3 tech hits)",
        f"Product ownership (both contributors): {f['both_contribute']}",
    ]
    return _dimension(raw, reasoning)


def score_innovation(extracted: dict, features: dict) -> dict:
    f = features

    diff = min(f["diff_hits"] * 2.5, 10.0)
    prob_insight = f["sentiment_peak"] * 10
    solution_dist = 7.0 if f["diff_hits"] > 1 else 4.0
    category_novelty = 5.0

    raw = diff * 0.30 + prob_insight * 0.25 + solution_dist * 0.25 + category_novelty * 0.20

    reasoning = [
        f"Differentiation keyword hits: {f['diff_hits']} (only, unique, first, unlike...)",
        f"Problem insight via sentiment peak: {f['sentiment_peak']:.2f} → {prob_insight:.1f}/10",
        f"Solution distinctiveness: {solution_dist}/10 (threshold: >1 diff hits)",
        f"Category novelty: {category_novelty}/10 (default, no external data)",
    ]
    return _dimension(raw, reasoning)


def score_credibility(extracted: dict, features: dict) -> dict:
    f = features

    ts = f["traction_specificity"]
    chain = 8.0 if ts > 3 else 5.0
    evidence = 7.0 if ts > 0 else 3.0
    delivery = f["avg_transcript_confidence"] * 10

    raw = ts * 0.35 + chain * 0.25 + evidence * 0.25 + delivery * 0.15

    reasoning = [
        f"Traction specificity: {ts:.1f}/10 (numbers ≥1000 in transcript)",
        f"Problem→solution→traction chain: {chain}/10",
        f"Evidence concreteness: {evidence}/10",
        f"Delivery trust (avg transcript confidence): {f['avg_transcript_confidence']:.2f} → {delivery:.1f}/10",
    ]
    return _dimension(raw, reasoning)


def score_confidence(extracted: dict, features: dict) -> dict:
    f = features

    assertive = min(f["assertive_hits"] * 2, 10.0)
    delivery_bal = f["speaker_balance"]
    sent_emo_peak = ((f["sentiment_peak"] + f["joy_confidence"]) / 2) * 10
    pacing = 7.0 if f["has_close_silence"] else 5.0
    clarity = f["avg_transcript_confidence"] * 10

    raw = assertive * 0.25 + delivery_bal * 0.20 + sent_emo_peak * 0.20 + pacing * 0.20 + clarity * 0.15

    reasoning = [
        f"Assertive language hits: {f['assertive_hits']} → {assertive:.1f}/10 (will, solves, proven...)",
        f"Delivery balance: {delivery_bal:.1f}/10",
        f"Sentiment+emotion peak: sentiment={f['sentiment_peak']:.2f}, joy={f['joy_confidence']:.2f} → {sent_emo_peak:.1f}/10",
        f"Pacing/close control (silence detected): {f['has_close_silence']} → {pacing}/10",
        f"Transcript clarity: {f['avg_transcript_confidence']:.2f} → {clarity:.1f}/10",
    ]
    return _dimension(raw, reasoning)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_b_layer(data: dict) -> dict:
    extracted = b1_extract(data)
    features = b3_features(extracted)

    return {
        "video_id": extracted["video_id"],
        "video_name": extracted["video_name"],
        "team_strength": score_team_strength(extracted, features),
        "technical_strength": score_technical_strength(extracted, features),
        "innovation": score_innovation(extracted, features),
        "credibility": score_credibility(extracted, features),
        "confidence": score_confidence(extracted, features),
    }
