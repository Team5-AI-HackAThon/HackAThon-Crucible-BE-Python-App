import openai
import os
import json
import re


class LLMService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

    def analyze_sentiment_with_gpt(self, transcript: str, video_data: dict = None) -> dict:
        """Use GPT to analyze sentiment using transcript + Video Indexer signals."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        vi_context = ""
        if video_data:
            sentiments = video_data.get("video_sentiments", [])
            emotions = video_data.get("emotions", [])
            insights = video_data.get("insights", {})

            if sentiments:
                vi_context += f"\nVideo Indexer Sentiments: " + ", ".join(
                    f"{s['sentiment_type']} (score={s['average_score']:.2f})" for s in sentiments
                )
            if emotions:
                vi_context += f"\nFacial Emotions Detected: " + ", ".join(
                    f"{e['emotion_type']} (confidence={e['confidence']:.2f})" for e in emotions
                )
            if insights.get("keywords"):
                vi_context += f"\nTop Keywords: {', '.join(insights['keywords'][:10])}"
            if insights.get("topics"):
                vi_context += f"\nTopics: {', '.join(insights['topics'])}"
            if insights.get("speakers"):
                vi_context += f"\nSpeakers: " + ", ".join(
                    f"{s['name']} ({s['word_count']} words, {s['talk_ratio']*100:.1f}% talk time)"
                    for s in insights["speakers"]
                )

        prompt = f"""You are analyzing a video. Use ALL available signals below — transcript, facial emotions, video sentiment scores, keywords, and speaker data — to determine the overall sentiment accurately.

Transcript:
\"\"\"{transcript}\"\"\"
{vi_context}

Important: words like "struggling", "problem", "hate" may describe challenges that were SOLVED — read the full context before judging.

Respond with a JSON object only, no extra text:
{{
  "sentiment": "<positive|neutral|negative|mixed>",
  "confidence": <0.0 to 1.0>,
  "reason": "<1-2 sentence explanation referencing the signals above>",
  "key_phrases": ["<phrase1>", "<phrase2>", "<phrase3>"]
}}"""

        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"GPT returned invalid JSON: {raw}")
