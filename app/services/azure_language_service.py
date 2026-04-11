import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


class AzureLanguageService:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.key = os.getenv("AZURE_LANGUAGE_KEY")

        self.client = None
        if self.endpoint and self.key:
            self.client = TextAnalyticsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key)
            )

    def is_configured(self) -> bool:
        return self.client is not None

    def analyze_sentiment(
        self,
        text: str,
        language: str = "en",
        include_opinion_mining: bool = False
    ) -> dict:
        if not self.client:
            raise ValueError("Azure Language service is not configured")

        result = self.client.analyze_sentiment(
            documents=[text],
            language=language,
            show_opinion_mining=include_opinion_mining
        )[0]

        if result.is_error:
            raise ValueError(f"Azure Language API error: {result.error.message}")
        
        return {
            "sentiment": result.sentiment,
            "confidence_scores": {
                "positive": result.confidence_scores.positive,
                "neutral": result.confidence_scores.neutral,
                "negative": result.confidence_scores.negative,
            },
            "sentences": [
                {
                    "text": sentence.text,
                    "sentiment": sentence.sentiment,
                    "confidence_scores": {
                        "positive": sentence.confidence_scores.positive,
                        "neutral": sentence.confidence_scores.neutral,
                        "negative": sentence.confidence_scores.negative,
                    },
                }
                for sentence in result.sentences
            ],
        }