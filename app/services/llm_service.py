import anthropic
import openai
from typing import List, Optional, Literal
import os
import time
import json
from app.models import Message, PerformanceMetrics

class LLMService:
    def __init__(self):
        # OpenAI Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        # Anthropic Configuration
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        
        # Default provider
        self.default_provider = os.getenv("LLM_PROVIDER", "openai")
        
        # Initialize clients
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            
        if self.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        else:
            self.anthropic_client = None
    
    def get_available_providers(self) -> List[str]:
        """Return list of available providers based on API keys"""
        providers = []
        if self.openai_client:
            providers.append("openai")
        if self.anthropic_client:
            providers.append("anthropic")
        return providers
    
    def analyze_sentiment_with_gpt(self, transcript: str, video_data: dict = None) -> dict:
        """Use GPT to analyze sentiment using transcript + Video Indexer signals."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        # Build context from Video Indexer data if available
        vi_context = ""
        if video_data:
            sentiments = video_data.get("video_sentiments", [])
            emotions = video_data.get("emotions", [])
            insights = video_data.get("insights", {})

            if sentiments:
                sent_lines = ", ".join(
                    f"{s['sentiment_type']} (score={s['average_score']:.2f})"
                    for s in sentiments
                )
                vi_context += f"\nVideo Indexer Sentiments: {sent_lines}"

            if emotions:
                emo_lines = ", ".join(
                    f"{e['emotion_type']} (confidence={e['confidence']:.2f})"
                    for e in emotions
                )
                vi_context += f"\nFacial Emotions Detected: {emo_lines}"

            if insights.get("keywords"):
                vi_context += f"\nTop Keywords: {', '.join(insights['keywords'][:10])}"

            if insights.get("topics"):
                vi_context += f"\nTopics: {', '.join(insights['topics'])}"

            if insights.get("speakers"):
                spk_lines = ", ".join(
                    f"{s['name']} ({s['word_count']} words, {s['talk_ratio']*100:.1f}% talk time)"
                    for s in insights["speakers"]
                )
                vi_context += f"\nSpeakers: {spk_lines}"

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
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # fallback if GPT wraps in markdown
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"GPT returned invalid JSON: {raw}")

    async def chat_completion(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        provider: Optional[Literal["openai", "anthropic"]] = None
    ) -> dict:
        """
        Send messages to LLM and get response with performance metrics
        """
        # Determine which provider to use
        selected_provider = provider or self.default_provider
        
        # Start timing
        start_time = time.perf_counter()
        
        try:
            if selected_provider == "openai":
                result = await self._openai_completion(messages, max_tokens, temperature, start_time)
            elif selected_provider == "anthropic":
                result = await self._anthropic_completion(messages, max_tokens, temperature, start_time)
            else:
                raise ValueError(f"Unsupported provider: {selected_provider}")
            
            return result
            
        except Exception as e:
            raise Exception(f"{selected_provider.upper()} API error: {str(e)}")
    
    async def _openai_completion(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        temperature: float,
        start_time: float
    ) -> dict:
        """Handle OpenAI API calls with performance tracking"""
        
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # Make API call
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Calculate timing
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Extract response
        response_text = response.choices[0].message.content
        
        # Usage statistics
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # Calculate tokens per second
        tokens_per_second = output_tokens / total_time if total_time > 0 else 0
        
        # Build performance metrics
        performance = PerformanceMetrics(
            total_time_seconds=round(total_time, 4),
            tokens_per_second=round(tokens_per_second, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
        
        return {
            "response": response_text,
            "model": response.model,
            "provider": "openai",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "performance": performance
        }
    
    async def _anthropic_completion(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        temperature: float,
        start_time: float
    ) -> dict:
        """Handle Anthropic API calls with performance tracking"""
        
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        # Separate system messages from conversation
        system_prompt = None
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})
        
        # Prepare API parameters
        api_params = {
            "model": self.anthropic_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation_messages
        }
        
        if system_prompt:
            api_params["system"] = system_prompt
        
        # Make API call
        response = self.anthropic_client.messages.create(**api_params)
        
        # Calculate timing
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Extract response
        response_text = response.content[0].text
        
        # Usage statistics
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        
        # Calculate tokens per second
        tokens_per_second = output_tokens / total_time if total_time > 0 else 0
        
        # Build performance metrics
        performance = PerformanceMetrics(
            total_time_seconds=round(total_time, 4),
            tokens_per_second=round(tokens_per_second, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
        
        return {
            "response": response_text,
            "model": response.model,
            "provider": "anthropic",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "performance": performance
        }
