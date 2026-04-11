import os
import time

import httpx


class AzureVideoIndexerService:
    BASE_URL = "https://api.videoindexer.ai"

    def __init__(self):
        self.account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
        self.location = os.getenv("AZURE_VIDEO_INDEXER_LOCATION", "trial")
        self.api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")

    def is_configured(self) -> bool:
        configured = bool(self.account_id and self.api_key)
        print(f"[video:init] is_configured={configured} account_id={'set' if self.account_id else 'MISSING'} api_key={'set' if self.api_key else 'MISSING'} location={self.location}", flush=True)
        return configured

    def _get_access_token(self) -> str:
        url = f"{self.BASE_URL}/Auth/{self.location}/Accounts/{self.account_id}/AccessToken"
        print(f"[video:auth] requesting access token from {url}", flush=True)
        with httpx.Client(timeout=httpx.Timeout(30)) as client:
            response = client.get(
                url,
                params={"allowEdit": "true"},
                headers={
                    "Cache-Control": "no-cache",
                    "Ocp-Apim-Subscription-Key": self.api_key,
                },
            )
        print(f"[video:auth] response status={response.status_code}", flush=True)
        if response.status_code != 200:
            print(f"[video:auth] ERROR body={response.text}", flush=True)
            raise ValueError(
                f"Failed to get Video Indexer access token: {response.status_code} {response.text}"
            )
        token = response.text.strip().strip('"')
        print(f"[video:auth] token received, length={len(token)}", flush=True)
        return token

    def _upload_video_url(self, video_url: str, name: str, access_token: str) -> str:
        url = f"{self.BASE_URL}/{self.location}/Accounts/{self.account_id}/Videos"
        print(f"[video:upload_url] starting url_upload name={name} video_url={video_url}", flush=True)
        with httpx.Client(timeout=httpx.Timeout(connect=10, read=60, write=10, pool=5)) as client:
            response = client.post(
                url,
                params={
                    "name": name,
                    "accessToken": access_token,
                    "videoUrl": video_url,
                    "language": "auto",
                },
            )
        print(f"[video:upload_url] response status={response.status_code}", flush=True)
        if response.status_code not in (200, 201):
            print(f"[video:upload_url] ERROR body={response.text}", flush=True)
            raise ValueError(f"Video URL upload failed: {response.status_code} {response.text}")
        video_id = response.json().get("id")
        if not video_id:
            print(f"[video:upload_url] ERROR no video_id in response body={response.text}", flush=True)
            raise ValueError("Video Indexer did not return a video ID after URL upload")
        print(f"[video:upload_url] success video_id={video_id}", flush=True)
        return video_id

    def _upload_video_file(self, file_path: str, name: str, access_token: str) -> str:
        url = f"{self.BASE_URL}/{self.location}/Accounts/{self.account_id}/Videos"
        file_size = os.path.getsize(file_path)
        print(f"[video:upload] starting upload url={url} name={name} file_size={file_size} bytes", flush=True)
        with open(file_path, "rb") as fh:
            with httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=300, pool=5)) as client:
                response = client.post(
                    url,
                    params={
                        "name": name,
                        "accessToken": access_token,
                        "language": "auto",
                    },
                    files={"file": (name, fh, "application/octet-stream")},
                )
        print(f"[video:upload] response status={response.status_code}", flush=True)
        if response.status_code not in (200, 201):
            print(f"[video:upload] ERROR body={response.text}", flush=True)
            raise ValueError(
                f"Video upload failed: {response.status_code} {response.text}"
            )
        video_id = response.json().get("id")
        if not video_id:
            print(f"[video:upload] ERROR no video_id in response body={response.text}", flush=True)
            raise ValueError("Video Indexer did not return a video ID after upload")
        print(f"[video:upload] success video_id={video_id}", flush=True)
        return video_id

    def _poll_for_completion(self, video_id: str, access_token: str, timeout_seconds: int = 600) -> dict:
        url = f"{self.BASE_URL}/{self.location}/Accounts/{self.account_id}/Videos/{video_id}/Index"
        print(f"[video:poll] starting poll video_id={video_id} timeout={timeout_seconds}s", flush=True)
        deadline = time.monotonic() + timeout_seconds
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            elapsed = round(time.monotonic() - (deadline - timeout_seconds), 1)
            with httpx.Client(timeout=httpx.Timeout(connect=10, read=30, write=5, pool=5)) as client:
                response = client.get(url, params={"accessToken": access_token})
            print(f"[video:poll] attempt={attempt} elapsed={elapsed}s http_status={response.status_code}", flush=True)
            if response.status_code != 200:
                print(f"[video:poll] ERROR body={response.text}", flush=True)
                raise ValueError(
                    f"Failed to poll video index: {response.status_code} {response.text}"
                )
            data = response.json()
            state = data.get("state", "")
            progress = data.get("videos", [{}])[0].get("processingProgress", "N/A")
            print(f"[video:poll] state={state} progress={progress}", flush=True)
            if state == "Processed":
                print(f"[video:poll] processing complete after {elapsed}s and {attempt} attempts", flush=True)
                return data
            if state == "Failed":
                print(f"[video:poll] ERROR processing failed", flush=True)
                raise ValueError(f"Video Indexer processing failed for video_id={video_id}")
            time.sleep(10)
        raise ValueError(f"Video processing timed out after {timeout_seconds}s")

    def _extract_transcript(self, index_data: dict) -> str:
        print("[video:extract] extracting transcript", flush=True)
        try:
            segments = index_data["videos"][0]["insights"].get("transcript", [])
            transcript = " ".join(s["text"] for s in segments if s.get("text")).strip()
            print(f"[video:extract] transcript segments={len(segments)} total_chars={len(transcript)}", flush=True)
            return transcript
        except (KeyError, IndexError) as e:
            print(f"[video:extract] WARNING no transcript found: {e}", flush=True)
            return ""

    def _extract_sentiments(self, index_data: dict) -> list:
        print("[video:extract] extracting sentiments", flush=True)
        try:
            raw = index_data["videos"][0]["insights"].get("sentiments", [])
        except (KeyError, IndexError) as e:
            print(f"[video:extract] WARNING no sentiments found: {e}", flush=True)
            return []

        result = []
        for item in raw:
            appearances = [
                {"start_time": a.get("adjustedStart", ""), "end_time": a.get("adjustedEnd", "")}
                for a in item.get("instances", [])
            ]
            avg_score = item.get("averageScore", 0.0)
            result.append({
                "sentiment_type": item.get("sentimentType", ""),
                "average_score": avg_score,
                "appearances": appearances,
            })
            print(f"[video:extract] sentiment type={item.get('sentimentType')} average_score={avg_score}", flush=True)
        print(f"[video:extract] total sentiment segments={len(result)}", flush=True)
        return result

    def _extract_emotions(self, index_data: dict) -> list:
        print("[video:extract] extracting emotions", flush=True)
        try:
            raw = index_data["videos"][0]["insights"].get("emotions", [])
        except (KeyError, IndexError) as e:
            print(f"[video:extract] WARNING no emotions found: {e}", flush=True)
            return []

        result = []
        for item in raw:
            instances = item.get("instances", [])
            confidence = instances[0].get("confidence", 0.0) if instances else 0.0
            appearances = [
                {"start_time": a.get("adjustedStart", ""), "end_time": a.get("adjustedEnd", "")}
                for a in instances
            ]
            result.append({
                "emotion_type": item.get("type", ""),
                "confidence": confidence,
                "appearances": appearances,
            })
            print(f"[video:extract] emotion type={item.get('type')} confidence={confidence}", flush=True)
        print(f"[video:extract] total emotion segments={len(result)}", flush=True)
        return result

    def _extract_insights(self, index_data: dict) -> dict:
        print("[video:extract] extracting insights", flush=True)
        try:
            insights = index_data["videos"][0]["insights"]
            stats = insights.get("statistics", {})

            keywords = [
                k["text"] for k in sorted(
                    insights.get("keywords", []),
                    key=lambda x: x.get("confidence", 0),
                    reverse=True
                )[:10]
            ]

            topics = [t["name"] for t in insights.get("topics", [])]

            talk_ratios = stats.get("speakerTalkToListenRatio", {})
            word_counts = stats.get("speakerWordCount", {})
            speakers = []
            for spk in insights.get("speakers", []):
                spk_id = spk["id"]
                speakers.append({
                    "id": spk_id,
                    "name": spk.get("name", f"Speaker #{spk_id}"),
                    "word_count": word_counts.get(str(spk_id), 0),
                    "talk_ratio": talk_ratios.get(str(spk_id), 0.0),
                })

            duration_seconds = index_data.get("durationInSeconds", 0.0)
            print(f"[video:extract] keywords={len(keywords)} topics={len(topics)} speakers={len(speakers)}", flush=True)

            return {
                "keywords": keywords,
                "topics": topics,
                "speakers": speakers,
                "duration_seconds": duration_seconds,
            }
        except (KeyError, IndexError) as e:
            print(f"[video:extract] WARNING insights extraction failed: {e}", flush=True)
            return {"keywords": [], "topics": [], "speakers": [], "duration_seconds": 0.0}

    def analyze_video_file(self, file_path: str, filename: str) -> dict:
        print(f"[video] ── START analyze_video_file filename={filename} ──", flush=True)
        if not self.is_configured():
            raise ValueError("Azure Video Indexer is not configured")

        start = time.monotonic()

        print("[video] STEP 1/4 getting access token", flush=True)
        access_token = self._get_access_token()

        print("[video] STEP 2/4 uploading video file", flush=True)
        video_id = self._upload_video_file(file_path, filename, access_token)

        print("[video] STEP 3/4 polling until processed", flush=True)
        index_data = self._poll_for_completion(video_id, access_token)

        print("[video] STEP 4/4 extracting transcript, sentiments, emotions and insights", flush=True)
        transcript = self._extract_transcript(index_data)
        video_sentiments = self._extract_sentiments(index_data)
        emotions = self._extract_emotions(index_data)
        insights = self._extract_insights(index_data)

        total = round(time.monotonic() - start, 1)
        print(f"[video] ── DONE total_time={total}s transcript_len={len(transcript)} sentiments={len(video_sentiments)} emotions={len(emotions)} ──", flush=True)

        return {
            "video_id": video_id,
            "transcript": transcript,
            "video_sentiments": video_sentiments,
            "emotions": emotions,
            "insights": insights,
            "response_time_seconds": total,
            "raw_index_data": index_data,
        }

    def analyze_video_url(self, video_url: str, name: str) -> dict:
        """Same pipeline as analyze_video_file but submits a URL instead of uploading a file."""
        print(f"[video] ── START analyze_video_url name={name} ──", flush=True)
        if not self.is_configured():
            raise ValueError("Azure Video Indexer is not configured")

        start = time.monotonic()

        print("[video] STEP 1/4 getting access token", flush=True)
        access_token = self._get_access_token()

        print("[video] STEP 2/4 submitting video URL", flush=True)
        video_id = self._upload_video_url(video_url, name, access_token)

        print("[video] STEP 3/4 polling until processed", flush=True)
        index_data = self._poll_for_completion(video_id, access_token)

        print("[video] STEP 4/4 extracting transcript, sentiments, emotions and insights", flush=True)
        transcript = self._extract_transcript(index_data)
        video_sentiments = self._extract_sentiments(index_data)
        emotions = self._extract_emotions(index_data)
        insights = self._extract_insights(index_data)

        total = round(time.monotonic() - start, 1)
        print(f"[video] ── DONE total_time={total}s transcript_len={len(transcript)} ──", flush=True)

        return {
            "video_id": video_id,
            "transcript": transcript,
            "video_sentiments": video_sentiments,
            "emotions": emotions,
            "insights": insights,
            "response_time_seconds": total,
            "raw_index_data": index_data,
        }
