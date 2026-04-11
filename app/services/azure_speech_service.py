import os
import threading

import azure.cognitiveservices.speech as speechsdk


class AzureSpeechService:
    def __init__(self):
        self.key = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION")
        self.endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")

    def is_configured(self) -> bool:
        return bool(self.key and (self.region or self.endpoint))

    def transcribe_file(self, file_path: str, language: str = "en-US") -> str:
        if not self.is_configured():
            raise ValueError("Azure Speech service is not configured")

        if self.endpoint:
            speech_config = speechsdk.SpeechConfig(subscription=self.key, endpoint=self.endpoint)
        else:
            speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        speech_config.speech_recognition_language = language
        audio_config = speechsdk.audio.AudioConfig(filename=file_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        all_text = []
        error_holder = []
        done_event = threading.Event()

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = (evt.result.text or "").strip()
                if text:
                    all_text.append(text)

        def on_canceled(evt):
            details = evt.result.cancellation_details
            error_details = details.error_details or ""
            if "401" in error_details or "Authentication error" in error_details:
                error_holder.append(
                    "Speech authentication failed (401). "
                    "Verify AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are from the same Azure Speech resource. "
                    f"Reason: {details.reason} - {error_details}"
                )
            elif details.reason == speechsdk.CancellationReason.Error:
                error_holder.append(
                    "Speech recognition canceled. "
                    "This often happens when audio format/header is unsupported. "
                    f"Reason: {details.reason} - {error_details}"
                )
            done_event.set()

        def on_session_stopped(_evt):
            done_event.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(on_session_stopped)

        try:
            recognizer.start_continuous_recognition()
            done_event.wait(timeout=300)
            recognizer.stop_continuous_recognition()
        except Exception as e:
            raise ValueError(
                "Speech transcription failed while reading audio file. "
                "Use WAV (PCM) format, for example 16kHz mono 16-bit. "
                f"Details: {str(e)}"
            ) from e

        if error_holder:
            raise ValueError(error_holder[0])

        transcript = " ".join(all_text).strip()
        if not transcript:
            raise ValueError(
                "No speech could be recognized from audio. "
                "Use a clip with clear spoken voice (not music/noise), increase volume, "
                "and set the correct speech language (for example: en-US, en-GB, ta-IN)."
            )
        return transcript
