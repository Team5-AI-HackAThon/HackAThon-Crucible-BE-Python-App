import os
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient


class DocumentIntelligenceService:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        self.client: Optional[DocumentIntelligenceClient] = None
        if self.endpoint and self.key:
            self.client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key),
            )

    def is_configured(self) -> bool:
        return self.client is not None

    def extract_text(self, file_bytes: bytes) -> str:
        if not self.client:
            raise ValueError("Azure Document Intelligence service is not configured")

        try:
            # Send raw binary bytes directly. Passing AnalyzeDocumentRequest +
            # content_type for binary files can cause UnsupportedContent errors.
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-read",
                body=file_bytes,
            )
            result = poller.result()
        except HttpResponseError as e:
            raise ValueError(f"Document Intelligence request failed: {str(e)}") from e

        lines = []
        for page in (result.pages or []):
            for line in (page.lines or []):
                if line.content:
                    lines.append(line.content)

        # DOCX may sometimes populate paragraphs/content even when page lines are empty.
        paragraphs = [
            p.content.strip()
            for p in (result.paragraphs or [])
            if getattr(p, "content", None) and p.content.strip()
        ]

        content_text = (result.content or "").strip()

        extracted = "\n".join(lines).strip()
        if not extracted and paragraphs:
            extracted = "\n".join(paragraphs).strip()
        if not extracted and content_text:
            extracted = content_text

        if not extracted:
            page_count = len(result.pages or [])
            line_count = sum(len(page.lines or []) for page in (result.pages or []))
            paragraph_count = len(result.paragraphs or [])
            content_length = len(result.content or "")
            raise ValueError(
                "No readable text found in document. "
                f"Diagnostics: pages={page_count}, lines={line_count}, "
                f"paragraphs={paragraph_count}, content_length={content_length}."
            )
        return extracted
