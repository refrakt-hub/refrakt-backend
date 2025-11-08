"""Conversational assistant orchestration leveraging OpenAI lightweight model."""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from config import get_settings
from services.ai_service import AIService
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.job_context_service import JobContextService
from utils.vector_store import VectorStoreManager


DEFAULT_SYSTEM_PROMPT = (
    "You are Refrakt Assistant, a helpful guide for the Refrakt ML framework. "
    "Use the provided context when available and clearly acknowledge when information is missing. "
    "Never reveal API keys, secrets, or personal data. "
    "When the user is ready to start training, set intent to 'training_request' and supply a concise "
    "training_prompt suitable for YAML generation. "
    "Otherwise set intent to 'general'. If you cannot help, set intent to 'unknown'. "
    "Always respond with JSON: "
    '{"reply": "...", "intent": "...", "confidence": 0.8, "training_prompt": null}.'
)

FALLBACK_REPLY = (
    "I am still gathering details on that. Could you clarify what you would like to know about Refrakt?"
)

JOB_ID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)


@dataclass
class AssistantResult:
    """Structured assistant output."""

    intent: str
    reply: str
    confidence: float
    conversation_id: str
    training_prompt: Optional[str] = None


class AssistantService:
    """Manage Refrakt conversational flows with retrieval grounding."""

    def __init__(
        self,
        ai_service: AIService,
        job_context_service: Optional[JobContextService] = None,
        system_prompt: Optional[str] = None,
    ):
        self._ai_service = ai_service
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._sessions: Dict[str, List[dict]] = {}
        self._max_history = 12
        self._logger = logging.getLogger(__name__)

        settings = get_settings()
        self._retrieval_enabled = settings.ASSISTANT_RETRIEVAL_ENABLED
        self._max_context_chunks = settings.ASSISTANT_MAX_CONTEXT_CHUNKS
        self._doc_service: Optional[DocumentService] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._vector_store: Optional[VectorStoreManager] = None

        if self._retrieval_enabled:
            self._doc_service = DocumentService()
            self._embedding_service = EmbeddingService()
            store_path = settings.ASSISTANT_INDEX_PATH / "static_store"
            self._vector_store = VectorStoreManager(store_path)
            self._initialize_vector_store(reindex=settings.ASSISTANT_REINDEX_ON_START)

        self._job_context_service = job_context_service

    def process_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AssistantResult:
        """Route a user message through the conversation model."""

        conv_id = conversation_id or str(uuid.uuid4())
        history = self._sessions.setdefault(conv_id, [])

        # Truncate history if needed
        if len(history) > self._max_history:
            history[:] = history[-self._max_history :]

        messages = [
            {"role": "system", "content": self._system_prompt},
            *history,
            {
                "role": "user",
                "content": self._format_user_message(message, user_id=user_id),
            },
        ]

        context_sections = self._collect_context_sections(message)
        if context_sections:
            context_message = self._build_context_message(context_sections)
            messages.insert(1, {"role": "system", "content": context_message})

        raw_response = self._ai_service.generate_conversation_turn(
            messages,
            response_format={"type": "json_object"},
        )

        parsed = self._parse_response(raw_response)

        assistant_message = {"role": "assistant", "content": parsed["reply"]}
        history.append({"role": "user", "content": message})
        history.append(assistant_message)

        return AssistantResult(
            intent=parsed["intent"],
            reply=parsed["reply"],
            training_prompt=parsed.get("training_prompt"),
            confidence=float(parsed.get("confidence", 0.0)),
            conversation_id=conv_id,
        )

    def _parse_response(self, raw: str) -> Dict[str, object]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        reply = str(data.get("reply") or "").strip()
        intent = str(data.get("intent") or "general")
        training_prompt = data.get("training_prompt") or None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if not reply:
            self._logger.debug("Assistant returned empty reply; using fallback.")
            reply = FALLBACK_REPLY

        return {
            "reply": reply,
            "intent": intent,
            "training_prompt": training_prompt,
            "confidence": confidence,
        }

    @staticmethod
    def _format_user_message(message: str, user_id: Optional[str] = None) -> str:
        if user_id:
            return f"USER_ID: {user_id}\nMESSAGE: {message}"
        return message

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _initialize_vector_store(self, reindex: bool) -> None:
        if not self._vector_store or not self._embedding_service or not self._doc_service:
            return

        self._vector_store.load_or_initialize()
        if reindex or not self._vector_store.is_ready():
            self.rebuild_static_index()

    def rebuild_static_index(self) -> bool:
        if not (self._vector_store and self._embedding_service and self._doc_service):
            return False
        try:
            chunks = self._doc_service.load_static_corpus()
            payload = [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]
            self._vector_store.rebuild(payload, self._embedding_service.embed)
            self._logger.info("Assistant static knowledge index rebuilt (%s chunks).", len(payload))
            self._retrieval_enabled = True
            return True
        except Exception as exc:
            self._logger.exception("Failed to rebuild assistant index: %s", exc)
            self._retrieval_enabled = False
            return False

    def _collect_context_sections(self, message: str) -> List[str]:
        sections: List[str] = []
        if self._retrieval_enabled:
            static_sections = self._static_context_lookup(message)
            sections.extend(static_sections)

        if self._job_context_service:
            job_ids = self._extract_job_ids(message)
            if job_ids:
                sections.extend(self._job_context_service.build_job_context(job_ids))

        return sections

    def _static_context_lookup(self, message: str) -> List[str]:
        if not (self._vector_store and self._embedding_service and self._vector_store.is_ready()):
            return []

        results = self._vector_store.search(
            message,
            self._embedding_service.embed,
            k=self._max_context_chunks,
            min_score=0.12,
        )
        context_lines: List[str] = []
        for result in results:
            source = result.metadata.get("source", "reference")
            snippet = result.text.strip()
            if not snippet:
                continue
            context_lines.append(f"[{source}] {snippet}")
        return context_lines

    @staticmethod
    def _build_context_message(sections: Sequence[str]) -> str:
        formatted = "\n\n".join(section.strip() for section in sections if section.strip())
        return f"CONTEXT:\n{formatted}"

    @staticmethod
    def _extract_job_ids(message: str) -> List[str]:
        if not message:
            return []
        return list({match.lower() for match in JOB_ID_PATTERN.findall(message)})



