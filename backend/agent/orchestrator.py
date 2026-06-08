from __future__ import annotations

import logging
import re
from time import perf_counter

from backend.agent.context_agent import ContextAgent
from backend.agent.memory_writer import MemoryWriterAgent
from backend.agent.prompt_builder import PromptBuilderAgent
from backend.agent.rag_controller import RagControllerAgent
from backend.agent.response_formatter import ResponseFormatterAgent
from backend.agent.task_router import TaskRouterAgent
from backend.agent.validation_agent import ValidationAgent
from backend.api.schemas import GenerateRequest, GenerateResponse, ValidationResult
from backend.model.generation_config import default_generation_options
from backend.model.model_factory import create_model_provider

logger = logging.getLogger(__name__)
INSUFFICIENT_PROJECT_CONTEXT_MESSAGE = (
    "I do not have enough indexed context to explain this project accurately. Please run indexing first."
)
MISSING_PROJECT_MAP_MESSAGE = "This project has indexed chunks, but the project map is missing. Please run full indexing."

TECHNOLOGY_TERMS = {
    "a2a",
    "adk",
    "angular",
    "django",
    "docker",
    "fastapi",
    "flask",
    "github copilot",
    "groq",
    "mongodb",
    "mysql",
    "ollama",
    "openai",
    "postgres",
    "qdrant",
    "qdrantstore",
    "rag",
    "ragcontrolleragent",
    "react",
    "redis",
    "spring",
    "sqlite",
    "vector database",
    "vector databases",
    "vue",
    "ai code assistant",
}


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


class AgentOrchestrator:
    def __init__(self) -> None:
        self.task_router = TaskRouterAgent()
        self.context_agent = ContextAgent()
        self.rag_controller = RagControllerAgent()
        self.prompt_builder = PromptBuilderAgent()
        self.model_provider = create_model_provider()
        self.validation_agent = ValidationAgent()
        self.memory_writer = MemoryWriterAgent()
        self.response_formatter = ResponseFormatterAgent()

    def run(self, request: GenerateRequest) -> GenerateResponse:
        total_start = perf_counter()

        step_start = perf_counter()
        task = self.task_router.detect(request.instruction, request.task)
        timing_routing_ms = _elapsed_ms(step_start)

        step_start = perf_counter()
        context = self.context_agent.build(request, task)
        timing_context_ms = _elapsed_ms(step_start)

        query_parts = [context.instruction]
        if not (context.task == "project_explain" and context.explanation_scope == "project"):
            query_parts.append(context.code)
        query = "\n".join(part for part in query_parts if part)
        step_start = perf_counter()
        rag_enabled = request.use_rag
        if context.task == "auto_complete" and "use_rag" not in request.model_fields_set:
            rag_enabled = False
        rag = self.rag_controller.decide(
            query,
            enabled=rag_enabled,
            language=context.language,
            active_file=context.file_path,
            project_path=context.project_path,
            task=context.task,
            explanation_scope=context.explanation_scope,
        )
        timing_rag_ms = _elapsed_ms(step_start)
        logger.info(
            "RAG decision for /generate: use_rag=%s best_score=%s threshold=%s skip_reason=%s sources=%s",
            rag.use_rag,
            rag.best_score,
            rag.threshold,
            rag.skip_reason,
            len(rag.sources),
        )

        if (
            context.task == "project_explain"
            and context.explanation_scope == "project"
            and _insufficient_project_context(rag)
        ):
            refusal_message = _project_context_refusal_message(rag)
            validation = ValidationResult(valid=True, syntax_valid=None, validator="project_context_guard")
            metadata = self._base_metadata(
                timing_rag_ms=timing_rag_ms,
                timing_model_ms=0,
                timing_validation_ms=0,
                prompt_length_chars=0,
                generated_length_chars=len(refusal_message),
            )
            response = GenerateResponse(
                task=context.task,
                language=context.language,
                generated_code="",
                explanation=refusal_message,
                used_rag=False,
                rag_sources=[],
                validation=validation,
                metadata={
                    "stored_in_memory": False,
                    "empty_model_output": False,
                    "fallback_output": False,
                    "rag_best_score": rag.best_score,
                    "rag_threshold": rag.threshold,
                    "rag_skip_reason": rag.skip_reason or "insufficient_project_context",
                    "validator_used": validation.validator,
                    "validation_duration_ms": validation.duration_ms,
                    "validation_errors": validation.errors,
                    "validation_warnings": validation.warnings,
                    **self._rag_metadata(rag, context, rag_requested=rag_enabled),
                    **metadata,
                },
            )
            response.metadata["timing_total_ms"] = _elapsed_ms(total_start)
            return response

        step_start = perf_counter()
        prompt = self.prompt_builder.build(context, rag)
        timing_prompt_ms = _elapsed_ms(step_start)

        step_start = perf_counter()
        raw_output = self.model_provider.generate(prompt, default_generation_options(context.task))
        timing_model_ms = _elapsed_ms(step_start)

        formatted = self.response_formatter.extract(raw_output, context)
        grounding_metadata: dict[str, object] = {}
        if context.task == "project_explain" and context.explanation_scope == "project":
            grounded_output, blocked_terms = _enforce_project_explain_grounding(formatted.explanation, rag.sources)
            if blocked_terms:
                raw_output = grounded_output
                formatted = self.response_formatter.extract(raw_output, context)
                grounding_metadata["grounding_blocked_terms"] = blocked_terms
        step_start = perf_counter()
        if context.task == "project_explain":
            validation = self.validation_agent.validate_explanation(formatted.explanation)
        else:
            validation = self.validation_agent.validate(
                formatted.code,
                context.language,
                project_path=context.project_path,
                run_project_tests=request.run_tests,
            )
        timing_validation_ms = _elapsed_ms(step_start)

        step_start = perf_counter()
        stored = self.memory_writer.maybe_store(
            formatted.code,
            context.language,
            context.task,
            validation,
            accepted=request.accepted,
            file_path=context.file_path,
            is_fallback=formatted.is_fallback or formatted.is_empty,
        )
        timing_memory_ms = _elapsed_ms(step_start)
        timing_metadata = self._base_metadata(
            timing_rag_ms=timing_rag_ms,
            timing_model_ms=timing_model_ms,
            timing_validation_ms=timing_validation_ms,
            prompt_length_chars=len(prompt),
            generated_length_chars=len(raw_output),
        )
        timing_metadata.update(self._rag_metadata(rag, context, rag_requested=rag_enabled))
        timing_metadata.update(grounding_metadata)
        response = self.response_formatter.format(raw_output, context, rag, validation, stored, timing_metadata)
        response.metadata["timing_total_ms"] = _elapsed_ms(total_start)
        logger.info(
            (
                "Request timing: total_ms=%s routing_ms=%s context_ms=%s rag_ms=%s "
                "prompt_ms=%s model_ms=%s validation_ms=%s memory_ms=%s model_name=%s "
                "prompt_length_chars=%s generated_length_chars=%s"
            ),
            response.metadata["timing_total_ms"],
            timing_routing_ms,
            timing_context_ms,
            timing_rag_ms,
            timing_prompt_ms,
            timing_model_ms,
            timing_validation_ms,
            timing_memory_ms,
            timing_metadata["model_name"],
            timing_metadata["prompt_length_chars"],
            timing_metadata["generated_length_chars"],
        )
        logger.debug("Validation result for task=%s: %s", context.task, validation.model_dump())
        return response

    def _base_metadata(
        self,
        *,
        timing_rag_ms: int,
        timing_model_ms: int,
        timing_validation_ms: int,
        prompt_length_chars: int,
        generated_length_chars: int,
    ) -> dict[str, object]:
        return {
            "timing_total_ms": 0,
            "timing_rag_ms": timing_rag_ms,
            "timing_model_ms": timing_model_ms,
            "timing_validation_ms": timing_validation_ms,
            "model_name": getattr(self.model_provider, "model", getattr(self.model_provider, "name", "unknown")),
            "prompt_length_chars": prompt_length_chars,
            "generated_length_chars": generated_length_chars,
        }

    def _rag_metadata(self, rag, context, rag_requested: bool = True) -> dict[str, object]:
        source_files = _source_files_used(rag.sources)
        rag_context_available = bool(rag.use_rag and rag.sources)
        return {
            "project_id": rag.project_id,
            "project_path": rag.project_path,
            "qdrant_collection": rag.qdrant_collection,
            "rag_raw_results_count": rag.raw_results_count,
            "rag_filtered_results_count": rag.filtered_results_count,
            "rag_sources_project_ids": rag.sources_project_ids,
            "rag_project_point_count": rag.project_point_count,
            "project_map_exists": rag.project_map_exists,
            "reliable_source_count": rag.reliable_source_count,
            "explanation_scope": context.explanation_scope,
            "selected_code_primary": context.selected_code_primary,
            "active_file_path": context.active_file_path,
            "has_selection": context.has_selection,
            "chat_memory_used": bool(context.chat_history),
            "chat_history_message_count": len(context.chat_history or []),
            "rag_fallback_used": bool(rag_requested and not rag_context_available),
            "rag_context_available": rag_context_available,
            "project_map_used": _project_map_used(rag.sources),
            "source_file_count": len(source_files),
            "source_files_used": source_files,
            "source_diversity_score": _source_diversity_score(rag.sources, source_files),
        }


def _insufficient_project_context(rag) -> bool:
    if not rag.use_rag:
        return True
    return not (rag.project_map_exists or rag.reliable_source_count > 0)


def _project_context_refusal_message(rag) -> str:
    if (rag.project_point_count or 0) > 0 and not rag.project_map_exists:
        return MISSING_PROJECT_MAP_MESSAGE
    return INSUFFICIENT_PROJECT_CONTEXT_MESSAGE


def _enforce_project_explain_grounding(explanation: str, sources: list) -> tuple[str, list[str]]:
    source_blocks = []
    for source in sources:
        source_blocks.append(
            "\n".join(
                [
                    source.content,
                    str(source.file_path or ""),
                    " ".join(
                        str(value)
                        for value in source.metadata.values()
                        if isinstance(value, (str, int, float, bool))
                    ),
                ]
            )
        )
    source_text = "\n".join(source_blocks).lower()
    explanation_lower = explanation.lower()
    blocked = sorted(
        term
        for term in TECHNOLOGY_TERMS
        if re.search(rf"\b{re.escape(term)}\b", explanation_lower) and not re.search(rf"\b{re.escape(term)}\b", source_text)
    )
    if blocked:
        return INSUFFICIENT_PROJECT_CONTEXT_MESSAGE, blocked
    return explanation, []


def _project_map_used(sources: list) -> bool:
    return any(source.metadata.get("source") == "project_map" or source.chunk_type == "project_map" for source in sources)


def _source_files_used(sources: list) -> list[str]:
    files: list[str] = []
    for source in sources:
        path = (
            source.metadata.get("relative_file_path")
            or source.metadata.get("relative_path")
            or source.file_path
        )
        if path and path not in files:
            files.append(str(path))
    return files


def _source_diversity_score(sources: list, source_files: list[str]) -> float:
    if not sources:
        return 0.0
    return round(len(source_files) / len(sources), 3)
