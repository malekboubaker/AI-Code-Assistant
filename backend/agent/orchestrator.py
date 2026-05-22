from __future__ import annotations

import logging
from time import perf_counter

from backend.agent.context_agent import ContextAgent
from backend.agent.memory_writer import MemoryWriterAgent
from backend.agent.prompt_builder import PromptBuilderAgent
from backend.agent.rag_controller import RagControllerAgent
from backend.agent.response_formatter import ResponseFormatterAgent
from backend.agent.task_router import TaskRouterAgent
from backend.agent.validation_agent import ValidationAgent
from backend.api.schemas import GenerateRequest, GenerateResponse
from backend.model.generation_config import default_generation_options
from backend.model.model_factory import create_model_provider

logger = logging.getLogger(__name__)


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

        query = "\n".join(part for part in [context.instruction, context.code] if part)
        step_start = perf_counter()
        rag_enabled = request.use_rag
        if context.task == "auto_complete" and "use_rag" not in request.model_fields_set:
            rag_enabled = False
        rag = self.rag_controller.decide(query, enabled=rag_enabled, language=context.language)
        timing_rag_ms = _elapsed_ms(step_start)
        logger.info(
            "RAG decision for /generate: use_rag=%s best_score=%s threshold=%s skip_reason=%s sources=%s",
            rag.use_rag,
            rag.best_score,
            rag.threshold,
            rag.skip_reason,
            len(rag.sources),
        )

        step_start = perf_counter()
        prompt = self.prompt_builder.build(context, rag)
        timing_prompt_ms = _elapsed_ms(step_start)

        step_start = perf_counter()
        raw_output = self.model_provider.generate(prompt, default_generation_options(context.task))
        timing_model_ms = _elapsed_ms(step_start)

        formatted = self.response_formatter.extract(raw_output, context)
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
        timing_metadata = {
            "timing_total_ms": 0,
            "timing_rag_ms": timing_rag_ms,
            "timing_model_ms": timing_model_ms,
            "timing_validation_ms": timing_validation_ms,
            "model_name": getattr(self.model_provider, "model", getattr(self.model_provider, "name", "unknown")),
            "prompt_length_chars": len(prompt),
            "generated_length_chars": len(raw_output),
        }
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
