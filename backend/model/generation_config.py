from backend.api.schemas import TaskName
from backend.config.settings import settings
from backend.model.base import GenerationOptions
from backend.model.token_manager import token_manager
import logging

logger = logging.getLogger(__name__)

def default_generation_options(
    task: TaskName | None = None, 
    response_id: str | None = None,
    prompt: str | None = None,
) -> GenerationOptions:
    
    # 1. Use TokenManager to count prompt tokens
    prompt_tokens = token_manager.count_tokens(prompt) if prompt else 0
    
    # 2. Get context window from settings
    context_window = settings.model_context_window
    
    # 3. Calculate dynamic budget
    budget_info = token_manager.calculate_budget(prompt_tokens, context_window)
    max_tokens = budget_info["generation_budget"]
    
    logger.debug("Dynamic Token Allocation: %s", budget_info)
    
    temperature = 0.1 if task == "auto_complete" else 0.2
    
    # Extend GenerationOptions if needed, but we'll attach metadata in orchestrator
    # We'll pass max_tokens directly.
    return GenerationOptions(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        response_id=response_id,
        metadata=budget_info,
    )
