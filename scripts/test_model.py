from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.model.generation_config import default_generation_options
from backend.model.model_factory import create_model_provider


if __name__ == "__main__":
    provider = create_model_provider()
    print(provider.generate("[TASK: code_gen]\nLanguage: python\nWrite a hello world function.", default_generation_options()))
