"""
Model registry — name → configured LLMProvider (from config/models.yaml)
"""

import os

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_FILE = os.path.join(PROJECT_ROOT, 'config', 'models.yaml')

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except ImportError:
    pass


def load_model_config(path: str = MODELS_FILE) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def list_models(benchmark_only: bool = False, path: str = MODELS_FILE) -> list:
    models = load_model_config(path)["models"]
    return [name for name, cfg in models.items() if cfg.get("benchmark", True) or not benchmark_only]


def get_provider(name: str = None, path: str = MODELS_FILE):
    """Build the provider for a model name (or the configured default)."""
    config = load_model_config(path)
    name = name or config["default"]
    models = config["models"]
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(models)}")
    cfg = models[name]

    if cfg["provider"] == "openai_compat":
        from src.llm.openai_compat import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            name=name,
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key_env=cfg.get("api_key_env"),
            timeout=float(cfg.get("timeout", 60)),
        )
    if cfg["provider"] == "scratch":
        from src.llm.scratch import LocalScratchProvider
        return LocalScratchProvider(name=name, checkpoint=os.path.join(PROJECT_ROOT, cfg["checkpoint"]))
    if cfg["provider"] == "scripted":
        from src.llm.scripted import ScriptedControlProvider
        return ScriptedControlProvider(name=name, model=cfg.get("model", name))
    raise ValueError(f"Unknown provider type '{cfg['provider']}' for model '{name}'")
