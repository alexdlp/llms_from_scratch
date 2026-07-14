

PIPELINE_REGISTRY: dict[str, type] = {}


def register_pipeline(name: str):
    """Decorator to register pipeline classes dynamically."""
    def decorator(cls):
        PIPELINE_REGISTRY[name] = cls
        return cls
    return decorator

def create_pipeline(cfg):
    name = cfg.model.name.lower()
    cls = PIPELINE_REGISTRY.get(name)

    if cls is None:
        raise ValueError(
            f"❌ No pipeline registered under '{name}'. "
            f"Available: {list(PIPELINE_REGISTRY.keys())}"
        )

    return cls(cfg)

from . import pipelines
from .pipelines.base_pipeline import BasePipeline
