"""Model factories: classical ML registry (Table 1) and the customised CNN (Table 2)."""
from .ml_models import ML_REGISTRY, build_ml_models

__all__ = ["ML_REGISTRY", "build_ml_models"]
# The CNN lives in ``skinlesion.models.cnn`` (imported lazily so the ML pipeline does not need torch).
