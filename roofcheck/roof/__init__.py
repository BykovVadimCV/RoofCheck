"""Explicit-mask roof snow generator, alongside the existing depth-first renderer."""
from .inputs import RoofSample, load_sample
from .generator import GenerationConfig
from .pipeline import RoofResult, generate, save_result

__all__ = ['RoofSample','load_sample','GenerationConfig','RoofResult','generate','save_result']
