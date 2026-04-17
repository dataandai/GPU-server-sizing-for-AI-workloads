"""
Distribution sampling for Monte Carlo simulation.
Supports fixed, uniform, normal, lognormal, poisson, and empirical distributions.
"""

from __future__ import annotations
import numpy as np
from .config import DistributionSpec, DistributionType


class DistributionSampler:
    """Samples values from a specified probability distribution."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def sample(self, spec: DistributionSpec, size: int = 1) -> np.ndarray:
        """
        Sample `size` values from the specified distribution.
        Returns integer-rounded, clipped values (token counts must be non-negative integers).
        """
        match spec.type:
            case DistributionType.FIXED:
                values = np.full(size, spec.value)

            case DistributionType.UNIFORM:
                values = self.rng.uniform(spec.low, spec.high, size=size)

            case DistributionType.NORMAL:
                values = self.rng.normal(spec.mean, spec.std, size=size)

            case DistributionType.LOGNORMAL:
                values = self.rng.lognormal(spec.mu, spec.sigma, size=size)

            case DistributionType.POISSON:
                values = self.rng.poisson(spec.lam, size=size).astype(float)

            case DistributionType.EMPIRICAL:
                if not spec.values:
                    raise ValueError("Empirical distribution requires non-empty values list")
                probs = spec.probabilities if spec.probabilities else None
                if probs:
                    probs = np.array(probs, dtype=float)
                    probs = probs / probs.sum()  # Normalize
                indices = self.rng.choice(len(spec.values), size=size, p=probs)
                values = np.array([spec.values[i] for i in indices], dtype=float)

            case _:
                raise ValueError(f"Unknown distribution type: {spec.type}")

        # Apply clipping
        if spec.min_clip is not None:
            values = np.maximum(values, spec.min_clip)
        if spec.max_clip is not None:
            values = np.minimum(values, spec.max_clip)

        return values

    def sample_int(self, spec: DistributionSpec, size: int = 1) -> np.ndarray:
        """Sample and round to non-negative integers."""
        values = self.sample(spec, size)
        return np.maximum(0, np.round(values)).astype(int)

    def sample_single(self, spec: DistributionSpec) -> float:
        """Sample a single value."""
        return float(self.sample(spec, size=1)[0])

    def sample_single_int(self, spec: DistributionSpec) -> int:
        """Sample a single integer value."""
        return int(self.sample_int(spec, size=1)[0])
