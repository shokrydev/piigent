"""Distribution Matching for Data Realism.

Ensures synthetic data matches real data distributions
to prevent overfitting to synthetic artifacts.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Distribution:
    """A probability distribution over discrete values."""
    counts: Dict[str, int] = field(default_factory=dict)
    total: int = 0

    def add(self, value: str) -> None:
        """Add a value observation."""
        self.counts[value] = self.counts.get(value, 0) + 1
        self.total += 1

    def probability(self, value: str) -> float:
        """Get probability of a value."""
        if self.total == 0:
            return 0.0
        return self.counts.get(value, 0) / self.total

    def to_probabilities(self) -> Dict[str, float]:
        """Convert to probability dict."""
        if self.total == 0:
            return {}
        return {k: v / self.total for k, v in self.counts.items()}


@dataclass
class DistributionComparison:
    """Comparison between two distributions."""
    kl_divergence: float
    js_divergence: float
    chi_square: float
    missing_in_synthetic: List[str]
    novel_in_synthetic: List[str]


class DistributionMatcher:
    """Ensures synthetic data matches real data distributions.

    Computes various divergence metrics between real and synthetic
    distributions to detect when synthetic data differs significantly.

    Example:
        matcher = DistributionMatcher(real_samples)

        kl_scores = matcher.compute_divergences(synthetic_samples)
        if kl_scores["entity_type_kl"] > 0.5:
            print("Entity type distribution differs significantly!")

        adjusted_config = matcher.adjust_synpii_config(current_config)
    """

    def __init__(self, real_samples: Optional[List[Dict]] = None):
        """Initialize the distribution matcher.

        Args:
            real_samples: Real samples to compute reference distribution
        """
        self.real_distribution = self._compute_distribution(real_samples or [])

    def _compute_distribution(
        self,
        samples: List[Dict],
    ) -> Dict[str, Distribution]:
        """Compute distributions from samples.

        Args:
            samples: List of document dicts with 'entities' key

        Returns:
            Dict of distribution name to Distribution
        """
        distributions = {
            "entity_types": Distribution(),
            "entity_density": Distribution(),
            "doc_lengths": Distribution(),
            "entity_lengths": Distribution(),
        }

        for sample in samples:
            # Entity types
            entities = sample.get("entities", [])
            for entity in entities:
                entity_type = entity.get("entity_type", entity.get("type", "UNKNOWN"))
                distributions["entity_types"].add(entity_type)

                # Entity length
                length = entity.get("end", 0) - entity.get("start", 0)
                length_bucket = self._bucket_length(length)
                distributions["entity_lengths"].add(length_bucket)

            # Entity density (entities per 100 chars)
            text = sample.get("text", "")
            if text:
                density = len(entities) / (len(text) / 100 + 0.1)
                density_bucket = self._bucket_density(density)
                distributions["entity_density"].add(density_bucket)

                # Document length
                length_bucket = self._bucket_doc_length(len(text))
                distributions["doc_lengths"].add(length_bucket)

        return distributions

    def _bucket_length(self, length: int) -> str:
        """Bucket entity length."""
        if length <= 5:
            return "tiny"
        elif length <= 15:
            return "short"
        elif length <= 30:
            return "medium"
        else:
            return "long"

    def _bucket_density(self, density: float) -> str:
        """Bucket entity density."""
        if density < 1:
            return "sparse"
        elif density < 3:
            return "normal"
        elif density < 5:
            return "dense"
        else:
            return "very_dense"

    def _bucket_doc_length(self, length: int) -> str:
        """Bucket document length."""
        if length < 200:
            return "short"
        elif length < 500:
            return "medium"
        elif length < 1000:
            return "long"
        else:
            return "very_long"

    def compute_divergences(
        self,
        synthetic_samples: List[Dict],
    ) -> Dict[str, float]:
        """Compute KL divergence between real and synthetic distributions.

        Args:
            synthetic_samples: Synthetic samples to compare

        Returns:
            Dict of divergence name to KL divergence value
        """
        synthetic_dist = self._compute_distribution(synthetic_samples)

        divergences = {}
        for name in self.real_distribution:
            if name in synthetic_dist:
                kl = self._kl_divergence(
                    self.real_distribution[name],
                    synthetic_dist[name],
                )
                divergences[f"{name}_kl"] = kl

        return divergences

    def _kl_divergence(
        self,
        real: Distribution,
        synthetic: Distribution,
    ) -> float:
        """Compute KL divergence D(real || synthetic)."""
        epsilon = 1e-10
        kl = 0.0

        all_values = set(real.counts.keys()) | set(synthetic.counts.keys())

        for value in all_values:
            p = real.probability(value) + epsilon
            q = synthetic.probability(value) + epsilon
            kl += p * math.log(p / q)

        return kl

    def _js_divergence(
        self,
        real: Distribution,
        synthetic: Distribution,
    ) -> float:
        """Compute Jensen-Shannon divergence (symmetric)."""
        # Create mixture distribution
        mixture = Distribution()
        all_values = set(real.counts.keys()) | set(synthetic.counts.keys())

        for value in all_values:
            mixture.counts[value] = (real.counts.get(value, 0) + synthetic.counts.get(value, 0))
            mixture.total = real.total + synthetic.total

        # JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        kl_pm = self._kl_divergence(real, mixture)
        kl_qm = self._kl_divergence(synthetic, mixture)

        return 0.5 * (kl_pm + kl_qm)

    def compare_distributions(
        self,
        synthetic_samples: List[Dict],
    ) -> Dict[str, DistributionComparison]:
        """Compare distributions with detailed analysis.

        Args:
            synthetic_samples: Synthetic samples to compare

        Returns:
            Dict of distribution name to comparison
        """
        synthetic_dist = self._compute_distribution(synthetic_samples)
        comparisons = {}

        for name in self.real_distribution:
            real = self.real_distribution[name]
            synthetic = synthetic_dist.get(name, Distribution())

            # Find missing and novel values
            real_values = set(real.counts.keys())
            synthetic_values = set(synthetic.counts.keys())

            missing = list(real_values - synthetic_values)
            novel = list(synthetic_values - real_values)

            comparisons[name] = DistributionComparison(
                kl_divergence=self._kl_divergence(real, synthetic),
                js_divergence=self._js_divergence(real, synthetic),
                chi_square=0.0,  # Could implement chi-square test
                missing_in_synthetic=missing,
                novel_in_synthetic=novel,
            )

        return comparisons

    def adjust_synpii_config(
        self,
        current_config: Dict,
        divergences: Dict[str, float],
        threshold: float = 0.5,
    ) -> Dict:
        """Adjust SynPII config to reduce divergence.

        Args:
            current_config: Current SynPII configuration
            divergences: KL divergences from compute_divergences
            threshold: Divergence threshold to trigger adjustment

        Returns:
            Modified configuration
        """
        adjusted = dict(current_config)

        # Adjust entity type probabilities
        if divergences.get("entity_types_kl", 0) > threshold:
            real_probs = self.real_distribution["entity_types"].to_probabilities()
            if "entity_probabilities" not in adjusted:
                adjusted["entity_probabilities"] = {}
            adjusted["entity_probabilities"].update(real_probs)

        # Adjust density
        if divergences.get("entity_density_kl", 0) > threshold:
            real_density = self.real_distribution["entity_density"]
            if real_density.probability("dense") > real_density.probability("sparse"):
                adjusted["min_entities"] = adjusted.get("min_entities", 1) + 1
            else:
                adjusted["max_entities"] = adjusted.get("max_entities", 5) - 1

        # Adjust document length
        if divergences.get("doc_lengths_kl", 0) > threshold:
            real_lengths = self.real_distribution["doc_lengths"]
            most_common = max(real_lengths.counts, key=real_lengths.counts.get)
            adjusted["document_length"] = most_common

        return adjusted

    def update_real_distribution(
        self,
        samples: List[Dict],
        merge: bool = True,
    ) -> None:
        """Update the real distribution with new samples.

        Args:
            samples: New samples to incorporate
            merge: Whether to merge with existing or replace
        """
        new_dist = self._compute_distribution(samples)

        if merge:
            for name in new_dist:
                if name in self.real_distribution:
                    for value, count in new_dist[name].counts.items():
                        self.real_distribution[name].counts[value] = (
                            self.real_distribution[name].counts.get(value, 0) + count
                        )
                        self.real_distribution[name].total += count
                else:
                    self.real_distribution[name] = new_dist[name]
        else:
            self.real_distribution = new_dist

    def get_summary(self) -> Dict:
        """Get summary of real data distributions."""
        return {
            name: dist.to_probabilities()
            for name, dist in self.real_distribution.items()
        }
