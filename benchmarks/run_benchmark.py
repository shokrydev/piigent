#!/usr/bin/env python3
"""Run PII detection benchmark on synthetic German clinical data.

Usage:
    python benchmarks/run_benchmark.py --num-docs 100 --output results/benchmark.json
    python benchmarks/run_benchmark.py --dataset benchmarks/datasets/german_clinical.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synpii import SynPII
from benchmarks.metrics import calculate_dataset_metrics
from graph.privacy_flow import run_flow

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_benchmark(
    documents: list,
    preset: str = "clinical",
    use_ministral: bool = False,
    ministral_model: str = "ministral-3:8b",
    use_gliner: bool = False,
    confidence_threshold: float = 0.7,
) -> dict:
    """Run pipeline on documents and calculate metrics.

    Args:
        documents: List of document dicts with 'text' and 'annotations'.
        preset: Recognizer preset.
        use_ministral: Enable Ministral LLM.
        use_gliner: Enable GLiNER NER.
        confidence_threshold: HITL threshold.

    Returns:
        Benchmark results dict.
    """
    results = []
    total_time = 0
    hitl_triggered = 0

    logger.info(f"Running benchmark on {len(documents)} documents...")
    logger.info(f"  Preset: {preset}")
    logger.info(f"  Ministral: {'enabled' if use_ministral else 'disabled'}")
    logger.info(f"  GLiNER: {'enabled' if use_gliner else 'disabled'}")
    logger.info(f"  Confidence threshold: {confidence_threshold}")
    logger.info("")

    for i, doc in enumerate(documents):
        start_time = time.time()

        result = run_flow(
            document=doc["text"],
            confidence_threshold=confidence_threshold,
            human_in_loop=False,  # Autonomous mode for benchmarking
            preset=preset,
            use_ministral=use_ministral,
            ministral_model=ministral_model,
            use_gliner=use_gliner,
        )

        elapsed = time.time() - start_time
        total_time += elapsed

        # Check if HITL would have been triggered
        if result.get("min_confidence", 1.0) < confidence_threshold:
            hitl_triggered += 1

        # Collect results for metrics
        detected = result.get("detected_entities", [])
        ground_truth = doc["annotations"]
        results.append((detected, ground_truth))

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(documents)} documents...")

    # Calculate metrics
    metrics = calculate_dataset_metrics(results)

    # Add operational metrics
    metrics["operational"] = {
        "total_documents": len(documents),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_doc": round(total_time / len(documents), 3),
        "docs_per_minute": round(len(documents) / (total_time / 60), 1) if total_time > 0 else 0,
        "hitl_triggered_count": hitl_triggered,
        "hitl_rate": round(hitl_triggered / len(documents), 3),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run PII detection benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to existing dataset JSON file",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=50,
        help="Number of documents to generate (if no dataset provided)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="clinical",
        choices=["clinical", "full_german", "minimal"],
        help="Recognizer preset",
    )
    parser.add_argument(
        "--use-ministral",
        action="store_true",
        help="Enable Ministral LLM recognizer",
    )
    parser.add_argument(
        "--ministral-model",
        type=str,
        default="ministral-3:8b",
        help="Ollama model name for Ministral",
    )
    parser.add_argument(
        "--use-gliner",
        action="store_true",
        help="Enable GLiNER NER recognizer",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for HITL",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Load or generate dataset
    if args.dataset:
        logger.info(f"Loading dataset from {args.dataset}...")
        with open(args.dataset, "r", encoding="utf-8") as f:
            data = json.load(f)
        documents = data["documents"]
    else:
        logger.info(f"Generating {args.num_docs} synthetic documents with SynPII...")
        synpii = SynPII(preset="clinical_de", seed=args.seed)
        documents = []
        for _ in range(args.num_docs):
            doc = synpii.generate_document()
            documents.append({
                "id": doc.id,
                "template_type": doc.template_type,
                "text": doc.text,
                "annotations": [
                    {
                        "entity_type": ann.entity_type,
                        "text": ann.text,
                        "start": ann.start,
                        "end": ann.end,
                    }
                    for ann in doc.annotations
                ],
            })

        # Save generated dataset
        dataset_path = Path("benchmarks/datasets/generated.json")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump({"documents": documents}, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved dataset to {dataset_path}")

    logger.info("")

    # Run benchmark
    results = run_benchmark(
        documents=documents,
        preset=args.preset,
        use_ministral=args.use_ministral,
        ministral_model=args.ministral_model,
        use_gliner=args.use_gliner,
        confidence_threshold=args.threshold,
    )

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info("")

    overall = results["overall"]
    logger.info("Overall Metrics:")
    logger.info(f"  Precision: {overall['precision']:.3f}")
    logger.info(f"  Recall:    {overall['recall']:.3f}")
    logger.info(f"  F1 Score:  {overall['f1']:.3f}")
    logger.info(f"  TP: {overall['true_positives']}, FP: {overall['false_positives']}, FN: {overall['false_negatives']}")

    logger.info("")
    logger.info("Per-Entity-Type Metrics:")
    for entity_type, metrics in sorted(results["per_type"].items()):
        logger.info(
            f"  {entity_type:20} | P: {metrics['precision']:.2f} | "
            f"R: {metrics['recall']:.2f} | F1: {metrics['f1']:.2f} | "
            f"Support: {metrics['support']}"
        )

    logger.info("")
    ops = results["operational"]
    logger.info("Operational Metrics:")
    logger.info(f"  Documents:       {ops['total_documents']}")
    logger.info(f"  Total time:      {ops['total_time_seconds']:.1f}s")
    logger.info(f"  Throughput:      {ops['docs_per_minute']:.1f} docs/min")
    logger.info(f"  HITL would trigger: {ops['hitl_triggered_count']} ({ops['hitl_rate']:.1%})")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("")
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
