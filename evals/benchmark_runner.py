"""
CLI Benchmark Runner for Quantitative RAG Groundedness and Retrieval Evaluation.

Usage:
    python evals/benchmark_runner.py [--dataset PATH] [--mode auto|live|mock] [--k INT] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.rag_evaluator import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Run quantitative RAG retrieval and groundedness benchmark evaluation."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "golden_eval_dataset.json"),
        help="Path to the golden evaluation dataset JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "live", "mock"],
        default="auto",
        help="Execution mode: auto (live if key present, else mock), live (Gemini API), or mock (deterministic offline).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-K documents to retrieve from vector store for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "reports"),
        help="Directory to save JSON and Markdown benchmark reports.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed case-by-case outputs to stdout.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset file not found at: {args.dataset}")
        sys.exit(1)

    print("=" * 70)
    print("🚀 Running Quantitative RAG Groundedness & Retrieval Benchmark")
    print(f"📁 Dataset: {args.dataset}")
    print(f"⚙️ Mode: {args.mode.upper()} | Top-K: {args.k}")
    print("=" * 70)

    evaluator = RAGEvaluator(mode=args.mode)
    summary = evaluator.run_benchmark(dataset_path=args.dataset, top_k=args.k)

    # Print markdown report to stdout
    md_report = summary.to_markdown_report()
    print("\n" + md_report)

    # Save reports
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "rag_benchmark_summary.json")
    md_path = os.path.join(args.output_dir, "rag_benchmark_summary.md")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(summary.model_dump_json(indent=2))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"\n💾 Benchmark JSON saved to: {json_path}")
    print(f"📄 Markdown Scorecard saved to: {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
