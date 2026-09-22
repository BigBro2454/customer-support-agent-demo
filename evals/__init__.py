"""
Evaluation suite for Customer Support Agent RAG Pipeline.
Implements Google L5 Quantitative Groundedness, Retrieval Recall@K, and Guardrail Adherence Benchmarking.
"""

from evals.rag_evaluator import (
    RAGEvalCase,
    RAGEvaluator,
    CaseEvalResult,
    BenchmarkSummary,
)

__all__ = [
    "RAGEvalCase",
    "RAGEvaluator",
    "CaseEvalResult",
    "BenchmarkSummary",
]
