import time
from typing import List, Dict, Any
from src.retrieval.hybrid_retriever import HybridRetriever, Strategy
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.evaluation.logger import EvalLogger

# Singleton retriever for standard non-benchmark flows (matching original behavior but upgraded to hybrid)
default_retriever = HybridRetriever(strategy="hybrid")
evaluator = RagasEvaluator()
logger = EvalLogger()

def add_docs(texts: List[str]) -> None:
    """Add documents to the retrieval index."""
    default_retriever.add_docs(texts)

def retrieve(query: str, k: int = 3) -> List[str]:
    """Retrieve documents using the default hybrid strategy."""
    return default_retriever.retrieve(query, k=k)

def evaluate_and_log(query: str, answer: str, contexts: List[str], strategy: Strategy = "hybrid", skip_eval: bool = False, latency: float = 0.0, cost: int = 0) -> Dict[str, float]:
    """Evaluate an answer using RAGAS and log the result."""
    if skip_eval:
        scores = {
            "faithfulness": 0.0, 
            "answer_relevancy": 0.0, 
            "context_precision": 0.0, 
            "context_recall": 0.0
        }
    else:
        scores = evaluator.evaluate_response(query, answer, contexts)
        
    logger.log(query, strategy, scores, latency, cost)
    return scores

def run_retrieval_benchmark(query: str, docs: List[str], skip_eval: bool = False) -> Dict[str, Any]:
    """Run a single query across all four retrieval strategies and evaluate."""
    import asyncio
    import time
    
    # Needs to be called with some docs already in the index if we want meaningful results
    # We create a fresh hybrid retriever and populate it so it's isolated
    benchmark_retriever = HybridRetriever(strategy="hybrid")
    if docs:
        benchmark_retriever.add_docs(docs)

    strategies: List[Strategy] = ["dense", "sparse", "hybrid", "reranker"]
    results = {}

    for strat in strategies:
        benchmark_retriever.strategy = strat
        start = time.perf_counter()
        
        contexts = benchmark_retriever.retrieve(query, k=5)
        
        latency = time.perf_counter() - start
        
        # We need a dummy answer for evaluation if we don't call an LLM here
        dummy_answer = " ".join([c[:200] for c in contexts]) if contexts else "No contexts"
        
        scores = evaluate_and_log(
            query=query, 
            answer=dummy_answer, 
            contexts=contexts, 
            strategy=strat, 
            skip_eval=skip_eval,
            latency=latency,
            cost=200 # Dummy token cost
        )
        
        results[strat] = {
            "scores": scores,
            "latency": latency * 1000, # ms
            "contexts": contexts
        }

    return results
