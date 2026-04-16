import os
from typing import List, Dict, Any

class RagasEvaluator:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        self.ready = bool(self.openai_key or self.groq_key)
        if not self.ready:
            print("Warning: Neither OPENAI_API_KEY nor GROQ_API_KEY found, RAGAS might fail if it relies on default LLMs.")

    def evaluate_response(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        fallback_scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }
        
        if not contexts:
            # If no contexts retrieved, RAG is likely 0
            return fallback_scores

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from langchain_openai import ChatOpenAI
            from langchain_openai import OpenAIEmbeddings

            # Build dummy dataset
            # Note: Context recall typically expects "ground_truth", using answer as a proxy here if strictly required
            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [answer]
            }
            
            # Using ChatOpenAI or similar as judge
            # For simplicity, if OPENAI config is there
            llm = ChatOpenAI(model="gpt-3.5-turbo") if self.openai_key else None
            embeddings = OpenAIEmbeddings() if self.openai_key else None
            
            dataset = Dataset.from_dict(data)
            
            # Call evaluate
            kwargs = {}
            if llm:
                kwargs["llm"] = llm
            if embeddings:
                kwargs["embeddings"] = embeddings
                
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                raise_exceptions=False,
                **kwargs
            )
            
            return {
                "faithfulness": result.get("faithfulness", 0.0),
                "answer_relevancy": result.get("answer_relevancy", 0.0),
                "context_precision": result.get("context_precision", 0.0),
                "context_recall": result.get("context_recall", 0.0)
            }
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            return fallback_scores
