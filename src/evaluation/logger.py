import json
import os
import time
from typing import Dict, Any

class EvalLogger:
    def __init__(self, log_dir: str = "eval_results"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.filepath = os.path.join(self.log_dir, "eval_logs.jsonl")

    def log(self, query: str, strategy: str, scores: Dict[str, float], latency: float, cost: int) -> None:
        entry = {
            "timestamp": time.time(),
            "query": query,
            "strategy": strategy,
            "scores": scores,
            "latency_ms": latency * 1000,
            "cost_tokens": cost
        }
        try:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Failed to write log: {e}")
