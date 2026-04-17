import os, time, hashlib, json, random
from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import RateLimitError

load_dotenv()

MODEL = "groq/llama-3.1-8b-instant"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(prompt):
    key = hashlib.md5(prompt.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")

def call_llm(prompt: str, max_tokens: int = 512, retries: int = 3, wait_time: int = 25) -> str:
    cache_file = _cache_path(prompt)

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)["response"]

    for attempt in range(retries):
        try:
            response = completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens
            )
            text = response.choices[0].message.content
            with open(cache_file, "w") as f:
                json.dump({"response": text}, f)
            return text

        except RateLimitError:
            if attempt < retries - 1:
                delay = wait_time * (2 ** attempt) + random.uniform(0, 2)
                print(f"Rate limit hit. Waiting {delay:.1f}s (attempt {attempt + 1})...")
                time.sleep(delay)
            else:
                raise






