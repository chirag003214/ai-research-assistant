import os, time, hashlib, json
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

def call_llm(prompt, max_tokens=512, retries=3, wait_time=25):
    cache_file = _cache_path(prompt)

    if os.path.exists(cache_file):
        return json.load(open(cache_file))["response"]

    for attempt in range(retries):
        try:
            response = completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens
            )
            text = response.choices[0].message.content
            json.dump({"response": text}, open(cache_file, "w"))
            return text

        except RateLimitError:
            if attempt < retries - 1:
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise






