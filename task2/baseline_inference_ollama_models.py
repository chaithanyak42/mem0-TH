# baseline_inference_ollama.py
"""
Baseline–benchmark: Ollama models (FP16 vs 4‑bit).
Outputs per‑model CSV + JSON with latency & throughput stats.
Run:  python baseline_inference_ollama.py
"""

import asyncio, csv, json, os, statistics, sys, time, shutil, subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm   # ←  NEW  (pip install tqdm)


try:
    from ollama import Client          # pip install ollama
except ImportError:
    sys.exit("❌  pip install ollama first")

# ---------------------------------------------------------------------
# 0.  CONFIG
# ---------------------------------------------------------------------


MODELS = [
    "llama3.1_8b-instruct-finetuned-q4_K_M", # Only benchmark the fine-tuned GGUF model
]

BENCH_DIR   = Path("benchmarks")
BENCH_DIR.mkdir(exist_ok=True)

NUM_PROMPTS      = 100
MAX_NEW_TOKENS   = 64               # keep it small → fast
TEMP             = 0.2
PROMPTS: List[str] = [
    "Explain photosynthesis in one concise paragraph.",
    "Translate the sentence \"Good morning, how are you?\" into French.",
    "Provide three vegetarian protein sources and one quick recipe idea.",
    "Summarize the plot of \"The Matrix\" in two sentences.",
    "Write a haiku that captures the feeling of autumn rain.",
    "List five potential names for a cat‑themed coffee shop.",
    "Compare nuclear fission and fusion in a short bullet list.",
    "Generate Python code that reverses a string.",
    "Suggest a 30‑minute beginner workout with no equipment.",
    "What are the primary causes of climate change?",
    "Turn the proverb \"Actions speak louder than words\" into an emoji story.",
    "Describe the taste profile of Ethiopian Yirgacheffe coffee.",
    "Draft a polite email declining a meeting invitation.",
    "Give me three fun facts about Jupiter.",
    "Create a two‑line slogan for a reusable water‑bottle brand.",
    "Outline the steps to brew pour‑over coffee.",
    "Translate \"Knowledge is power\" into Latin.",
    "Provide a one‑sentence definition of machine learning.",
    "Recommend a weekend itinerary for Paris first‑timers.",
    "Explain the difference between HTTP and HTTPS.",
    "Generate a SQL query to select all rows where \"status\" = \"active\".",
    "Write a six‑word memoir about summer camp.",
    "Name four common renewable‑energy technologies.",
    "Describe the aroma of freshly baked bread.",
    "Suggest a book similar to \"Atomic Habits\".",
    "Summarize how blockchains achieve consensus.",
    "List three advantages of electric vehicles.",
    "Give a Shakespearean compliment in modern English.",
    "Provide a single sentence in iambic pentameter.",
    "Explain what DNS does in a computer network.",
    "Convert 98 °F to Celsius.",
    "Create a witty hashtag for a tech conference.",
    "Draft a tweet announcing a product‑launch livestream.",
    "Recommend two beginner‑friendly houseplants.",
    "What is the Pythagorean theorem?",
    "Write JavaScript to check if a number is prime.",
    "Describe the sound of a distant thunderstorm.",
    "Give three mindfulness tips for reducing stress.",
    "Summarize Immanuel Kant's \"categorical imperative\".",
    "Translate \"I would like a cup of tea\" into Japanese (romaji).",
    "Explain the term \"serverless computing\".",
    "Provide a fun fact about honey‑bees.",
    "Suggest a side dish for grilled salmon.",
    "Write a two‑sentence horror story.",
    "Outline the lifecycle of a butterfly.",
    "Recommend a podcast for learning Spanish.",
    "Compare SSDs and HDDs in two sentences.",
    "Generate an HTML snippet for a red button.",
    "Name three key features of the Rust language.",
    "Explain what a VPN is to a 10‑year‑old.",
    "Provide the chemical formula of table salt.",
    "Give two historical events that occurred in 1969.",
    "Translate \"Happy New Year\" into Mandarin (pinyin).",
    "Suggest a study playlist genre.",
    "Describe the smell of fresh cut grass in five words.",
    "Write a catchy title for a blog on productivity.",
    "Explain why the sky appears blue.",
    "List three famous impressionist painters.",
    "Create a one‑line bash command to count files in a folder.",
    "Suggest a movie for fans of \"Inception\".",
    "Outline the stages of the water cycle.",
    "Provide a synonym for \"resilient\".",
    "Draft a restaurant tagline centered on sustainability.",
    "Explain the term \"yield farming\" in DeFi.",
    "Recommend a board game for four players.",
    "Convert 250 km to miles.",
    "Describe the texture of velvet.",
    "List three benefits of meditation.",
    "Write C code to swap two integers.",
    "Name two main ingredients in hummus.",
    "Translate \"Cheers!\" into German.",
    "Summarize the Hero's Journey in one sentence.",
    "Give an example of an oxymoron.",
    "Suggest a dessert that pairs with espresso.",
    "Explain what an API gateway does.",
    "Provide the RGB values for pure magenta.",
    "Recommend a beginner friendly Linux distro.",
    "Describe what a Doppler radar measures.",
    "Draft a one‑line motivational quote about learning.",
    "List four famous landmarks in Rome.",
    "Write a limerick about a curious cat.",
    "Translate \"Thank you very much\" into Portuguese.",
    "Explain the difference between RAM and ROM.",
    "Suggest three ways to conserve water at home.",
    "Provide the first line of the Fibonacci sequence.",
    "Write a CSS rule to center text.",
    "Name the author of \"Pride and Prejudice\".",
    "Summarize quantum entanglement in one sentence.",
    "Recommend a high‑protein vegan snack.",
    "Convert 45 °C to Fahrenheit.",
    "Describe the feeling of walking on a sandy beach.",
    "List three uses of graphene.",
    "Create a catchy app‑store description for a note‑taking app (max 25 words).",
    "Explain what a compiler does.",
    "Provide an example of a homophone pair.",
    "Suggest a gift for someone who loves astronomy.",
    "Translate \"See you tomorrow\" into Italian.",
    "Write a short apology text for sending the wrong file.",
    "Give two key takeaways from the Agile manifesto.",
    "Explain \"Occam's razor\" in plain English.",
]


# ---------------------------------------------------------------------
# 1.  HELPER – ensure Ollama model
# ---------------------------------------------------------------------
def have_ollama() -> None:
    if shutil.which("ollama") is None:
        sys.exit("❌  Ollama binary not found – install from https://ollama.com first")

def ensure_model(tag: str) -> None:
    have_ollama()
    listed = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
    if tag.split(":")[0] not in listed:
        print(f"⏬ pulling {tag} …"); sys.stdout.flush()
        subprocess.run(["ollama", "pull", tag], check=True)
    else:
        print(f"✅ model {tag} already present.")

# ---------------------------------------------------------------------
# 2.  BENCHMARK CORE
# ---------------------------------------------------------------------
async def bench_model(tag: str) -> Dict:
    ensure_model(tag)
    client  = Client()
    results = []
    total_tokens = total_time = 0.0

    # warm‑up
    await asyncio.to_thread(
        client.chat,
        model=tag,
        messages=[{"role":"user","content":"Hello"}],
        options={"temperature":TEMP,"num_predict":4},
    )

    for idx, prompt in enumerate(tqdm(PROMPTS, desc=f"{tag}", ncols=88), 1):
        start = time.perf_counter()
        chat  = await asyncio.to_thread(
            client.chat,
            model   = tag,
            messages=[{"role":"user","content":prompt}],
            options = {"temperature":TEMP, "num_predict":MAX_NEW_TOKENS},
        )
        dur         = time.perf_counter() - start
        response_txt= chat["message"]["content"]
        gen_tokens  = chat.get("eval_count", MAX_NEW_TOKENS) # Use .get() for safety
        tps         = gen_tokens / dur if dur else 0

        # quick inline preview (trim long lines)
        preview = response_txt.strip().replace("\n", " ")[:80]
        # Use tqdm's write method to avoid interfering with the progress bar
        tqdm.write(f"  • {idx:3d}/{NUM_PROMPTS}  {preview}…")

        results.append({
            "index": idx,
            "latency_s": dur,
            "gen_tokens": gen_tokens,
            "tps": tps,
            "response_txt": response_txt
        })
        total_tokens += gen_tokens
        total_time   += dur

    latencies = [r["latency_s"] for r in results]
    summary   = {
        "model"              : tag,
        "timestamp"          : datetime.utcnow().isoformat(),
        "num_prompts"        : NUM_PROMPTS,
        "avg_latency_s"      : statistics.mean(latencies),
        "median_latency_s"   : statistics.median(latencies),
        "p95_latency_s"      : statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else float('nan'), # Handle case with < 20 results
        "overall_toks_per_s" : total_tokens / total_time if total_time else 0
    }
    # ---------- persist ----------
    root = BENCH_DIR / tag.replace(":","_")
    root.mkdir(exist_ok=True)
    if results: # Ensure results list is not empty before accessing keys
        with open(root / "raw.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# ---------------------------------------------------------------------
# 3.  MAIN
# ---------------------------------------------------------------------
async def main():
    have_ollama()
    all_summaries = []
    for tag in MODELS:
        print(f"\n=== BENCHMARK {tag} ===")
        summary = await bench_model(tag)
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2))

    # little table for console
    print("\n─ Aggregate ─")
    for s in all_summaries:
        print(f"{s['model']:<36} "
              f"avg {s['avg_latency_s']:.2f}s  "
              f"tok/s {s['overall_toks_per_s']:.1f}")

if __name__ == "__main__":
    asyncio.run(main())