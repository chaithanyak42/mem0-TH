"""
Benchmark – pre-merged 4-bit Llama-3.1-8B           │  Run:
----------------------------------------------------------------┤  python baseline_fine_tuned_4bit_llama31_8b.py
Loads the pre-merged 4-bit model directly and times generation
for each prompt.

Deps: torch, transformers ≥4.51, bitsandbytes, accelerate, tqdm
"""

# ---------------------------------------------------------------------
# 0.  Imports & logging
# ---------------------------------------------------------------------
import asyncio, csv, json, logging, statistics, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1.  Benchmark constants
# ---------------------------------------------------------------------
MODEL_ID      = "cosmos98a/mem0-merged-llama3.1-8b-4bit" # Pre-merged 4-bit model

BENCH_DIR        = Path("benchmarks")
BENCH_DIR.mkdir(exist_ok=True)

MAX_NEW_TOKENS = 64
TEMP           = 0.1
TOP_P          = 0.9

# Paste your 100 prompts here ↓
PROMPTS: List[str] = [
    "Explain photosynthesis in one concise paragraph.",
    "Translate the sentence Good morning, how are you? into French.",
    "Provide three vegetarian protein sources and one quick recipe idea.",
    "Summarize the plot of The Matrix in two sentences.",
    "Write a haiku that captures the feeling of autumn rain.",
    "List five potential names for a cat-themed coffee shop.",
    "Compare nuclear fission and fusion in a short bullet list.",
    "Generate Python code that reverses a string.",
    "Suggest a 30-minute beginner workout with no equipment.",
    "What are the primary causes of climate change?",
    "Turn the proverb Actions speak louder than words into an emoji story.",
    "Describe the taste profile of Ethiopian Yirgacheffe coffee.",
    "Draft a polite email declining a meeting invitation.",
    "Give me three fun facts about Jupiter.",
    "Create a two‑line slogan for a reusable water‑bottle brand.",
    "Outline the steps to brew pour‑over coffee.",
    "Translate Knowledge is power into Latin.",
    "Provide a one‑sentence definition of machine learning.",
    "Recommend a weekend itinerary for Paris first‑timers.",
    "Explain the difference between HTTP and HTTPS.",
    "Generate a SQL query to select all rows where status = active.",
    "Write a six‑word memoir about summer camp.",
    "Name four common renewable‑energy technologies.",
    "Describe the aroma of freshly baked bread.",
    "Suggest a book similar to Atomic Habits.",
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
    "Summarize Immanuel Kant's categorical imperative.",
    "Translate I would like a cup of tea into Japanese (romaji).",
    "Explain the term serverless computing.",
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
    "Translate Happy New Year into Mandarin (pinyin).",
    "Suggest a study playlist genre.",
    "Describe the smell of fresh cut grass in five words.",
    "Write a catchy title for a blog on productivity.",
    "Explain why the sky appears blue.",
    "List three famous impressionist painters.",
    "Create a one‑line bash command to count files in a folder.",
    "Suggest a movie for fans of Inception.",
    "Outline the stages of the water cycle.",
    "Provide a synonym for resilient.",
    "Draft a restaurant tagline centered on sustainability.",
    "Explain the term yield farming in DeFi.",
    "Recommend a board game for four players.",
    "Convert 250 km to miles.",
    "Describe the texture of velvet.",
    "List three benefits of meditation.",
    "Write C code to swap two integers.",
    "Name two main ingredients in hummus.",
    "Translate Cheers! into German.",
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
    "Translate Thank you very much into Portuguese.",
    "Explain the difference between RAM and ROM.",
    "Suggest three ways to conserve water at home.",
    "Provide the first line of the Fibonacci sequence.",
    "Write a CSS rule to center text.",
    "Name the author of Pride and Prejudice.",
    "Summarize quantum entanglement in one sentence.",
    "Recommend a high‑protein vegan snack.",
    "Convert 45 °C to Fahrenheit.",
    "Describe the feeling of walking on a sandy beach.",
    "List three uses of graphene.",
    "Create a catchy app‑store description for a note‑taking app (max 25 words).",
    "Explain what a compiler does.",
    "Provide an example of a homophone pair.",
    "Suggest a gift for someone who loves astronomy.",
    "Translate See you tomorrow into Italian.",
    "Write a short apology text for sending the wrong file.",
    "Give two key takeaways from the Agile manifesto.",
    "Explain Occam's razor in plain English.",
]

# ---------------------------------------------------------------------
# 2.  Load model ------------------------------------------------
# ---------------------------------------------------------------------
def load_model_tokenizer(
    model_id: str = MODEL_ID
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the pre-merged 4-bit model and tokenizer directly."""
    logger.info(f"🔄 Loading model: {model_id} (4-bit NF4)")
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map={"": 0},          # <-- Force all layers to cuda:0
    )
    logger.info(f"✅ model loaded in {time.time() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model.eval()
    # No need for model.to("cuda") here, it's already on the correct device
    return model, tokenizer


# ---------------------------------------------------------------------
# 3.  Generation helper (sync) ----------------------------------------
# ---------------------------------------------------------------------
def generate_once(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, int]:
    """One prompt → response text & (# new tokens)."""
    prompt_txt = prompt.strip()          # no chat template
    if not prompt_txt.endswith("\n"):
        prompt_txt += "\n"
    prompt_txt += "### Response:\n"
    inputs = tok([prompt_txt], return_tensors="pt").to(model.device)
    inp_len = inputs.input_ids.shape[-1]

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=tok.pad_token_id,
        eos_token_id=[
            tok.eos_token_id,
            tok.convert_tokens_to_ids("<|eot_id|>"),
        ],
    )

    new_ids = gen_ids[:, inp_len:]
    return tok.decode(new_ids[0], skip_special_tokens=True), new_ids.shape[-1]


# ---------------------------------------------------------------------
# 4.  Benchmark coroutine ---------------------------------------------
# ---------------------------------------------------------------------
async def bench() -> Dict:
    if not PROMPTS:
        sys.exit("❌ PROMPTS list is empty – paste the 100 prompts first.")

    model, tok = load_model_tokenizer()

    # warm-up
    _ = await asyncio.to_thread(generate_once, model, tok, "Hello", 4, TEMP, TOP_P)

    results, tot_tokens, tot_time = [], 0.0, 0.0
    for idx, prompt in enumerate(
        tqdm(PROMPTS, desc="Benchmark", ncols=90), start=1
    ):
        t0 = time.perf_counter()
        try:
            resp, n_toks = await asyncio.to_thread(
                generate_once, model, tok, prompt, MAX_NEW_TOKENS, TEMP, TOP_P
            )
            dur = time.perf_counter() - t0
            tps = n_toks / dur if dur > 0 else 0
            tot_tokens += n_toks
            tot_time += dur
        except Exception as e:
            resp, n_toks, dur, tps = f"ERROR: {e}", 0, 0.0, 0.0
            logger.error(f"Prompt {idx} failed: {e}")

        tqdm.write(
            f"• {idx:3d}/{len(PROMPTS)} – {n_toks} toks  {dur*1000:6.1f} ms  {tps:6.1f} tps"
        )
        results.append(
            dict(index=idx, latency_s=dur, gen_tokens=n_toks, tps=tps, response=resp)
        )

    # aggregate
    lats = [r["latency_s"] for r in results if r["gen_tokens"] > 0]
    summary = {
        "model": MODEL_ID,
        "timestamp": datetime.utcnow().isoformat(),
        "prompts": len(PROMPTS),
        "avg_latency_s": statistics.mean(lats) if lats else None,
        "median_latency_s": statistics.median(lats) if lats else None,
        "p95_latency_s": statistics.quantiles(lats, n=20)[-1] if len(lats) >= 20 else None,
        "overall_toks_per_s": tot_tokens / tot_time if tot_time else 0,
    }

    # save
    safe_name = MODEL_ID.replace("/", "_")
    root      = BENCH_DIR / safe_name
    root.mkdir(exist_ok=True)

    with open(root / "raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(bench())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
