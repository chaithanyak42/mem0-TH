import asyncio, csv, json, os, subprocess, sys, shutil
from datetime import datetime
from pathlib import Path
from typing import List

from mem0 import AsyncMemory
from mem0.llms.configs import LlmConfig
from mem0.configs.base import MemoryConfig

# ── models already pulled ──────────────────────────────────────────────
MODEL_TAGS: List[str] = [
    "llama4:17b-scout-16e-instruct-fp16",
    "llama3.1:8b-instruct-fp16",
    "llama4:17b-scout-16e-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
]

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
USER_ID, AGENT_ID = "demo-user", "demo-agent"
MEMORIES = [
    "My favourite colour is blue.",
    "I live in Hyderabad.",
    "I'm allergic to peanuts.",
]
QUERY = "Where do I live and what food should I avoid?"

BENCH_DIR = Path("benchmarks")
BENCH_DIR.mkdir(exist_ok=True)

# ── helper to confirm model present ────────────────────────────────────
def ensure_model(tag: str) -> None:
    if shutil.which("ollama") is None:
        sys.exit("❌  Ollama CLI not found – install from https://ollama.com first.")
    listed = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if tag.split(":")[0] not in listed:
        subprocess.run(["ollama", "pull", tag], check=True)

# ── run one add/search round‑trip ──────────────────────────────────────
async def smoke_test(tag: str) -> dict:
    ensure_model(tag)

    llm_cfg = LlmConfig(provider="ollama", config={
        "model": tag,
        "ollama_base_url": OLLAMA_URL,
        "temperature": 0.2,
        "max_tokens": 512,
    })
    mem_cfg = MemoryConfig(version="v1.1");  mem_cfg.llm = llm_cfg
    mem = AsyncMemory(config=mem_cfg)

    for m in MEMORIES:
        await mem.add(m, user_id=USER_ID, agent_id=AGENT_ID)

    res = await mem.search(QUERY, user_id=USER_ID, agent_id=AGENT_ID, limit=5)
    hits = [r["memory"] for r in res.get("results", [])]

    ok_city  = any("hyderabad" in h.lower() for h in hits)
    ok_food  = any("peanut"    in h.lower() for h in hits)
    status   = ok_city and ok_food

    print(f"{'✅' if status else '❌'}  {tag:<45}  hits→ {hits[:2]}")
    return {
        "model": tag,
        "pass": status,
        "num_hits": len(hits),
        "hits": hits,
        "timestamp": datetime.utcnow().isoformat(),
    }

# ── main ───────────────────────────────────────────────────────────────
async def _main():
    rows = [await smoke_test(tag) for tag in MODEL_TAGS]

    # save results
    csv_path  = BENCH_DIR / "baseline_ollama_add_search.csv"
    json_path = BENCH_DIR / "baseline_ollama_add_search.json"
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, rows[0].keys()).writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))

    # paste‑ready README table
    print("\nMarkdown summary\n----------------")
    print("| Model | PASS | #Hits |")
    print("|-------|------|-------|")
    for r in rows:
        print(f"| `{r['model']}` | {'✅' if r['pass'] else '❌'} | {r['num_hits']} |")
    print(f"\nArtifacts saved → {csv_path.name} & {json_path.name} in ./benchmarks/")

if __name__ == "__main__":
    asyncio.run(_main())