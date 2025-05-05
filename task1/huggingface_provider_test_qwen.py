import asyncio, os, json
from datetime import datetime
from mem0 import AsyncMemory
from mem0.llms.configs import LlmConfig
from mem0.configs.base import MemoryConfig

if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set. Default embedder might fail.")
    
USER_ID  = "demo-user-qwen" 
AGENT_ID = "demo-agent-qwen"
MEMORIES = [
    "My favorite fruit is mango.",
    "I enjoy reading science fiction.",
    "I have a meeting tomorrow at 10 AM."
]
QUERY = "What is my favorite genre of book?"

async def main():
    print("Initializing AsyncMemory with Qwen provider...")

    qwen_llm_config = LlmConfig(
        provider="qwen",
        config={"model_path": "./models/Qwen_Qwen2.5-7B-Instruct-1M"}
    )

    mem_config = MemoryConfig()

    mem_config.llm = qwen_llm_config

    mem = AsyncMemory(config=mem_config)

    print(f"Adding {len(MEMORIES)} memories for user_id: {USER_ID}...")
    for text in MEMORIES:
        add_result = await mem.add(
            text,
            user_id=USER_ID,
            agent_id=AGENT_ID,
            metadata={"timestamp": datetime.utcnow().isoformat(), "source": "qwen_baseline_test"}
        )
      

    print("Memories added.")
    print("-" * 20)

    await asyncio.sleep(1)

    
    print(f"Searching memory with query: '{QUERY}' for user_id: {USER_ID}")
    try:
        search_result = await mem.search(QUERY, user_id=USER_ID)
        print(f"Search results:\n{json.dumps(search_result, indent=2)}")
        print("-" * 20)

     
        if search_result and search_result.get("results"):
            print("Verification: Search retrieved results (using default embedder).")
        else:
            print("Verification: Search failed to retrieve results.")

    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    print("Running Qwen Baseline Integration Test...")
    asyncio.run(main())
