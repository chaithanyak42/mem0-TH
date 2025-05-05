import asyncio, os, json
from datetime import datetime
from mem0 import AsyncMemory

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Set OPENAI_API_KEY before running this script.")

USER_ID = "demo-user"
AGENT_ID = "demo-agent"
MEMORIES = [
    "My favorite color is blue.",
    "I live in Hyderabad.",
    "I'm allergic to peanuts."
]
QUERY = "Where do I live and what food should I avoid?"

async def main():
    # Instantiate AsyncMemory directly
    print("Initializing AsyncMemory...")
    mem = AsyncMemory()
    print("AsyncMemory initialized.")

    # Add multiple memories with metadata in a loop
    print(f"Adding {len(MEMORIES)} memories for user_id: {USER_ID}...")
    for text in MEMORIES:
        add_result = await mem.add(
            text,
            user_id=USER_ID,
            agent_id=AGENT_ID,
            metadata={"timestamp": datetime.utcnow().isoformat(), "source": "baseline_test"}
        )
        # Optional: print add result for verification, can be verbose
        # print(f"  Added '{text}'. Result: {json.dumps(add_result, indent=2)}")

    print("Memories added.")
    print("-" * 20)
    
    # Wait a moment if necessary (e.g., for background processing or eventual consistency)
    await asyncio.sleep(1) 

    # Perform a single search relevant to the added memories
    print(f"Searching memory with query: '{QUERY}' for user_id: {USER_ID}")
    try:
        search_result = await mem.search(QUERY, user_id=USER_ID)
        print(f"Search results:\n{json.dumps(search_result, indent=2)}")
        print("-" * 20)

        if search_result and search_result.get("results"):
            print("Verification successful: Found relevant memories.")
        else:
            print("Verification potentially failed: No relevant memories found.")
            
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    # Ensure the OPENAI_API_KEY check happens before running the async main
    if "OPENAI_API_KEY" in os.environ:
        print("OPENAI_API_KEY found. Proceeding...")
        asyncio.run(main())
    else:
        # Error is raised before this point if key is missing
        pass # Should not be reached due to the raise 