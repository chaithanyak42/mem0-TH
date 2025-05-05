"""
Gradio Chat Interface for Mem0 with a fine-tuned LLM backend.
Based on: https://www.zinyando.com/upgrading-your-ai-friend-building-a-gradio-gui-for-autogen-and-mem0-chatbots/
"""
import asyncio
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import gradio as gr
from dotenv import load_dotenv

# --- Mem0 Imports ---
from mem0 import AsyncMemory
from mem0.configs.base import MemoryConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.utils.factory import LlmFactory # Import factory to get LLM instance

# --- Configuration ---
load_dotenv() # Load environment variables if needed (e.g., API keys, though not for local models)

MODEL_TAG = "cosmos98a/mem0-merged-llama3.1-8b-4bit" # Your fine-tuned model ID
LLM_PROVIDER = "llama31_8b_finetuned_4bit" # Provider name registered in factory
EMBED_TAG = "nomic-embed-text" # Ollama embedder
EMBED_PROVIDER = "ollama"
VECTOR_PROVIDER = "qdrant"

USER_ID_PREFIX = "chat_user" # Use a prefix for user IDs in this app
AGENT_ID = "mem0-gradio-demo"

# Qdrant/History paths (relative to script location or use absolute paths)
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")
# Use distinct paths for this app's data if desired
QDRANT_PATH = os.path.join(mem0_dir, f"qdrant_data_{AGENT_ID}")
HISTORY_DB_PATH = os.path.join(mem0_dir, f"history_{AGENT_ID}.db")

# Constants
EMBEDDING_DIMS = 768 # For nomic-embed-text
MAX_HISTORY_LEN = 20 # Max recent messages to include in LLM prompt context
MAX_MEMORIES_IN_PROMPT = 5 # Max relevant memories to include in LLM prompt

# --- Mem0 Initialization ---
print("🔧 Configuring Mem0...")

# --- LLM config ---
llm_config_obj = LlmConfig(
    provider=LLM_PROVIDER,
    config={
        "model_id": MODEL_TAG,
        # Add any other necessary config params for your custom LLM class
        # temperature: 0.2 # Example: Lower temp for more deterministic responses
    }
)

# --- Embedder config (Ollama) ---
embedder_config_obj = EmbedderConfig(
    provider=EMBED_PROVIDER,
    config={
        "model": EMBED_TAG,
        "embedding_dims": EMBEDDING_DIMS,
        "ollama_base_url": None # Use default localhost
    }
)

# --- Vector store config (Qdrant) ---
# Ensure the directory exists for on-disk storage
Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
vector_store_config_obj = VectorStoreConfig(
    provider=VECTOR_PROVIDER,
    config={
        # Collection name based on Agent ID - assumes one collection for the app
        "collection_name": f"mem0_{AGENT_ID}_collection",
        "embedding_model_dims": EMBEDDING_DIMS, # Must match embedder
        "path": QDRANT_PATH,
        "on_disk": True
    }
)

# --- Main Memory Config ---
mem0_config = MemoryConfig(
    version="v1.1",
    llm=llm_config_obj,
    embedder=embedder_config_obj,
    vector_store=vector_store_config_obj,
    history_db_path=HISTORY_DB_PATH,
)

# Instantiate AsyncMemory
try:
    print("🚀 Initializing AsyncMemory...")
    # Note: Using a single AsyncMemory instance globally. Be mindful of concurrency if scaling.
    mem = AsyncMemory(mem0_config)
    print("✅ AsyncMemory initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing AsyncMemory: {e}")
    print("   Check provider names and factory registration.")
    sys.exit(1)

# Instantiate LLM directly using the factory for generation
try:
    print(f"🚀 Initializing LLM provider '{LLM_PROVIDER}' via factory...")
    # Pass the *inner* config dict to the factory
    llm_instance = LlmFactory.create(LLM_PROVIDER, llm_config_obj.config)
    print(f"✅ LLM instance created successfully.")
except Exception as e:
    print(f"❌ Error initializing LLM via factory: {e}")
    sys.exit(1)


# --- Helper Functions ---
def generate_user_id(name: str) -> str:
    """Creates a simple user ID based on the name."""
    # Basic sanitization - replace spaces, lowercase
    sanitized_name = name.strip().lower().replace(" ", "_")
    return f"{USER_ID_PREFIX}_{sanitized_name}"

async def create_prompt(user_input: str, user_id: str, chat_history: List[Tuple[str, str]]) -> str:
    """Generate the prompt based on user input, conversation history, and memory context."""
    print(f"🧠 Creating prompt for user '{user_id}'...")
    # 1. Search for relevant memories
    try:
        print(f"   Searching memories for query: '{user_input[:50]}...'")
        search_results = await mem.search(
            user_input,
            user_id=user_id,
            agent_id=AGENT_ID, # Filter by agent if needed
            limit=MAX_MEMORIES_IN_PROMPT
        )
        # Ensure results exist and format them
        retrieved_memories = [m["memory"] for m in search_results.get("results", [])] if search_results else []
        memory_context = "\\n".join([f"- {mem}" for mem in retrieved_memories])
        print(f"   Found {len(retrieved_memories)} relevant memories.")
    except Exception as e:
        print(f"   ⚠️ Error searching memories: {e}")
        memory_context = "No memories found or error retrieving them."

    # 2. Format recent chat history
    # Gradio history is [(user, bot), (user, bot), ...]
    # We need to format it for the LLM prompt
    recent_history = chat_history[-MAX_HISTORY_LEN:]
    history_str = "\\n".join([f"{name}: {message}" for name, message in recent_history])

    # 3. Construct the final prompt
    # Adapt this prompt template as needed for your fine-tuned model's expected format
    # This uses a generic approach, similar to the blog post but without AutoGen specifics
    prompt = f"""You are a helpful AI assistant with memory. Your goal is to assist the user based on the current conversation and relevant past memories.

Relevant Memories (if any):
{memory_context if memory_context else "None"}

Recent Chat History:
{history_str if history_str else "This is the start of the conversation."}

User ({user_id}): {user_input}
Assistant:""" # Model should continue from here

    print(f"   Generated prompt:\n-------\n{prompt[:500]}...\n-------")
    return prompt

async def chatbot_response(user_input: str, chat_history: List[Tuple[str, str]], user_id: str) -> Tuple[List[Tuple[str, str]], str]:
    """Handle chat interaction, generate response, update memory, and update the UI."""
    assistant_name = "Assistant" # Name for the bot in chat history

    if not user_input:
        return chat_history, "" # No input, do nothing

    print(f"💬 User '{user_id}' input: {user_input}")
    # Immediately display the user's message in the chat
    chat_history.append((user_id, user_input))

    # 1. Generate a prompt including context (memories + history)
    prompt_for_llm = await create_prompt(user_input, user_id, chat_history)

    # 2. Get response from the LLM
    assistant_reply = ""
    try:
        print(f"   Generating LLM response...")
        # Use the llm_instance created via factory
        # Note: llm_instance might be sync or async depending on its implementation.
        # Assuming generate_response is async based on common patterns, otherwise adjust call.
        # Pass the prompt as a list of messages if required by your LLM wrapper
        messages_for_llm = [{"role": "user", "content": prompt_for_llm}] # Adjust role/content structure if needed
        
        # Check if the method is async
        if asyncio.iscoroutinefunction(llm_instance.generate_response):
             assistant_reply = await llm_instance.generate_response(messages=messages_for_llm)
        else:
             # Run sync function in thread pool executor if needed in async context
             loop = asyncio.get_running_loop()
             assistant_reply = await loop.run_in_executor(
                 None, llm_instance.generate_response, messages_for_llm
             )
        
        print(f"   LLM Reply: {assistant_reply[:100]}...")

    except Exception as e:
        print(f"   ❌ Error generating LLM response: {e}")
        assistant_reply = f"Sorry, I encountered an error trying to respond: {e}"

    # 3. Store the *actual* user input and assistant response in Mem0
    # Format as a list of messages for mem.add
    conversation_to_add = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_reply},
    ]
    try:
        print(f"   Adding interaction to memory for user '{user_id}'...")
        # Use infer=True so mem0 processes the interaction (e.g., extracts facts)
        add_result = await mem.add(
            conversation_to_add,
            user_id=user_id,
            agent_id=AGENT_ID, # Associate with this app instance
            infer=True
        )
        # Handle potential errors during add, like the JSON issues seen before
        if not add_result or not add_result.get("results"):
             print(f"   ⚠️ Mem0 add operation returned unexpected result: {add_result}")
        else:
             print(f"   ✅ Interaction added to memory.")
    except Exception as e:
        print(f"   ❌ Error adding to memory: {e}")
        # Potentially inform the user in the chat?
        # assistant_reply += "\n(Note: There was an issue saving this interaction to memory.)"


    # 4. Add the bot's reply to the chat history for Gradio display
    chat_history.append((assistant_name, assistant_reply))

    # 5. Return updated history and clear input box
    return chat_history, ""


# --- Gradio Interface ---
print("🎨 Building Gradio interface...")

with gr.Blocks(title="Mem0 Fine-tuned Chat") as app:
    gr.Markdown("# Chat with Mem0 (Fine-tuned Llama 3.1 8B)")
    gr.Markdown(f"Using LLM: `{MODEL_TAG}` and Embedder: `{EMBED_TAG}`")

    # State to store chat history and user name/ID
    chat_history = gr.State([])
    user_id = gr.State("")

    with gr.Row() as name_group:
        name_input = gr.Textbox(
            label="Enter your name to start chat:",
            placeholder="Your name here",
            interactive=True,
            scale=3, # Give more width to input
        )
        start_button = gr.Button("Start Chat", scale=1)

    with gr.Column(visible=False) as chat_group:
        gr.Markdown(lambda u: f"**Chatting as: {u}**", inputs=user_id) # Display current user ID
        chat_interface = gr.Chatbot(
            label="Chat History",
            height=600,
            bubble_full_width=False,
            show_label=False,
            # render=False # Removed if defined inline
        )
        with gr.Row():
            user_input_box = gr.Textbox(
                label="Your message:",
                placeholder="Type your message and press Enter...",
                interactive=True,
                scale=4, # Make input wider
                autofocus=True,
            )
            send_button = gr.Button("Send", scale=1)

    # --- Event Handlers ---
    def start_chat_ui_update(name: str) -> tuple:
        """Updates UI visibility and sets user_id state when chat starts."""
        if not name.strip():
            # Maybe show an error using gr.Warning or disable button
            print("⚠️ Name cannot be empty.")
            # To prevent proceeding, we don't update visibility
            # This requires rethinking the output structure slightly if we want to show error
            # For simplicity, just return current state essentially
            return (
                gr.update(visible=True),  # name_group stays visible
                gr.update(visible=False), # chat_group stays hidden
                gr.update(),              # chat_interface no change
                gr.update(),              # user_input_box no change
                "",                       # user_id state stays empty
            )
        generated_user_id = generate_user_id(name)
        print(f"🚀 Starting chat for user: {name} (ID: {generated_user_id})")
        # Hide name input, show chat
        return (
            gr.update(visible=False), # Hide name_group
            gr.update(visible=True),  # Show chat_group
            [],                       # Clear chatbot display
            gr.update(value="", placeholder="Type your message..."), # Clear input box
            generated_user_id,        # Set user_id state
        )

    start_button.click(
        fn=start_chat_ui_update,
        inputs=[name_input],
        outputs=[name_group, chat_group, chat_interface, user_input_box, user_id],
        queue=False, # Run quickly
    )

    # Define submit action for Textbox (Enter key) and Button click
    submit_action = user_input_box.submit if hasattr(user_input_box, 'submit') else send_button.click

    submit_action(
        fn=chatbot_response,
        inputs=[user_input_box, chat_history, user_id],
        outputs=[chat_interface, user_input_box], # Update chat, clear input
        # queue=True # Allow requests to queue for potentially long LLM responses
    )
    # Also link the button click explicitly if submit doesn't cover it or for clarity
    if not hasattr(user_input_box, 'submit'):
         send_button.click(
             fn=chatbot_response,
             inputs=[user_input_box, chat_history, user_id],
             outputs=[chat_interface, user_input_box],
         )

# --- Launch App ---
if __name__ == "__main__":
    print("🚀 Launching Gradio app...")
    # Set share=True to get a public link (useful for remote access/testing)
    app.launch(share=False)
