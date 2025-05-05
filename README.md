# Mem0 Small Model Integration Challenge - Progress Log

This document details the process and results of integrating smaller, efficient language models with the Mem0 memory layer.

## Task 1: Environment & Baseline Setup

### 1.1 Mem0 Installation & SDK Verification

**Objective:** Install the Mem0 Open Source Python SDK locally and verify basic `add`/`search` functionality using the default configuration (OpenAI backend).

**Steps:**

1.  **Clone Repository:** `git clone https://github.com/mem0ai/mem0.git`
2.  **Navigate to Directory:** `cd mem0`
3.  **Create Virtual Environment:** `uv venv .venv --seed`
4.  **Activate Environment:** `source .venv/bin/activate`
5.  **Install Poetry:** `uv pip install poetry`
6.  **Install Dependencies:** `make install`
7.  **Set OpenAI Key:** Ensure the `OPENAI_API_KEY` environment variable is set, as the default configuration uses OpenAI.
8.  **Run Verification Script:** Execute `python mem0_installation_verification.py`. This script initializes `AsyncMemory` and performs basic `add` and `search` operations.

**Result:**

The `mem0_installation_verification.py` script completed successfully. The output below confirms that memories were added and relevant results were retrieved using the default SDK setup:

```text
OPENAI_API_KEY found. Proceeding...
Initializing AsyncMemory...
AsyncMemory initialized.
Adding 3 memories for user_id: demo-user...
Memories added.
--------------------
Searching memory with query: 'Where do I live and what food should I avoid?' for user_id: demo-user
Search results:
{
  "results": [
    {
      "id": "017db352-3121-496e-b1a7-3f650ca32e26", // Example ID
      "memory": "Lives in Hyderabad",
      // ... (other fields omitted for brevity) ...
    },
    {
      "id": "d6e9ccaa-3dae-4390-a288-d0831a0c0f70", // Example ID
      "memory": "Allergic to peanuts",
      // ... (other fields omitted for brevity) ...
    },
    // ... (other results omitted for brevity) ...
  ]
}
--------------------
Verification successful: Found relevant memories.
```

**Status:** Completed.

### 1.2 Model Acquisition & Baseline Integration (Ollama)

**Objective:** Acquire the baseline models (Llama 4 Scout variant, 8B Instruct variant) and verify their integration as Mem0 backends using Ollama.

**Note:** While the challenge suggested acquiring models directly from Hugging Face, access issues with Llama 4 Scout and time constraints led to using Ollama as the primary method for obtaining and serving the baseline models.

**Steps:**

1.  **Install Ollama:** Ensured Ollama was installed and running (`curl -fsSL https://ollama.com/install.sh | sh`).
2.  **Pull Models:** Used `ollama pull <model_tag>` to acquire the necessary models.
    *   `llama4:17b-scout-16e-instruct-fp16` (Llama 4 Scout equivalent, FP16)
    *   `llama3.1:8b-instruct-fp16` (8B Instruct equivalent, FP16)
    *   `llama4:17b-scout-16e-instruct-q4_K_M` (Llama 4 Scout equivalent, Quantized)
    *   `llama3.1:8b-instruct-q4_K_M` (8B Instruct equivalent, Quantized)
3.  **Run Integration Script:** Execute `python baseline_mem0_ollama_demo.py`.
    *   This script configures `AsyncMemory`'s `LlmConfig` to use the `"ollama"` provider.
    *   It iterates through the specified model tags, initializing `AsyncMemory` for each.
    *   For each model, it performs basic `add` and `search` operations to confirm functionality.

**Result:**

The `baseline_mem0_ollama_demo.py` script ran successfully for all listed Ollama models. The output confirms that each model could be integrated with Mem0 and handle basic memory operations:

```text
✅  llama4:17b-scout-16e-instruct-fp16           hits→ ['Lives in Hyderabad', 'Allergic to peanuts']
✅  llama3.1:8b-instruct-fp16                   hits→ ['Lives in Hyderabad', 'Allergic to peanuts']
✅  llama4:17b-scout-16e-instruct-q4_K_M        hits→ ['Lives in Hyderabad', 'Allergic to peanuts']
✅  llama3.1:8b-instruct-q4_K_M                 hits→ ['Lives in Hyderabad', 'Allergic to peanuts']
```
*(Results indicate successful retrieval of relevant memories for each baseline model config)*

**Status:** Completed.

## Task 2: Baseline Benchmarking

### 2.1 Inference Metrics (Latency & Throughput)

**Objective:** Measure the average latency and token throughput for the baseline Ollama models using a standard set of prompts.

**Steps:**

1.  Executed the `baseline_inference_ollama_models.py` script against the baseline models served via Ollama.
    *   Used 100 prompts (`NUM_PROMPTS = 100`).
    *   Limited generation to `MAX_NEW_TOKENS = 64`.
    *   Recorded latency and calculated overall tokens per second.
2.  Results were saved to `./benchmarks/<model_tag_dir>/summary.json`.

**Results:**

The following table summarizes the baseline inference performance:

| Model           | Fine-Tuned | Precision | Format | Backend | LoRA Merged | Avg Latency (s) | Throughput (tok/s) |
| :-------------- | :--------: | :-------- | :----- | :------ | :---------: | :-------------: | :----------------: |
| LLaMA 4 Scout   | ❌         | FP16      | Base   | Ollama  |     N/A     |      45.66      |        1.2         |
| LLaMA 4 Scout   | ❌         | Q4_K_M    | Base   | Ollama  |     N/A     |      1.11       |        49.0        |
| LLaMA 3.1 8B    | ❌         | FP16      | Base   | Ollama  |     N/A     |      0.54       |       103.8        |
| LLaMA 3.1 8B    | ❌         | Q4_K_M    | Base   | Ollama  |     N/A     |      0.39       |       140.1        |

*(TBF = To Be Filled with results from corresponding `summary.json` files)*
*(Latency values rounded to 2 decimal places, Throughput to 1 decimal place for readability)*

**Status:** Completed.

### 2.2 Memory Retrieval Quality (Precision@5)

**Objective:** Measure the baseline retrieval precision using the Ollama models and the `nomic-embed-text` embedder.

**Steps:**

1.  Executed the `retrieval_precision_baseline_ollama.py` script.
    *   This script first clears any existing Qdrant/History data for the specified user/agent ID.
    *   It then uses `AsyncMemory` configured with a specific baseline Ollama `MODEL_TAG` and the `nomic-embed-text` embedder (`EMBED_TAG`).
    *   It adds 100 synthetic memories (`MEMORIES`) using `mem.add`.
    *   It executes 100 corresponding queries (`QUERIES`) using `mem.search` with `limit=5`.
    *   It calculates precision@5 by checking if the target memory ID (corresponding to the query) is present in the top 5 search results.
2.  The script was run for the baseline Ollama models (e.g., `llama3.1:8b-instruct-q4_K_M`).
3.  The summary results were saved to `./benchmarks/<model_tag_dir>/retrieval_summary.json`.

**Result:**

The baseline retrieval benchmark yielded a **Precision@5 of 0.95 (95%)**. This indicates that for 95% of the queries, the correct corresponding memory was retrieved within the top 5 results using the baseline Ollama LLM backends and the `nomic-embed-text` embedder.

*Result based on `benchmarks/llama3.1_8b-instruct-q4_K_M/retrieval_summary.json`*
```json
{
  "model": "llama3.1:8b-instruct-q4_K_M",
  "embedder": "nomic-embed-text",
  "timestamp": "2025-05-03T12:07:24.463006",
  "num_queries": 100,
  "precision_at_5": 0.95
}
```

**Status:** Completed.

## Task 3: Fine-Tuning Pipeline

**Objective:** Fine-tune both Llama 4 Scout and the Llama 3.1 8B instruct model using QLoRA/LoRA with Unsloth AI to improve performance on memory-centric tasks. Prepare a suitable dataset and export the fine-tuned weights.

**Steps & Results:**

1.  **Environment Setup:**
    *   Installed Unsloth AI 
    *   Configured for 4-bit quantization (`load_in_4bit=True`).

2.  **Dataset Preparation:**
    *   A dataset of approximately 5000 examples suitable for conversational memory tasks was assembled.
    *   ~3000 samples were custom-generated using LLMs.
    *   ~2000 samples were sourced from the Alpaca dataset.
    *   *(Dataset details/formatting can be found within the fine-tuning notebooks).*

3.  **Training Implementation & Execution:**
    *   **Llama 3.1 8B:**
        *   Successfully fine-tuned using Unsloth AI (QLoRA/LoRA).
        *   Hyperparameter tuning (e.g., rank, batch size) and VRAM monitoring were performed.
        *   *Script:* `fine-tuning/llama_31_8b_fine_tuning.ipynb`
    *   **Llama 4 Scout:**
        *   Initial attempts to fine-tune using Unsloth encountered errors.
        *   Switched approach to use standard Hugging Face `transformers` library with PEFT (Parameter-Efficient Fine-Tuning) for LoRA.
        *   Fine-tuning process completed successfully, showing potential accuracy improvements during training/evaluation steps within the notebook.
        *   *Script:* `fine-tuning/llama_4_scout.ipynb`

4.  **Export Weights:**
    *   **Llama 3.1 8B:** Successfully exported the fine-tuned model in multiple formats:
        *   GGUF (4-bit quantized)
        *   FP16 LoRA adapters
        *   4-bit merged weights (compatible with `transformers`) - Model ID: `cosmos98a/mem0-merged-llama3.1-8b-4bit`
    *   **Llama 4 Scout:** Exporting the fine-tuned weights **failed** due to GPU memory constraints encountered during the merging/saving process with the `transformers` + PEFT setup.

**Status:** Completed (with Llama 4 Scout export failure).

## Task 4: Post-Fine-Tuning Evaluation

### 4.1 Re-Benchmarking: Inference Metrics (Fine-Tuned)

**Objective:** Measure latency and throughput for the fine-tuned Llama 3.1 8B model variants and compare them to the baseline performance.

**Steps:**

1.  Executed inference benchmark scripts (similar to `baseline_inference_ollama_models.py`, adapted for different backends like vLLM, llama.cpp, Transformers) against the exported fine-tuned Llama 3.1 8B models.
    *   Used the same set of 100 prompts and `MAX_NEW_TOKENS = 64`.
    *   Recorded latency and calculated overall tokens per second.
2.  Results were saved to corresponding `./benchmarks/` subdirectories.
3.  *(Note: Llama 4 Scout re-benchmarking was skipped due to the inability to export fine-tuned weights in Task 3).*

**Results:**

The following table summarizes the inference performance for the *fine-tuned* Llama 3.1 8B model variants tested so far:

| Model        | Fine-Tuned | Precision | Format           | Backend      | LoRA Merged    | Avg Latency (s) | Throughput (tok/s) |
| :----------- | :--------: | :-------- | :--------------- | :----------- | :------------: | :-------------: | :----------------: |
| LLaMA 3.1 8B | ✅         | FP16      | Merged           | vLLM         |      ✅ Yes      |      0.49       |       130.0        |
| LLaMA 3.1 8B | ✅         | 4-bit     | GGUF             | llama.cpp    |      ✅ Yes      |      0.52       |       116.7        |
| LLaMA 3.1 8B | ✅         | 4-bit     | HuggingFace      | Transformers | ✅ Yes (merged) |      1.39       |        46.1        |
| LLaMA 3.1 8B | ✅         | 4-bit     | HF Runtime Merge | Transformers | ✅ Yes (manual) |      1.30       |        46.0        |

*(TBF = To Be Filled)*
*(Comparison to baseline: The fine-tuned FP16 model served via vLLM shows lower latency (0.49s vs 0.54s) and higher throughput (130.0 vs 103.8 tok/s) compared to the baseline FP16 model served via Ollama. Further comparisons pending results for other formats/backends.)*

**Analysis & Comparison:**

*   **Llama 3.1 8B FP16:** The fine-tuned FP16 model, when served via vLLM, demonstrated improved inference performance compared to the baseline FP16 model served via Ollama. Latency decreased from 0.54s to 0.49s (~9% reduction), and throughput increased from 103.8 tok/s to 130.0 tok/s (~25% improvement). It's important to note that the change in serving backend (Ollama vs. vLLM) likely contributes significantly to this difference alongside the fine-tuning.
*   **Llama 3.1 8B 4-bit:** Comparing the fine-tuned 4-bit models to the baseline Q4\_K\_M model served via Ollama is less direct due to different quantization formats (GGUF, HF 4-bit vs. Q4\_K\_M) and backends (llama.cpp, Transformers vs. Ollama).
    *   The fine-tuned GGUF model (via llama.cpp) showed slightly *higher* latency (0.52s vs. 0.39s) and *lower* throughput (116.7 vs. 140.1 tok/s) compared to the baseline Q4\_K\_M via Ollama.
    *   The fine-tuned HuggingFace Transformers variants (merged and runtime merge) exhibited significantly higher latency (~1.3-1.4s vs. 0.39s) and lower throughput (~46 tok/s vs. 140.1 tok/s) compared to the baseline Q4\_K\_M via Ollama.
    *   Among the fine-tuned 4-bit options, the GGUF/llama.cpp combination offered the best inference performance.
*   **Llama 4 Scout:** Re-benchmarking could not be performed for the fine-tuned Llama 4 Scout model. Although the model was successfully fine-tuned using Hugging Face Transformers + PEFT (Task 3), attempts to export the merged weights failed due to GPU memory limitations. Without the exported weights, the model could not be loaded into an inference backend for testing.

**Summary:** Fine-tuning provided inference performance gains for the FP16 model when combined with an optimized backend like vLLM. However, for the 4-bit models, the fine-tuning process combined with the available export formats and backends did not outperform the baseline Q4\_K\_M model served via Ollama in terms of latency or throughput. Backend choice and quantization method appear to be critical factors influencing performance.

**Status:** Completed.

### 4.2 Retrieval Improvement

**Objective:** Evaluate memory search precision using the fine-tuned Llama 3.1 8B model and compare it against the baseline.

**Steps:**

1.  Executed the `retrieval_precision_finetuned.py` script.
    *   This script configures `AsyncMemory` to use the fine-tuned LLM (`cosmos98a/mem0-merged-llama3.1-8b-4bit`) via the custom `llama31_8b_finetuned_4bit` provider (which uses the `llama31_8b_finetuned.py` implementation).
    *   It used the same `nomic-embed-text` Ollama embedder and Qdrant vector store setup as the baseline retrieval test.
    *   The same 100 synthetic memory/query pairs were used.
2.  Precision@5 was calculated based on whether the correct memory ID was retrieved in the top 5 results for each query.
3.  *(Note: Llama 4 Scout retrieval evaluation was skipped due to the inability to export fine-tuned weights in Task 3).*

**Result:**

The retrieval benchmark using the fine-tuned Llama 3.1 8B model yielded a **Precision@5 of 0.80 (80%)**.

*Result based on `benchmarks/finetuned_retrieval_summary.json` (generated by `retrieval_precision_finetuned.py`):*
```json
{
  "model": "cosmos98a/mem0-merged-llama3.1-8b-4bit",
  "embedder": "nomic-embed-text",
  "timestamp": "2025-05-05T16:28:54.777586",
  "num_total_queries": 100,
  "num_processed_queries": 80,
  "num_hits": 64,
  "precision_at_5": 0.8
}
```

**Analysis & Comparison:**

*   The retrieval precision **decreased** significantly from the baseline score of **0.95 (95%)** down to **0.80 (80%)** after fine-tuning the LLM backend.
*   As observed during the script execution (errors logged during `mem.add`), the fine-tuned model struggled to consistently produce the structured JSON output required by `mem0`'s internal memory processing and consolidation logic.
*   This resulted in 20% of the memories failing to be added correctly (`num_processed_queries: 80` out of 100).
*   Furthermore, the likely suboptimal quality or duplication of the memories that *were* added (due to poor consolidation) probably contributed to the lower retrieval success rate for the processed queries (64 hits / 80 processed = 80% precision).
*   This indicates that while fine-tuning might improve the model on its training task, it negatively impacted its ability to follow the specific structured output formats needed by `mem0`'s inference-based memory addition process in this instance.

**Status:** Completed (with Llama 4 Scout skipped).

## Task 5: Mem0 Integration & Demo

**Objective:** Demonstrate the end-to-end integration of the fine-tuned Llama 3.1 8B model with Mem0 via a sample chat interface.

**Steps & Results:**

1.  **Backend Configuration:**
    *   The Mem0 SDK was configured within the sample application to use the fine-tuned Llama 3.1 8B model (`cosmos98a/mem0-merged-llama3.1-8b-4bit`) loaded via the custom `llama31_8b_finetuned_4bit` provider.
    *   Configuration also included the Ollama `nomic-embed-text` embedder and the Qdrant vector store.

2.  **Sample Client Implementation:**
    *   A Gradio web interface was developed (`app.py`).
    *   This application allows a user to enter their name (to create a user ID) and then engage in a chat conversation.
    *   On each user message:
        *   The application retrieves relevant past memories for the user using `mem.search`.
        *   It constructs a prompt containing the user message, recent chat history, and retrieved memories.
        *   It calls the fine-tuned LLM (instantiated separately via `LlmFactory`) to generate a response based on the prompt.
        *   It adds the user's input and the LLM's response to Mem0 using `mem.add`.
        *   The conversation is displayed in the Gradio chat interface.
    *   *Script:* `app.py`

3.  **Demo Walkthrough:**
    *   A video demonstration (3-5 minutes) showcasing the fine-tuning process (briefly), benchmarking results, and a live walkthrough of the Gradio chat application (`app.py`) interacting with the fine-tuned model and Mem0 is provided as part of the deliverables.

**Status:** Implementation completed. Demo video is a separate deliverable.

---

This concludes the documentation of the technical implementation steps for the Mem0 Small Model Integration Challenge.
