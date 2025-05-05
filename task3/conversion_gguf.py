# conversion_gguf.py
"""
Converts a fine-tuned Hugging Face model (Base + LoRA) to GGUF format (Q4_K_M).

**ASSUMES:**
- `llama.cpp` repository is cloned in the current directory (`./llama.cpp`).
- `llama.cpp` Python requirements are installed (`uv pip install -r llama.cpp/requirements.txt`).
- `llama.cpp/convert_hf_to_gguf.py` script exists.
- The `quantize` tool has been compiled (`cd llama.cpp && make quantize && cd ..`).

Steps:
1. Loads base model + LoRA adapters in high precision (fp16/bf16).
2. Merges LoRA adapters into the base model.
3. Saves the merged high-precision model to a temporary directory.
4. Runs llama.cpp's convert_hf_to_gguf.py to create an intermediate FP16 GGUF.
5. Runs the compiled llama.cpp `quantize` tool to create the final Q4_K_M GGUF.
6. Cleans up the temporary directory and intermediate GGUF.

Requirements (Install these in your main environment):
- Python packages: torch, transformers, peft
- System tools: git, make, C++ compiler (build-essential on Debian/Ubuntu)

Run: python conversion_gguf.py
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
BASE_MODEL_ID = "unsloth/Meta-Llama-3.1-8B"  # Base model
LORA_REPO_ID = "cosmos98a/mem0-finetuned-llama3.1-8b-4b" # Your LoRA adapters repo
QUANTIZATION_TYPE = "Q4_K_M"  # Target GGUF quantization
INTERMEDIATE_PRECISION = "f16" # Precision for the intermediate GGUF
OUTPUT_GGUF_FILENAME = f"mem0-finetuned-llama31-8b-{QUANTIZATION_TYPE}.gguf"
LLAMA_CPP_DIR = Path("llama.cpp")  # Assumed to exist here
HF_TOKEN = None  # Optional: Set if your LoRA repo is private

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(command: list[str], cwd: Path | None = None, check: bool = True) -> None:
    """Runs a shell command."""
    logger.info(f"Running command: {' '.join(command)}" + (f" in {cwd}" if cwd else ""))
    process = subprocess.run(
        command,
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout:
        logger.info(f"Stdout:\n{process.stdout.strip()}")
    if process.stderr:
        logger.warning(f"Stderr:\n{process.stderr.strip()}") # Log stderr even on success
    if check and process.returncode != 0:
        logger.error(f"Command failed with return code {process.returncode}")
        raise subprocess.CalledProcessError(process.returncode, command, output=process.stdout, stderr=process.stderr)
    logger.info("Command finished successfully.")


def save_merged_model(temp_dir_path: Path) -> None:
    """Loads base+LoRA, merges, and saves to a temporary directory."""
    logger.info("--- Step 1: Loading and Merging Hugging Face Model ---")
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    logger.info(f"Using precision: {compute_dtype}")
    logger.info(f"Loading base model: {BASE_MODEL_ID} in {compute_dtype} (on CPU initially)")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    logger.info(f"Loading LoRA adapters from: {LORA_REPO_ID}")
    peft_model = PeftModel.from_pretrained(base_model, LORA_REPO_ID, token=HF_TOKEN)
    logger.info("Merging LoRA adapters into base model...")
    merged_model = peft_model.merge_and_unload()
    logger.info("Merge complete.")
    logger.info(f"Saving merged model and tokenizer to temporary directory: {temp_dir_path}")
    merged_model.save_pretrained(str(temp_dir_path))
    tokenizer.save_pretrained(str(temp_dir_path))
    logger.info("Merged model saved successfully.")
    logger.info("Cleaning up models from memory...")
    del merged_model, peft_model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cleaned.")


def convert_hf_to_f16_gguf(merged_model_path: Path, intermediate_gguf_path: Path) -> None:
    """Runs the llama.cpp conversion script to create an FP16 GGUF."""
    logger.info("--- Step 2: Converting Merged HF Model to FP16 GGUF ---")
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
         logger.error(f"llama.cpp conversion script not found at {convert_script}. Ensure llama.cpp is cloned correctly.")
         raise FileNotFoundError("llama.cpp conversion script 'convert_hf_to_gguf.py' not found.")

    logger.info(f"Using conversion script: {convert_script}")
    logger.info(f"Converting model at {merged_model_path} to {intermediate_gguf_path} (FP16)")

    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_model_path),
        "--outfile",
        str(intermediate_gguf_path),
        "--outtype",
        INTERMEDIATE_PRECISION, # Output f16
    ]
    run_command(cmd)
    logger.info(f"Intermediate FP16 GGUF conversion successful. Output at: {intermediate_gguf_path}")


def quantize_gguf(intermediate_gguf_path: Path, output_gguf_path: Path) -> None:
    """Runs the compiled llama.cpp quantize tool."""
    logger.info(f"--- Step 3: Quantizing FP16 GGUF to {QUANTIZATION_TYPE} ---")
    # Point to the correct executable name found via ls
    quantize_tool = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize" # <-- CORRECT NAME
    if not quantize_tool.exists():
        logger.error(f"Compiled quantize tool not found at {quantize_tool}.")
        logger.error("Please compile it first using the CMake build steps (see script comments or README).")
        raise FileNotFoundError("llama.cpp quantize tool not found in build/bin/.")

    logger.info(f"Using quantize tool: {quantize_tool}")
    logger.info(f"Quantizing {intermediate_gguf_path} to {output_gguf_path} ({QUANTIZATION_TYPE})")

    cmd = [
        str(quantize_tool),
        str(intermediate_gguf_path),
        str(output_gguf_path),
        QUANTIZATION_TYPE, # The target quantization type
    ]
    run_command(cmd)
    logger.info(f"Final {QUANTIZATION_TYPE} GGUF creation successful. Output at: {output_gguf_path}")


def main():
    """Orchestrates the conversion process."""
    # --- Pre-checks ---
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    # Point to the correct executable name
    quantize_tool = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize" # <-- CORRECT NAME
    if not LLAMA_CPP_DIR.is_dir():
        print(f"❌ Error: Cannot find `./llama.cpp` directory.")
        print(f"Please run: git clone https://github.com/ggerganov/llama.cpp.git")
        sys.exit(1)
    if not convert_script.exists():
        print(f"❌ Error: Cannot find the `convert_hf_to_gguf.py` script in ./llama.cpp.")
        print(f"Please ensure the clone was successful.")
        sys.exit(1)
    if not quantize_tool.exists():
         print(f"❌ Error: Cannot find the compiled `llama-quantize` tool in ./llama.cpp/build/bin/.")
         print(f"Please run the CMake build steps first:")
         print(f"  1. cd llama.cpp")
         print(f"  2. mkdir build && cd build")
         print(f"  3. cmake ..")
         print(f"  4. cmake --build . --config Release")
         print(f"  5. cd ../..")
         print(f"  (You might need: sudo apt update && sudo apt install build-essential cmake)")
         sys.exit(1)
    # --- End Pre-checks ---

    output_gguf_path = Path.cwd() / OUTPUT_GGUF_FILENAME
    intermediate_gguf_path = Path.cwd() / f"intermediate_{INTERMEDIATE_PRECISION}.gguf" # Temp file

    if output_gguf_path.exists():
        logger.warning(f"Output file {output_gguf_path} already exists. Skipping conversion.")
        print(f"✅ GGUF file already exists: {output_gguf_path}")
        return

    # Use a temporary directory for the intermediate merged HF model
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        logger.info(f"Created temporary directory for HF model: {temp_dir_path}")

        try:
            # 1. Load, merge, save HF model (in high precision)
            save_merged_model(temp_dir_path)

            # 2. Convert HF model to intermediate FP16 GGUF
            convert_hf_to_f16_gguf(temp_dir_path, intermediate_gguf_path)

            # 3. Quantize FP16 GGUF to final Q4_K_M GGUF
            quantize_gguf(intermediate_gguf_path, output_gguf_path)

            print(f"\n🎉 Successfully created GGUF file: {output_gguf_path}")
            print(f"\nNext steps:")
            print(f"1. Create an Ollama Modelfile (e.g., 'Modelfile') pointing to this GGUF:")
            print(f"   FROM ./{output_gguf_path.name}")
            print(f"   # Add TEMPLATE, PARAMETER etc. as needed")
            print(f"2. Register with Ollama: ollama create my-finetuned-llama-{QUANTIZATION_TYPE} -f Modelfile")
            print(f"3. Run Ollama: ollama run my-finetuned-llama-{QUANTIZATION_TYPE}")

        except Exception as e:
            logger.error(f"An error occurred during the conversion process: {e}", exc_info=True)
            print(f"\n❌ Conversion failed. See logs above for details.")
            # Clean up intermediate file even on error if it exists
            if intermediate_gguf_path.exists():
                logger.info(f"Cleaning up intermediate GGUF file: {intermediate_gguf_path}")
                intermediate_gguf_path.unlink()
        finally:
            logger.info(f"Temporary directory {temp_dir_path} cleaned up.")
            # Clean up intermediate file on success if it exists
            if 'e' not in locals() and intermediate_gguf_path.exists(): # Check if exception occurred
                 logger.info(f"Cleaning up intermediate GGUF file: {intermediate_gguf_path}")
                 intermediate_gguf_path.unlink()


if __name__ == "__main__":
    main()