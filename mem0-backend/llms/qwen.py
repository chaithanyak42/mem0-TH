from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# Import transformers and torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the base class
from mem0.llms.base import LLMBase
from mem0.configs.llms.base import BaseLlmConfig # Needed for type hint


class QwenLLM(LLMBase):
    """
    Integration for Qwen language models.
    """
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        """
        Initializes the QwenLLM provider.
        Args:
            config: Configuration object for the LLM. Expects 'model_path' in config.config dict.
        """
        super().__init__(config)
        
        model_path = self.config.config.get("model_path")
        if not model_path:
            raise ValueError("QwenLLM config must include 'model_path'")

        print(f"QwenLLM Initializing: Loading model from {model_path}...")

        try:
            # Determine device (prioritize GPU if available, fallback to CPU)
            # Note: For baseline test, forcing CPU might be safer if GPU RAM is limited.
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "cuda" # Force GPU for initial baseline integration test
            print(f"QwenLLM: Using device: {device}")

            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("QwenLLM: Tokenizer loaded.")

            # Load Model
            # Using 'auto' dtype and device_map='auto' might require accelerate
            # Loading directly to CPU might be simpler for baseline.
            # Consider adding load_in_8bit=True or load_in_4bit=True if memory is an issue,
            # but this requires bitsandbytes. Let's try standard loading first.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device=="cuda" else torch.float32, # Use float32 on CPU
                # device_map="auto" # Requires accelerate, might complicate baseline
            ).to(device) # Move model to the chosen device
            
            self.model.eval() # Set model to evaluation mode
            print(f"QwenLLM: Model loaded successfully to {device}.")

        except Exception as e:
            print(f"QwenLLM Error loading model: {e}")
            raise e


    def generate_response(self, messages: List[Dict], tools: Optional[List[Dict]] = None, tool_choice: str = "auto", **kwargs):
        """
        Generate a response based on the given messages using the Qwen model.
        Note: Baseline implementation does not handle tools/tool_choice effectively.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional keyword arguments passed from the caller.

        Returns:
            dict: The generated response in the format {"role": "assistant", "content": ...}
                  (Tool calls not handled in baseline).
        """
        print(f"QwenLLM generate_response called with {len(messages)} messages.")

        # 1. Format Prompt using Qwen's chat template
        try:
            # Ensure messages are in the correct format if needed (Mem0 usually provides this)
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"QwenLLM: Formatted Prompt:\n{formatted_prompt[:500]}...") # Log beginning of prompt
        except Exception as e:
            print(f"QwenLLM Error formatting prompt: {e}")
            # Fallback or re-raise
            return {"role": "assistant", "content": f"Error formatting prompt: {e}"}

        # 2. Tokenize Input
        try:
            model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)
        except Exception as e:
            print(f"QwenLLM Error tokenizing input: {e}")
            return {"role": "assistant", "content": f"Error tokenizing input: {e}"}

        # 3. Generate Output
        print(f"QwenLLM: Generating response...")
        try:
            # Use torch.no_grad() for inference to save memory
            with torch.no_grad():
                # Simple generation parameters - adjust as needed
                # max_new_tokens might need tuning depending on Mem0's expectations
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=kwargs.get("max_tokens", 256), # Use passed max_tokens or default
                    # Add other common params if needed: temperature, top_p, etc.
                    # Note: Tool usage would require more complex generation logic/kwargs
                )
            print(f"QwenLLM: Generation complete.")
        except Exception as e:
            print(f"QwenLLM Error during model generation: {e}")
            return {"role": "assistant", "content": f"Error during generation: {e}"}

        # 4. Decode Output
        try:
            # Decode only the newly generated tokens, excluding the input prompt
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Print the FULL raw response for debugging
            print(f"--- QwenLLM Raw Response Start ---")
            print(response_text)
            print(f"--- QwenLLM Raw Response End ---")
        except Exception as e:
            print(f"QwenLLM Error decoding output: {e}")
            return {"role": "assistant", "content": f"Error decoding output: {e}"}

        # 5. Return raw string content, as expected by the calling Mem0 code
        return response_text
