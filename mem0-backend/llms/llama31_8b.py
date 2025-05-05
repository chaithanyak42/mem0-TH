import logging
from typing import Dict, List, Optional

# Import transformers and torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the base class
from mem0.llms.base import LLMBase
from mem0.configs.llms.base import BaseLlmConfig # Needed for type hint

logger = logging.getLogger(__name__)

class Llama31_8B_InstructLLM(LLMBase):
    """
    Integration for Meta Llama 3.1 8B Instruct models.
    """
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        """
        Initializes the Llama31_8B_InstructLLM provider.
        Args:
            config: Configuration object for the LLM. Expects 'model_path' in config.config dict.
        """
        super().__init__(config)

        model_path = self.config.config.get("model_path")
        if not model_path:
            raise ValueError("Llama31_8B_InstructLLM config must include 'model_path'")

        logger.info(f"Llama31_8B_InstructLLM: Initializing model from {model_path}...")

        try:
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # Use bfloat16 if supported, otherwise float16. Requires Ampere+ GPU for bf16.
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            # elif torch.backends.mps.is_available(): # Uncomment if on Apple Silicon Mac with MPS
            #    self.device = torch.device("mps")
            #    compute_dtype = torch.float16 # MPS typically uses float16
            else:
                self.device = torch.device("cpu")
                compute_dtype = torch.float32 # Use float32 on CPU
                logger.warning("Llama31_8B_InstructLLM: No GPU detected, falling back to CPU. Performance will be slow.")

            logger.info(f"Llama31_8B_InstructLLM: Using device: {self.device} with dtype: {compute_dtype}")

            # Load Tokenizer
            # Make sure trust_remote_code=True is acceptable if needed, but Llama 3.1 shouldn't require it.
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("Llama31_8B_InstructLLM: Tokenizer loaded.")

            # Load Model
            # Consider adding load_in_8bit=True or load_in_4bit=True (with bitsandbytes)
            # if GPU memory is still an issue, even for 8B.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=compute_dtype,
                # device_map="auto" # Use if loading across multiple GPUs or offloading
            ).to(self.device) # Move model explicitly to the chosen device

            self.model.eval() # Set model to evaluation mode
            logger.info(f"Llama31_8B_InstructLLM: Model loaded successfully to {self.device}.")

        except ImportError as ie:
             logger.error(f"Llama31_8B_InstructLLM: Import error during loading. Did you install `transformers` and `torch`? Error: {ie}")
             raise ie
        except Exception as e:
            logger.error(f"Llama31_8B_InstructLLM: Error loading model: {e}")
            raise e


    def generate_response(self, messages: List[Dict], tools: Optional[List[Dict]] = None, tool_choice: str = "auto", **kwargs):
        """
        Generate a response based on the given messages using the Llama 3.1 8B model.
        Note: Baseline implementation does not effectively handle tools/tool_choice.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional keyword arguments passed from the caller.

        Returns:
            str: The raw string response from the LLM.
        """
        logger.debug(f"Llama31_8B_InstructLLM: generate_response called with {len(messages)} messages.")

        # 1. Format Prompt using Llama 3.1's chat template
        try:
            # The apply_chat_template function uses the tokenizer's configured chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # Adds the prompt for the assistant's turn
            )
            logger.debug(f"Llama31_8B_InstructLLM: Formatted Prompt (first 500 chars):\n{formatted_prompt[:500]}...")
        except Exception as e:
            logger.error(f"Llama31_8B_InstructLLM: Error formatting prompt: {e}")
            return f"Error formatting prompt: {e}" # Return error string

        # 2. Tokenize Input
        try:
            # Ensure tokenizer padding side is correct if batching (not needed here)
            # self.tokenizer.padding_side = "left"
            model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to(self.device)
        except Exception as e:
            logger.error(f"Llama31_8B_InstructLLM: Error tokenizing input: {e}")
            return f"Error tokenizing input: {e}" # Return error string

        # 3. Generate Output
        logger.debug(f"Llama31_8B_InstructLLM: Generating response...")
        try:
            # Use torch.no_grad() for inference to save memory and prevent gradient calculations
            with torch.no_grad():
                # Common generation parameters - adjust as needed
                # max_new_tokens from kwargs or default; consider Mem0 expectations
                generation_kwargs = {
                    "max_new_tokens": kwargs.get("max_tokens", 512),
                    "eos_token_id": [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")], # Llama3 specific EoS
                    "do_sample": kwargs.get("temperature", 0.7) > 0, # Sample only if temperature > 0
                    "temperature": float(kwargs.get("temperature", 0.7)),
                    "top_p": float(kwargs.get("top_p", 0.9)),
                }
                logger.debug(f"Llama31_8B_InstructLLM: Generation kwargs: {generation_kwargs}")

                generated_ids = self.model.generate(
                    **model_inputs,
                    **generation_kwargs
                )
            logger.debug(f"Llama31_8B_InstructLLM: Generation complete.")
        except Exception as e:
            logger.error(f"Llama31_8B_InstructLLM: Error during model generation: {e}")
            return f"Error during generation: {e}" # Return error string

        # 4. Decode Output
        try:
            # Decode only the newly generated tokens, excluding the input prompt
            response_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
            response_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
            logger.debug(f"Llama31_8B_InstructLLM: Decoded response (first 500 chars): {response_text[:500]}...")
        except Exception as e:
            logger.error(f"Llama31_8B_InstructLLM: Error decoding output: {e}")
            return f"Error decoding output: {e}" # Return error string

        # 5. Return raw string content
        return response_text