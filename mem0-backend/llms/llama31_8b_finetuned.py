import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import the base LLM class and config
from mem0.llms.base import LLMBase
from mem0.configs.llms.base import BaseLlmConfig # Needed for type hint in __init__

logger = logging.getLogger(__name__)


class Llama31_8B_Finetuned_4bit_LLM(LLMBase):
    """
    LLM wrapper for the specific fine-tuned 4-bit Llama 3.1 8B model
    loaded from Hugging Face using transformers.
    Requires `bitsandbytes`, `accelerate`, and `torch`.
    """

    def __init__(self, config: BaseLlmConfig):
        """
        Initializes the Llama31_8B_Finetuned_4bit_LLM.

        Args:
            config: BaseLlmConfig object. Expects 'model_id' key within config.config dictionary.
        """
        super().__init__(config) # Pass the whole config object up

        # Extract model_id from the nested config dictionary
        self.model_id = self.config.config.get("model_id")

        if not self.model_id:
            raise ValueError(
                "Llama31_8B_Finetuned_4bit_LLM config dict must include 'model_id'"
            )

        logger.info(
            f"Llama31_8B_Finetuned_4bit_LLM: Initializing model {self.model_id}..."
        )

        try:
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, # Recommended for Ampere+
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                logger.info(f"Using CUDA device with compute dtype: {compute_dtype}")
            else:
                self.device = torch.device("cpu")
                compute_dtype = torch.float32
                logger.warning(
                    "CUDA not available, falling back to CPU. 4-bit quantization may not work as expected or be efficient."
                )

            # Load Tokenizer
            logger.info(f"Loading tokenizer for {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info("Set tokenizer pad_token_id to eos_token_id.")
            logger.info("Tokenizer loaded.")

            # Load Model with quantization
            logger.info(f"Loading 4-bit quantized model {self.model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype, # Load weights in compute dtype for bnb
                device_map="auto",  # Automatically distribute across available GPUs (requires accelerate)
                # trust_remote_code=True # May be needed depending on the fine-tuned model structure
            )

            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
                logger.info("Set model.config.pad_token_id to tokenizer's eos_token_id.")

            self.model.eval()  # Set model to evaluation mode
            logger.info(
                f"Llama31_8B_Finetuned_4bit_LLM: Model {self.model_id} loaded successfully."
            )

        except ImportError as ie:
            logger.error(
                f"Import error during loading. Did you install 'transformers', 'torch', 'bitsandbytes', 'accelerate'? Error: {ie}"
            )
            raise ie
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}", exc_info=True)
            raise e

    def _generate(self, prompt_text: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> Tuple[str, int]:
        """
        Internal generation logic shared by public methods.

        Args:
            prompt_text: The formatted input prompt string.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling p.

        Returns:
            Tuple containing (generated response text, number of generated tokens).
        """
        logger.debug(f"Generating response for prompt (start): {prompt_text[:200]}...")

        try:
            # Tokenize the input prompt
            model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(
                self.model.device # Ensure inputs are on the same device as the model slices
            )
            input_token_len = model_inputs.input_ids.shape[-1]

            # Prepare generation arguments
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": float(temperature),
                "top_p": float(top_p),
                # Llama 3 specific EoS tokens for more reliable stopping
                "eos_token_id": [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                "pad_token_id": self.tokenizer.pad_token_id
            }
            logger.debug(f"Generation kwargs: {generation_kwargs}")

            # Use torch.inference_mode() for efficiency
            with torch.inference_mode():
                t_start = time.perf_counter()
                # Generate response IDs
                generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
                t_end = time.perf_counter()
                logger.debug(f"Raw generation took {t_end - t_start:.4f}s")

            # Decode the newly generated tokens (excluding the input prompt)
            response_ids = generated_ids[:, input_token_len:]
            response_text = self.tokenizer.batch_decode(
                response_ids, skip_special_tokens=True
            )[0]
            generated_token_count = response_ids.shape[-1] # Count tokens in the response part

            logger.debug(
                f"Generated {generated_token_count} tokens. Response (start): {response_text[:200]}..."
            )
            return response_text, generated_token_count

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            # Return error message and 0 tokens in case of failure
            return f"Error during generation: {e}", 0

    def generate_response(self, messages: List[Dict], tools: Optional[List[Dict]] = None, tool_choice: str = "auto", **kwargs) -> str:
        """
        Generate a response based on the given messages (implements LLMBase interface).
        Tools are ignored in this implementation.

        Args:
            messages (list): List of message dicts [{'role': 'user', 'content': '...'}].
            tools (list, optional): Ignored.
            tool_choice (str, optional): Ignored.
            **kwargs: Can include 'max_tokens', 'temperature', 'top_p' passed from Mem0 core.

        Returns:
            str: The generated response text.
        """
        try:
            # Apply the chat template defined in the tokenizer for Llama 3 instruct format
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # Crucial for assistant generation
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}. Falling back.")
            # Basic fallback if template fails
            formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

        # Extract generation parameters from kwargs or use defaults
        max_new_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.1) # Low temp for consistent benchmarks
        top_p = kwargs.get("top_p", 0.9)

        # Call internal generate method and return only the text
        response_text, _ = self._generate(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response_text

    def generate_response_with_stats(
        self, messages: List[Dict], max_new_tokens: int = 512, temperature: float = 0.1, top_p: float = 0.9
    ) -> Tuple[str, int]:
        """
        Generates a response and returns it along with the number of generated tokens.
        Used specifically for benchmarking.

        Args:
            messages (list): List of message dicts [{'role': 'user', 'content': '...'}].
            max_new_tokens (int): Max tokens for the generated response.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling p.

        Returns:
            Tuple[str, int]: The generated response text and the count of generated tokens.
        """
        try:
            # Apply the chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}. Falling back.")
            formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

        # Call internal generate method with specified parameters
        return self._generate(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
