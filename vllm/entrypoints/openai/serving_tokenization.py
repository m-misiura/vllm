# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import re
from functools import lru_cache
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import jinja2
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (DetokenizeRequest,
                                              DetokenizeResponse,
                                              ErrorResponse,
                                              TokenizeChatRequest,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                              TokenizerInfoResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import (AnyTokenizer, decode_tokens,
                                               encode_tokens)
from vllm.transformers_utils.tokenizers import MistralTokenizer

logger = init_logger(__name__)


class OpenAIServingTokenization(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

    async def create_tokenize(
        self,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> Union[TokenizeResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"tokn-{self._base_request_id(raw_request)}"

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if isinstance(request, TokenizeChatRequest):
                tool_dicts = (None if request.tools is None else
                              [tool.model_dump() for tool in request.tools])
                (
                    _,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    tool_dicts=tool_dicts,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.
                    chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    chat_template_kwargs=request.chat_template_kwargs,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                (request_prompts,
                 engine_prompts) = await self._preprocess_completion(
                     request,
                     tokenizer,
                     request.prompt,
                     add_special_tokens=request.add_special_tokens,
                 )
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        input_ids: list[int] = []
        for i, engine_prompt in enumerate(engine_prompts):
            self._log_inputs(request_id,
                             request_prompts[i],
                             params=None,
                             lora_request=lora_request,
                             prompt_adapter_request=prompt_adapter_request)

            # Silently ignore prompt adapter since it does not affect
            # tokenization (Unlike in Embeddings API where an error is raised)
            if isinstance(engine_prompt,
                          dict) and "prompt_token_ids" in engine_prompt:
                input_ids.extend(engine_prompt["prompt_token_ids"])

        token_strs = None
        if request.return_token_strs:
            token_strs = tokenizer.convert_ids_to_tokens(input_ids)

        return TokenizeResponse(tokens=input_ids,
                                token_strs=token_strs,
                                count=len(input_ids),
                                max_model_len=self.max_model_len)

    async def create_detokenize(
        self,
        request: DetokenizeRequest,
        raw_request: Request,
    ) -> Union[DetokenizeResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"tokn-{self._base_request_id(raw_request)}"

        (
            lora_request,
            prompt_adapter_request,
        ) = self._maybe_get_adapters(request)

        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        self._log_inputs(request_id,
                         request.tokens,
                         params=None,
                         lora_request=lora_request,
                         prompt_adapter_request=prompt_adapter_request)

        # Silently ignore prompt adapter since it does not affect tokenization
        # (Unlike in Embeddings API where an error is raised)

        prompt_input = await self._tokenize_prompt_input_async(
            request,
            tokenizer,
            request.tokens,
        )
        input_text = prompt_input["prompt"]

        return DetokenizeResponse(prompt=input_text)

    async def get_tokenizer_info(
            self) -> Union[TokenizerInfoResponse, ErrorResponse]:
        """Get comprehensive tokenizer information."""
        try:
            tokenizer = await self.engine_client.get_tokenizer()
            info = TokenizerInfo(tokenizer, self.model_config,
                                 self.chat_template).to_dict()
            return TokenizerInfoResponse(**info)
        except Exception as e:
            return self.create_error_response(
                f"Failed to get tokenizer info: {str(e)}")


class TokenizerInfo:

    def __init__(self, tokenizer: AnyTokenizer, model_config: ModelConfig,
                 chat_template: Optional[str]):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.chat_template = chat_template

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            **self._get_core_info(),
            **self._get_model_info(),
            **self._get_special_tokens(),
            **self._get_tokenizer_attributes(),
            **self._get_chat_template_info(),
        }

    def _get_core_info(self) -> Dict[str, Any]:
        """Get core tokenizer information."""
        vocab_size = getattr(self.tokenizer, 'vocab_size', None)
        tokenizer_type = type(self.tokenizer).__name__

        return {
            "tokenizer_type": tokenizer_type,
            "vocab_size": vocab_size,
            "tokenizer_backend": self._detect_backend(tokenizer_type),
            "is_cached": "Cached" in tokenizer_type,
            "is_fast": getattr(self.tokenizer, 'is_fast', False),
            "max_token_id": vocab_size - 1 if vocab_size else None,
        }

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        if not self.model_config:
            return {}

        return {
            "model_name":
            self.model_config.model,
            "tokenizer_name":
            getattr(self.model_config, 'tokenizer', None),
            "tokenizer_mode":
            getattr(self.model_config, 'tokenizer_mode', 'auto'),
            "trust_remote_code":
            getattr(self.model_config, 'trust_remote_code', False),
            "tokenizer_revision":
            getattr(self.model_config, 'tokenizer_revision', None),
            "is_gguf":
            getattr(self.model_config, 'gguf_file', None) is not None,
            "gguf_file":
            getattr(self.model_config, 'gguf_file', None),
        }

    def _get_special_tokens(self) -> Dict[str, Any]:
        """Get all special tokens using the official HuggingFace method."""
        special_tokens = {}

        # Use the official special_tokens_map property - this is the canonical way
        if hasattr(self.tokenizer, 'special_tokens_map'):
            for key, token_str in self.tokenizer.special_tokens_map.items():
                if key == "additional_special_tokens" and isinstance(
                        token_str, list):
                    # Handle additional special tokens as a clean mapping
                    for tok_str in token_str:
                        try:
                            token_id = self.tokenizer.convert_tokens_to_ids(
                                tok_str)
                            if token_id >= 0:  # Valid token ID
                                special_tokens[tok_str] = token_id
                        except:
                            pass
                else:
                    # Handle regular special tokens
                    if token_str:  # Make sure it's not None or empty
                        try:
                            token_id = getattr(self.tokenizer, f"{key}_id",
                                               None)
                            if token_id is not None and token_id >= 0:
                                special_tokens[token_str] = token_id
                        except:
                            pass

        return {"special_tokens": special_tokens}

    def _get_tokenizer_attributes(self) -> Dict[str, Any]:
        """Get tokenizer attributes and capabilities."""
        return {
            # HuggingFace attributes
            "model_max_length":
            getattr(self.tokenizer, 'model_max_length', None),
            "truncation_side":
            getattr(self.tokenizer, 'truncation_side', 'right'),
            "padding_side":
            getattr(self.tokenizer, 'padding_side', 'right'),
            "clean_up_tokenization_spaces":
            getattr(self.tokenizer, 'clean_up_tokenization_spaces', True),
            # Capabilities
            "supports_encoding":
            hasattr(self.tokenizer, 'encode') or callable(self.tokenizer),
            "supports_decoding":
            hasattr(self.tokenizer, 'decode'),
        }

    def _get_chat_template_info(self) -> Dict[str, Any]:
        """Get chat template information."""
        template, source = self._find_chat_template()

        if not template:
            return {
                "has_chat_template": False,
                "chat_template": None,
                "chat_template_source": "none",
                "supports_system_message": False,
                "supports_tools": False,
            }

        return {
            "has_chat_template":
            True,
            "chat_template":
            template,
            "chat_template_source":
            source,
            "supports_system_message":
            "system" in template.lower(),
            "supports_tools":
            any(word in template.lower() for word in ["tool", "function"]),
        }

    def _detect_backend(self, tokenizer_type: str) -> str:
        """Detect tokenizer backend from type name."""
        if isinstance(self.tokenizer, MistralTokenizer):
            return "mistral"
        elif "Fast" in tokenizer_type or getattr(self.tokenizer, 'is_fast',
                                                 False):
            return "huggingface_fast"
        elif "SentencePiece" in tokenizer_type:
            return "sentencepiece"
        elif "Tiktoken" in tokenizer_type:
            return "tiktoken"
        elif "Cached" in tokenizer_type:
            return "cached"
        else:
            return "huggingface_slow"

    def _safe_token_to_string(self, token) -> Optional[str]:
        """Convert token to string safely."""
        if hasattr(token, 'content'):
            return token.content
        elif token:
            return str(token)
        return None

    def _safe_get_token_id(self, token_str: str) -> Optional[int]:
        """Get token ID safely."""
        try:
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            return token_id if token_id >= 0 else None
        except:
            return None

    def _safe_get_token_id_by_attr(self, attr: str) -> Optional[int]:
        """Get token ID by attribute name safely."""
        try:
            token_id = getattr(self.tokenizer, f"{attr}_id", None)
            return token_id if isinstance(token_id,
                                          int) and token_id >= 0 else None
        except:
            return None

    def _find_chat_template(self) -> Tuple[Optional[str], str]:
        """Find chat template from various sources."""
        # Check tokenizer
        if hasattr(self.tokenizer,
                   'chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.chat_template, "tokenizer"

        # Check underlying tokenizer
        if hasattr(self.tokenizer, 'tokenizer'):
            underlying = self.tokenizer.tokenizer
            if hasattr(underlying,
                       'chat_template') and underlying.chat_template:
                return underlying.chat_template, "underlying"

        # Check config
        if self.chat_template:
            return self.chat_template, "config"

        return None, "none"