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

    async def get_tokenizer_info(self, include_chat_template: bool = False) -> Union[TokenizerInfoResponse, ErrorResponse]:
        """Get comprehensive tokenizer information."""
        try:
            tokenizer = await self.engine_client.get_tokenizer()
            chat_template = None
            if include_chat_template:
                chat_template = getattr(tokenizer, 'chat_template', None) or self.chat_template
            info = TokenizerInfo(tokenizer, self.model_config, chat_template).to_dict()
            return TokenizerInfoResponse(**info)
        except Exception as e:
            logger.exception("Failed to get tokenizer info.")
            return self.create_error_response(
                f"Failed to get tokenizer info: {e!s}")


class TokenizerInfo:

    def __init__(self, tokenizer: AnyTokenizer, model_config: ModelConfig,
                 chat_template: Optional[str]):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.chat_template = chat_template

    def to_dict(self) -> Dict[str, Any]:
        """Return the tokenizer configuration."""
        return self._get_tokenizer_config()

    # Use the tokenizer's init_kwargs as the base (this contains the original config)
    def _get_tokenizer_config(self) -> Dict[str, Any]:
        """Get tokenizer configuration directly from the tokenizer object."""
        config = dict(self.tokenizer.init_kwargs) if hasattr(self.tokenizer, 'init_kwargs') and self.tokenizer.init_kwargs else {}
        # Remove file path fields
        config.pop('vocab_file', None)
        config.pop('merges_file', None)
        # Remove chat_template from tokenizer's init_kwargs - we'll add it back conditionally
        config.pop('chat_template', None)
        config = self._make_json_serializable(config)
        config['tokenizer_class'] = self.tokenizer.__class__.__bases__[0].__name__
        # Only include chat_template if explicitly requested (self.chat_template will be None when not requested)
        if self.chat_template is not None:
            config['chat_template'] = self.chat_template
        return config

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert any non-JSON-serializable objects to serializable format."""
        # This is a special case to handle AddedToken objects, which are not
        # directly JSON-serializable but have a 'content' attribute.
        if hasattr(obj, 'content'):
            return obj.content
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj