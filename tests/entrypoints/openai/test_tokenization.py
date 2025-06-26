# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import pytest_asyncio
import requests

from vllm.transformers_utils.tokenizer import get_tokenizer

from ...utils import RemoteOpenAIServer
from .test_completion import zephyr_lora_added_tokens_files  # noqa: F401
from .test_completion import zephyr_lora_files  # noqa: F401

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def server(zephyr_lora_added_tokens_files: str):  # noqa: F811
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        # lora config
        "--enable-lora",
        "--lora-modules",
        f"zephyr-lora2={zephyr_lora_added_tokens_files}",
        "--max-lora-rank",
        "64",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def tokenizer_name(model_name: str,
                   zephyr_lora_added_tokens_files: str):  # noqa: F811
    return zephyr_lora_added_tokens_files if (
        model_name == "zephyr-lora2") else model_name


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_completions(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    for add_special in [False, True]:
        prompt = "vllm1 This is a test prompt."
        tokens = tokenizer.encode(prompt, add_special_tokens=add_special)

        response = requests.post(server.url_for("tokenize"),
                                 json={
                                     "add_special_tokens": add_special,
                                     "model": model_name,
                                     "prompt": prompt
                                 })
        response.raise_for_status()

        result = response.json()
        assert result["tokens"] == tokens
        assert result["count"] == len(tokens)
        assert result["max_model_len"] == 8192
        assert result["token_strs"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_chat(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    for add_generation in [False, True]:
        for add_special in [False, True]:
            conversation = [{
                "role": "user",
                "content": "Hi there!"
            }, {
                "role": "assistant",
                "content": "Nice to meet you!"
            }, {
                "role": "user",
                "content": "Can I ask a question? vllm1"
            }]
            for continue_final in [False, True]:
                if add_generation and continue_final:
                    continue
                if continue_final:
                    conversation.append({
                        "role": "assistant",
                        "content": "Sure,"
                    })

                prompt = tokenizer.apply_chat_template(
                    add_generation_prompt=add_generation,
                    continue_final_message=continue_final,
                    conversation=conversation,
                    tokenize=False)
                tokens = tokenizer.encode(prompt,
                                          add_special_tokens=add_special)

                response = requests.post(server.url_for("tokenize"),
                                         json={
                                             "add_generation_prompt":
                                             add_generation,
                                             "continue_final_message":
                                             continue_final,
                                             "add_special_tokens": add_special,
                                             "messages": conversation,
                                             "model": model_name
                                         })
                response.raise_for_status()

                result = response.json()
                assert result["tokens"] == tokens
                assert result["count"] == len(tokens)
                assert result["max_model_len"] == 8192
                assert result["token_strs"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_chat_with_tools(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    for add_generation in [False, True]:
        for add_special in [False, True]:
            conversation = [{
                "role":
                "user",
                "content":
                "What's the weather like in Paris today?",
            }]

            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            }
                        },
                    },
                },
            }]

            for continue_final in [False, True]:
                if add_generation and continue_final:
                    continue
                if continue_final:
                    conversation.append({
                        "role": "assistant",
                        "content": "Sure,"
                    })

                prompt = tokenizer.apply_chat_template(
                    add_generation_prompt=add_generation,
                    continue_final_message=continue_final,
                    conversation=conversation,
                    tools=tools,
                    tokenize=False,
                )
                tokens = tokenizer.encode(prompt,
                                          add_special_tokens=add_special)

                response = requests.post(
                    server.url_for("tokenize"),
                    json={
                        "add_generation_prompt": add_generation,
                        "continue_final_message": continue_final,
                        "add_special_tokens": add_special,
                        "messages": conversation,
                        "model": model_name,
                        "tools": tools,
                    },
                )
                response.raise_for_status()

                result = response.json()
                assert result["tokens"] == tokens
                assert result["count"] == len(tokens)
                assert result["max_model_len"] == 8192
                assert result["token_strs"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name, tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_with_return_token_strs(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    prompt = "This is a token_strs test prompt! vllm1"
    response = requests.post(
        server.url_for("tokenize"),
        json={
            "prompt": prompt,
            "model": model_name,
            "return_token_strs": True
        },
    )
    response.raise_for_status()

    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    tokens_str = tokenizer.convert_ids_to_tokens(tokens)

    result = response.json()
    assert result["tokens"] == tokens
    assert result["count"] == len(tokens)
    assert result["max_model_len"] == 8192
    assert result["token_strs"] == tokens_str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_detokenize(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    prompt = "This is a test prompt. vllm1"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    response = requests.post(server.url_for("detokenize"),
                             json={
                                 "model": model_name,
                                 "tokens": tokens
                             })
    response.raise_for_status()

    assert response.json() == {"prompt": prompt}

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_get_tokenizer_info_basic(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    """Test basic tokenizer info endpoint functionality."""
    response = requests.get(server.url_for("get_tokenizer_info"))
    response.raise_for_status()
    
    result = response.json()
    
    # Verify required fields are present
    assert "tokenizer_type" in result
    assert "vocab_size" in result
    assert "tokenizer_backend" in result
    assert "special_tokens" in result
    assert "model_name" in result
    
    # Verify data types
    assert isinstance(result["vocab_size"], int)
    assert isinstance(result["special_tokens"], dict)
    assert isinstance(result["supports_encoding"], bool)
    assert isinstance(result["supports_decoding"], bool)


@pytest.mark.asyncio
async def test_get_tokenizer_info_special_tokens(server: RemoteOpenAIServer):
    """Test that special tokens are correctly extracted."""
    response = requests.get(server.url_for("get_tokenizer_info"))
    response.raise_for_status()
    
    result = response.json()
    special_tokens = result["special_tokens"]
    
    # Should have some special tokens for zephyr model
    assert len(special_tokens) > 0
    
    # All values should be integers (token IDs)
    for token_str, token_id in special_tokens.items():
        assert isinstance(token_str, str)
        assert isinstance(token_id, int)
        assert token_id >= 0


@pytest.mark.asyncio
async def test_get_tokenizer_info_consistency_with_tokenize(
    server: RemoteOpenAIServer,
):
    """Test that tokenizer info is consistent with actual tokenization."""
    # Get tokenizer info
    info_response = requests.get(server.url_for("get_tokenizer_info"))
    info_response.raise_for_status()
    info = info_response.json()
    
    # Tokenize a simple prompt
    prompt = "Hello world!"
    tokenize_response = requests.post(
        server.url_for("tokenize"),
        json={"model": MODEL_NAME, "prompt": prompt}
    )
    tokenize_response.raise_for_status()
    tokenize_result = tokenize_response.json()
    
    # Verify consistency
    assert info["model_max_length"] == tokenize_result["max_model_len"]
    
    # Verify special tokens exist in vocab
    for token_str, token_id in info["special_tokens"].items():
        assert token_id < info["vocab_size"]


@pytest.mark.asyncio
async def test_get_tokenizer_info_chat_template(server: RemoteOpenAIServer):
    """Test chat template information is correctly detected."""
    response = requests.get(server.url_for("get_tokenizer_info"))
    response.raise_for_status()
    
    result = response.json()
    
    # Zephyr should have a chat template
    assert result["has_chat_template"] is True
    assert result["chat_template"] is not None
    assert result["chat_template_source"] in ["tokenizer", "config", "underlying"]
    assert isinstance(result["supports_system_message"], bool)
    assert isinstance(result["supports_tools"], bool)


@pytest.mark.asyncio
async def test_get_tokenizer_info_response_schema(server: RemoteOpenAIServer):
    """Test that the response matches the expected schema."""
    response = requests.get(server.url_for("get_tokenizer_info"))
    response.raise_for_status()
    
    result = response.json()
    
    # Required fields
    required_fields = {
        "tokenizer_type", "vocab_size", "tokenizer_backend", "is_cached", 
        "is_fast", "model_name", "special_tokens", "model_max_length",
        "truncation_side", "padding_side", "clean_up_tokenization_spaces",
        "supports_encoding", "supports_decoding", "has_chat_template",
        "chat_template_source", "supports_system_message", "supports_tools"
    }
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"