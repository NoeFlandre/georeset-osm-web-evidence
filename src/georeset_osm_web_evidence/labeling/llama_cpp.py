import os
from typing import Any


DEFAULT_REPO_ID = "unsloth/Qwen3.6-27B-MTP-GGUF"
DEFAULT_FILENAME = "Qwen3.6-27B-Q4_0.gguf"
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_CTX = 8192
DEFAULT_VERBOSE = False
DEFAULT_ENABLE_THINKING = False


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_llama_cpp_model_settings_from_env(env: dict | None = None) -> dict:
    if env is None:
        env = os.environ

    return {
        "repo_id": env.get("GEORESET_LLAMA_REPO_ID", DEFAULT_REPO_ID),
        "filename": env.get("GEORESET_LLAMA_FILENAME", DEFAULT_FILENAME),
        "model_kwargs": {
            "n_gpu_layers": int(
                env.get("GEORESET_LLAMA_N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS)
            ),
            "n_ctx": int(env.get("GEORESET_LLAMA_N_CTX", DEFAULT_N_CTX)),
            "verbose": _parse_bool(
                env.get("GEORESET_LLAMA_VERBOSE", str(DEFAULT_VERBOSE))
            ),
        },
        "chat_template_kwargs": {
            "enable_thinking": _parse_bool(
                env.get("GEORESET_LLAMA_ENABLE_THINKING", str(DEFAULT_ENABLE_THINKING))
            )
        },
    }


def apply_chat_template_kwargs(
    llm,
    chat_template_kwargs: dict[str, Any] | None,
    chat_format_module: Any | None = None,
):
    if not chat_template_kwargs:
        return llm

    base_chat_handler = getattr(llm, "chat_handler", None)
    if base_chat_handler is None:
        chat_handlers = getattr(llm, "_chat_handlers", {})
        chat_format = getattr(llm, "chat_format", None)
        base_chat_handler = chat_handlers.get(chat_format)

    if base_chat_handler is None:
        if chat_format_module is None:
            from llama_cpp import llama_chat_format as chat_format_module

        base_chat_handler = chat_format_module.get_chat_completion_handler(
            getattr(llm, "chat_format")
        )

    def chat_handler_with_template_kwargs(*args, **kwargs):
        return base_chat_handler(*args, **{**chat_template_kwargs, **kwargs})

    llm.chat_handler = chat_handler_with_template_kwargs
    return llm


def load_llama_cpp_model(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    llama_class: Any | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    **model_kwargs,
):
    if llama_class is None:
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise ImportError(
                "llama-cpp-python is required on the remote machine to load "
                "the GGUF model. Install it there before running this script."
            ) from error

        llama_class = Llama

    llm = llama_class.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        **model_kwargs,
    )
    return apply_chat_template_kwargs(llm, chat_template_kwargs)


def extract_chat_completion_text(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Could not extract assistant message from response") from error

    if not isinstance(content, str):
        raise ValueError("Assistant message content is not a string")

    return content.strip()


def create_llama_cpp_label_fn(
    llm,
    temperature: float = 0.0,
    max_tokens: int = 8,
    top_p: float = 1.0,
    top_k: int = 40,
):
    def label_fn(prompt: str) -> str:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        return extract_chat_completion_text(response)

    return label_fn
