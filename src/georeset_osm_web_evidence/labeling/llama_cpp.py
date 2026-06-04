from typing import Any


DEFAULT_REPO_ID = "unsloth/Qwen3.6-27B-MTP-GGUF"
DEFAULT_FILENAME = "Qwen3.6-27B-Q4_0.gguf"


def load_llama_cpp_model(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    llama_class: Any | None = None,
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

    return llama_class.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        **model_kwargs,
    )


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
    enable_thinking: bool = False,
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
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        return extract_chat_completion_text(response)

    return label_fn
