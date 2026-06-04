import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from georeset_osm_web_evidence.labeling.llama_cpp import (
    DEFAULT_FILENAME,
    DEFAULT_REPO_ID,
    create_llama_cpp_label_fn,
    extract_chat_completion_text,
    load_llama_cpp_model,
)
from scripts.labeling.run_llama_cpp_labeling_sample import (
    run_llama_cpp_labeling_sample,
)


class FakeLlama:
    loaded_with = None

    def __init__(self, response_text: str = "relevant"):
        self.response_text = response_text
        self.calls = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.loaded_with = kwargs
        return cls()

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": self.response_text,
                    }
                }
            ]
        }


class LlamaCppProviderTests(unittest.TestCase):
    def test_loads_default_unsloth_qwen_model_lazily(self):
        llm = load_llama_cpp_model(
            llama_class=FakeLlama,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False,
        )

        self.assertIsInstance(llm, FakeLlama)
        self.assertEqual(FakeLlama.loaded_with["repo_id"], DEFAULT_REPO_ID)
        self.assertEqual(FakeLlama.loaded_with["filename"], DEFAULT_FILENAME)
        self.assertEqual(FakeLlama.loaded_with["n_gpu_layers"], -1)
        self.assertEqual(FakeLlama.loaded_with["n_ctx"], 8192)
        self.assertEqual(FakeLlama.loaded_with["verbose"], False)

    def test_label_function_calls_llama_cpp_with_deterministic_non_thinking_settings(self):
        llm = FakeLlama(response_text=" irrelevant \n")
        label_fn = create_llama_cpp_label_fn(llm)

        response = label_fn("Classify this sentence.")

        self.assertEqual(response, "irrelevant")
        self.assertEqual(len(llm.calls), 1)
        call = llm.calls[0]
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(call["max_tokens"], 8)
        self.assertEqual(
            call["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(
            call["messages"],
            [
                {
                    "role": "user",
                    "content": "Classify this sentence.",
                }
            ],
        )

    def test_extracts_chat_completion_text(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": "relevant",
                    }
                }
            ]
        }

        self.assertEqual(extract_chat_completion_text(response), "relevant")

        with self.assertRaises(ValueError):
            extract_chat_completion_text({"choices": []})

    def test_script_labels_prompt_sample_with_injected_label_function(self):
        prompt_df = pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "prompt": "Prompt 1",
                    "model_input": "Sentence 1",
                },
                {
                    "sentence_id": "s2",
                    "prompt": "Prompt 2",
                    "model_input": "Sentence 2",
                },
            ]
        )

        def fake_label_fn(prompt: str) -> str:
            return "relevant" if prompt == "Prompt 1" else "irrelevant"

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "requests.parquet"
            output_path = Path(temporary_directory) / "labels.parquet"
            prompt_df.to_parquet(input_path, index=False)

            labeled_df = run_llama_cpp_labeling_sample(
                input_path=input_path,
                output_path=output_path,
                sample_size=2,
                label_fn=fake_label_fn,
            )
            saved_df = pd.read_parquet(output_path)

        self.assertEqual(labeled_df["llm_label"].to_list(), ["relevant", "irrelevant"])
        self.assertEqual(saved_df["llm_label"].to_list(), ["relevant", "irrelevant"])
        self.assertEqual(saved_df["parse_error"].to_list(), [None, None])


if __name__ == "__main__":
    unittest.main()
