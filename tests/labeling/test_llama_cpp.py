import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from georeset_osm_web_evidence.labeling.llama_cpp import (
    DEFAULT_ENABLE_THINKING,
    DEFAULT_FILENAME,
    DEFAULT_N_CTX,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_REPO_ID,
    apply_chat_template_kwargs,
    create_llama_cpp_label_fn,
    extract_chat_completion_text,
    get_llama_cpp_model_settings_from_env,
    load_llama_cpp_model,
)
from georeset_osm_web_evidence.labeling.llama_cpp_batch import (
    format_llm_labeling_summary,
    run_llama_cpp_prompt_batch,
)
from scripts.labeling.run_llama_cpp_labeling_sample import (
    run_llama_cpp_labeling_sample,
)
from scripts.labeling.run_llama_cpp_english_pilot_labeling import (
    run_llama_cpp_english_pilot_labeling,
)
from scripts.labeling.run_llama_cpp_context_query_pilot_labeling import (
    run_llama_cpp_context_query_pilot_labeling,
)
from scripts.labeling.run_llama_cpp_location_topic_pilot_labeling import (
    run_llama_cpp_location_topic_pilot_labeling,
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
    def test_reads_model_settings_from_environment(self):
        env = {
            "GEORESET_LLAMA_REPO_ID": "example/model",
            "GEORESET_LLAMA_FILENAME": "model-q4.gguf",
            "GEORESET_LLAMA_N_GPU_LAYERS": "32",
            "GEORESET_LLAMA_N_CTX": "4096",
            "GEORESET_LLAMA_VERBOSE": "1",
            "GEORESET_LLAMA_ENABLE_THINKING": "0",
        }

        settings = get_llama_cpp_model_settings_from_env(env)

        self.assertEqual(settings["repo_id"], "example/model")
        self.assertEqual(settings["filename"], "model-q4.gguf")
        self.assertEqual(settings["model_kwargs"]["n_gpu_layers"], 32)
        self.assertEqual(settings["model_kwargs"]["n_ctx"], 4096)
        self.assertEqual(settings["model_kwargs"]["verbose"], True)
        self.assertEqual(settings["chat_template_kwargs"]["enable_thinking"], False)

    def test_model_settings_defaults_target_qwen_smoke_test(self):
        settings = get_llama_cpp_model_settings_from_env({})

        self.assertEqual(settings["repo_id"], DEFAULT_REPO_ID)
        self.assertEqual(settings["filename"], DEFAULT_FILENAME)
        self.assertEqual(settings["model_kwargs"]["n_gpu_layers"], DEFAULT_N_GPU_LAYERS)
        self.assertEqual(settings["model_kwargs"]["n_ctx"], DEFAULT_N_CTX)
        self.assertEqual(settings["model_kwargs"]["verbose"], False)
        self.assertEqual(
            settings["chat_template_kwargs"]["enable_thinking"],
            DEFAULT_ENABLE_THINKING,
        )

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
        llm = FakeLlama(response_text='{"label":"irrelevant"}')
        label_fn = create_llama_cpp_label_fn(llm)

        response = label_fn("Classify this sentence.")

        self.assertEqual(response, '{"label":"irrelevant"}')
        self.assertEqual(len(llm.calls), 1)
        call = llm.calls[0]
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(call["max_tokens"], 24)
        self.assertNotIn("chat_template_kwargs", call)
        self.assertEqual(
            call["messages"],
            [
                {
                    "role": "user",
                    "content": "Classify this sentence.",
                }
            ],
        )

    def test_applies_chat_template_kwargs_to_chat_handler(self):
        calls = []

        def base_chat_handler(*args, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "relevant"}}]}

        class FakeModel:
            chat_handler = base_chat_handler

        model = FakeModel()

        apply_chat_template_kwargs(model, {"enable_thinking": False})
        response = model.chat_handler(messages=[], enable_thinking=True)

        self.assertEqual(response["choices"][0]["message"]["content"], "relevant")
        self.assertEqual(calls[0]["enable_thinking"], True)
        self.assertEqual(calls[0]["messages"], [])

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
            return (
                '{"label":"relevant"}'
                if prompt == "Prompt 1"
                else '{"label":"irrelevant"}'
            )

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

    def test_shared_llama_cpp_batch_runner_labels_and_saves_all_rows(self):
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
            return (
                '{"label":"relevant"}'
                if prompt == "Prompt 1"
                else '{"label":"irrelevant"}'
            )

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "requests.parquet"
            output_path = Path(temporary_directory) / "nested" / "labels.parquet"
            prompt_df.to_parquet(input_path, index=False)

            labeled_df = run_llama_cpp_prompt_batch(
                input_path=input_path,
                output_path=output_path,
                label_fn=fake_label_fn,
            )
            saved_df = pd.read_parquet(output_path)

        self.assertEqual(labeled_df["llm_label"].to_list(), ["relevant", "irrelevant"])
        self.assertEqual(saved_df["llm_label"].to_list(), ["relevant", "irrelevant"])
        self.assertEqual(saved_df["parse_error"].to_list(), [None, None])

    def test_shared_llama_cpp_batch_runner_can_limit_rows(self):
        prompt_df = pd.DataFrame(
            [
                {"sentence_id": "s1", "prompt": "Prompt 1", "model_input": "Sentence 1"},
                {"sentence_id": "s2", "prompt": "Prompt 2", "model_input": "Sentence 2"},
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "requests.parquet"
            output_path = Path(temporary_directory) / "labels.parquet"
            prompt_df.to_parquet(input_path, index=False)

            labeled_df = run_llama_cpp_prompt_batch(
                input_path=input_path,
                output_path=output_path,
                row_limit=1,
                label_fn=lambda prompt: '{"label":"relevant"}',
            )
            saved_df = pd.read_parquet(output_path)

        self.assertEqual(labeled_df["sentence_id"].to_list(), ["s1"])
        self.assertEqual(saved_df["sentence_id"].to_list(), ["s1"])

    def test_formats_llm_labeling_summary(self):
        labeled_df = pd.DataFrame(
            [
                {"llm_label": "relevant", "parse_error": None},
                {"llm_label": "irrelevant", "parse_error": None},
                {"llm_label": None, "parse_error": "invalid json"},
            ]
        )

        summary = format_llm_labeling_summary(labeled_df, Path("labels.parquet"))

        self.assertIn("Saved 3 LLM-labeled rows to labels.parquet", summary)
        self.assertIn("llm_label", summary)
        self.assertIn("relevant", summary)
        self.assertIn("irrelevant", summary)
        self.assertIn("parse_error", summary)
        self.assertIn("invalid json", summary)

    def test_english_pilot_script_labels_all_rows_by_default(self):
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
                {
                    "sentence_id": "s3",
                    "prompt": "Prompt 3",
                    "model_input": "Sentence 3",
                },
            ]
        )

        def fake_label_fn(prompt: str) -> str:
            return (
                '{"label":"relevant"}'
                if prompt in {"Prompt 1", "Prompt 3"}
                else '{"label":"irrelevant"}'
            )

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "english_requests.parquet"
            output_path = Path(temporary_directory) / "english_labels.parquet"
            prompt_df.to_parquet(input_path, index=False)

            labeled_df = run_llama_cpp_english_pilot_labeling(
                input_path=input_path,
                output_path=output_path,
                label_fn=fake_label_fn,
            )
            saved_df = pd.read_parquet(output_path)

        self.assertEqual(len(labeled_df), 3)
        self.assertEqual(
            labeled_df["llm_label"].to_list(),
            ["relevant", "irrelevant", "relevant"],
        )
        self.assertEqual(saved_df["llm_label"].to_list(), labeled_df["llm_label"].to_list())

    def test_context_query_pilot_script_labels_all_rows_by_default(self):
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
            return (
                '{"label":"relevant"}'
                if prompt == "Prompt 1"
                else '{"label":"irrelevant"}'
            )

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "context_requests.parquet"
            output_path = Path(temporary_directory) / "context_labels.parquet"
            prompt_df.to_parquet(input_path, index=False)

            labeled_df = run_llama_cpp_context_query_pilot_labeling(
                input_path=input_path,
                output_path=output_path,
                label_fn=fake_label_fn,
            )
            saved_df = pd.read_parquet(output_path)

        self.assertEqual(len(labeled_df), 2)
        self.assertEqual(labeled_df["llm_label"].to_list(), ["relevant", "irrelevant"])
        self.assertEqual(saved_df["llm_label"].to_list(), labeled_df["llm_label"].to_list())

    def test_location_topic_pilot_script_labels_all_rows_by_default(self):
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
            return (
                '{"label":"relevant"}'
                if prompt == "Prompt 1"
                else '{"label":"irrelevant"}'
            )

        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "location_topic_requests.parquet"
            output_path = Path(temporary_directory) / "location_topic_labels.parquet"
            prompt_df.to_parquet(input_path, index=False)

            labeled_df = run_llama_cpp_location_topic_pilot_labeling(
                input_path=input_path,
                output_path=output_path,
                label_fn=fake_label_fn,
            )
            saved_df = pd.read_parquet(output_path)

        self.assertEqual(len(labeled_df), 2)
        self.assertEqual(labeled_df["llm_label"].to_list(), ["relevant", "irrelevant"])
        self.assertEqual(saved_df["llm_label"].to_list(), labeled_df["llm_label"].to_list())


if __name__ == "__main__":
    unittest.main()
