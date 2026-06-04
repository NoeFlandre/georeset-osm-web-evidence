# Grid'5000 LLM Inference Guide

This guide explains how to run this project's LLM sentence-labeling smoke test
on a Grid'5000 GPU node with `llama-cpp-python`.

It is based on the current repository workflow:

- build prompt rows with `scripts/labeling/build_labeling_prompt_sample.py`
- load a GGUF model through `src/georeset_osm_web_evidence/labeling/llama_cpp.py`
- label a small sample with `scripts/labeling/run_llama_cpp_labeling_sample.py`
- retrieve `data/processed/labeling/llm_labels_sample.parquet`

Do not commit real Grid'5000 launchers, node names, personal paths, logs, SSH
keys, or tokens. Keep those in ignored local files such as `.grid5000/`.

## 1. What Runs Where

Use three environments:

| Environment | Purpose |
|---|---|
| Local machine | Edit code, run unit tests, sync repo/data, inspect returned artifacts. |
| Grid'5000 frontend | Submit OAR jobs, store persistent repo clone, store Hugging Face cache. |
| Grid'5000 GPU node | Build CUDA-enabled `llama-cpp-python`, load GGUF, run inference. |

Important distinction: the frontend may have CUDA toolkit modules, but it usually
does not expose the GPU driver library `libcuda.so.1`. Build and import the
CUDA-enabled `llama_cpp` package inside an allocated GPU job, not on the
frontend.

## 2. SSH Setup

Create a local SSH config outside the repository:

```sshconfig
Host g5k
  HostName access.grid5000.fr
  User <grid5000-username>
  IdentityFile <path-to-private-key>
  IdentitiesOnly yes

Host <site-alias>
  User <grid5000-username>
  ProxyJump g5k
  IdentityFile <path-to-private-key>
  IdentitiesOnly yes
```

Verify non-interactive access:

```bash
ssh -o BatchMode=yes g5k 'whoami'
ssh -o BatchMode=yes <site-alias> 'whoami'
```

If this fails with `Permission denied (publickey)`, fix SSH before touching
Python. Common causes are the wrong Grid'5000 username, the wrong private key,
or a key not loaded in the SSH agent.

## 3. Local Files To Keep Ignored

The repository `.gitignore` ignores local Grid'5000 material:

```text
.grid5000/
grid5000-local/
g5k-local/
*.oar
*.oar.log
*.oar.out
*.oar.err
```

Use these ignored locations for:

- OAR launcher scripts
- remote log copies
- site-specific notes
- local environment overrides
- tokens and private paths

## 4. Sync Repository And Data

On the frontend, keep a persistent project directory. From the local machine,
sync source code while excluding local and remote environments:

```bash
rsync -az \
  --exclude .git \
  --exclude .venv \
  --exclude .venv-g5k \
  --exclude data \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  ./ <site-alias>:~/georeset-osm-web-evidence/
```

Do not use `--delete` unless every remote-only path is excluded. In particular,
exclude `.venv-g5k`; otherwise rsync may damage the remote virtual environment.

Sync only the data files needed by the smoke test:

```bash
rsync -az data/processed/evidence/labeling_candidates.parquet \
  <site-alias>:~/georeset-osm-web-evidence/data/processed/evidence/

rsync -az data/processed/labeling/llm_labeling_requests_sample.parquet \
  <site-alias>:~/georeset-osm-web-evidence/data/processed/labeling/
```

If the request parquet does not exist yet, the remote job can regenerate it from
`labeling_candidates.parquet`.

## 5. Model And Runtime Settings

The current adapter defaults to:

```text
repo_id: unsloth/Qwen3.6-27B-MTP-GGUF
filename: Qwen3.6-27B-Q4_0.gguf
n_gpu_layers: -1
n_ctx: 8192 locally, commonly lowered to 4096 on Grid'5000 smoke tests
temperature: 0
enable_thinking: false
```

Runtime environment variables:

```bash
export GEORESET_LLAMA_REPO_ID="unsloth/Qwen3.6-27B-MTP-GGUF"
export GEORESET_LLAMA_FILENAME="Qwen3.6-27B-Q4_0.gguf"
export GEORESET_LLAMA_N_GPU_LAYERS="-1"
export GEORESET_LLAMA_N_CTX="4096"
export GEORESET_LLAMA_VERBOSE="1"
export GEORESET_LLAMA_ENABLE_THINKING="0"
```

Set Hugging Face cache paths to persistent storage:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
```

For a public model, unauthenticated Hugging Face downloads can work, but a
`HF_TOKEN` may improve rate limits. Keep tokens out of the repository.

## 6. CUDA Architecture

The CUDA wheel must be compiled for the allocated GPU architecture.

Useful architecture values:

| GPU class | CUDA architecture |
|---|---|
| NVIDIA A100 | `80` |
| NVIDIA A40 | `86` |
| NVIDIA H100 | `90` |

For a job that may land on A100 or A40, build for both:

```bash
export GEORESET_LLAMA_CPP_CUDA_ARCHITECTURES="80;86"
```

The CMake argument used by the working launcher is:

```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${GEORESET_LLAMA_CPP_CUDA_ARCHITECTURES}"
```

If you build only for `86` and the job runs on A100, inference can abort with:

```text
CUDA error: no kernel image is available for execution on the device
```

That means the wheel compiled successfully but for the wrong GPU target.

## 7. Remote Launcher Shape

Create a local ignored launcher such as:

```text
.grid5000/run_llama_cpp_smoke_<site>.sh
```

Then sync it to the frontend under an ignored or temporary name.

The launcher should do these steps:

1. `cd` into the remote repo.
2. Load Python, GCC, CMake, and CUDA modules compatible with each other.
3. Print `hostname`, `date`, `nvidia-smi`, Python, GCC, and `nvcc` versions.
4. Create `.venv-g5k` if missing.
5. Recreate `.venv-g5k` if it exists but is incomplete.
6. Activate `.venv-g5k`.
7. Install the project with `python -m pip install -e .`.
8. Install `llama-cpp-python` with CUDA if `import llama_cpp` fails.
9. Set Hugging Face cache paths and `GEORESET_LLAMA_*` variables.
10. Run the prompt builder.
11. Run the llama.cpp labeling smoke test.
12. Print the resulting labels and parse errors.

Use a regular Python virtual environment on Grid'5000 if `uv` is unavailable:

```bash
python -m venv .venv-g5k
. .venv-g5k/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Install CUDA-enabled `llama-cpp-python` inside the GPU job:

```bash
CUDACXX="$(which nvcc)" \
CC="$(which gcc)" \
CXX="$(which g++)" \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${GEORESET_LLAMA_CPP_CUDA_ARCHITECTURES}" \
FORCE_CMAKE=1 \
  python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

## 8. Submit An OAR Job

Use scripted OAR jobs for real runs:

```bash
ssh <site-alias> '
  cd ~/georeset-osm-web-evidence &&
  oarsub -t besteffort \
    -l /nodes=1/gpu=1,walltime=02:00:00 \
    -p "gpu_mem >= 40000" \
    -n georeset_llama_cpp_smoke \
    -O ~/codex_g5k_logs/georeset_llama_cpp_smoke.%jobid%.out \
    -E ~/codex_g5k_logs/georeset_llama_cpp_smoke.%jobid%.err \
    ./<remote-ignored-launcher>
'
```

Use `besteffort` for quick smoke tests when preemption is acceptable. Use a
regular scheduled job for long runs.

Monitor the job:

```bash
ssh <site-alias> 'oarstat -j <job-id> -s'
ssh <site-alias> 'tail -120 ~/codex_g5k_logs/georeset_llama_cpp_smoke.<job-id>.out'
ssh <site-alias> 'tail -120 ~/codex_g5k_logs/georeset_llama_cpp_smoke.<job-id>.err'
```

If you need to inspect the allocated node interactively:

```bash
ssh <site-alias> 'OAR_JOB_ID=<job-id> oarsh <allocated-node> "nvidia-smi"'
```

Do not commit the allocated node name.

## 9. Retrieve Results

After job success, retrieve the parquet:

```bash
rsync -az <site-alias>:~/georeset-osm-web-evidence/data/processed/labeling/llm_labels_sample.parquet \
  data/processed/labeling/llm_labels_sample.parquet
```

Optionally retrieve logs into ignored local storage:

```bash
mkdir -p .grid5000
rsync -az <site-alias>:~/codex_g5k_logs/georeset_llama_cpp_smoke.<job-id>.out .grid5000/
rsync -az <site-alias>:~/codex_g5k_logs/georeset_llama_cpp_smoke.<job-id>.err .grid5000/
```

Inspect locally:

```bash
uv run python - <<'PY'
import pandas as pd

path = "data/processed/labeling/llm_labels_sample.parquet"
df = pd.read_parquet(path)
print(df[["sentence_id", "model_input", "llm_label", "raw_response", "parse_error"]].to_string(index=False))
print(df["llm_label"].value_counts(dropna=False))
print(df["parse_error"].value_counts(dropna=False))
PY
```

## 10. Current Successful Smoke Test

The first successful remote smoke test used:

- `llama-cpp-python==0.3.25`
- `unsloth/Qwen3.6-27B-MTP-GGUF`
- `Qwen3.6-27B-Q4_0.gguf`
- `GEORESET_LLAMA_N_CTX=4096`
- `GEORESET_LLAMA_ENABLE_THINKING=0`
- CUDA build architectures `80;86`
- one A100-class GPU with 40GB VRAM

Returned local artifact:

```text
data/processed/labeling/llm_labels_sample.parquet
```

Result summary for 5 rows:

| Metric | Value |
|---|---:|
| Rows labeled | 5 |
| `relevant` | 4 |
| `irrelevant` | 1 |
| Parse errors | 0 |

This is only a smoke test. The labels prove the remote inference path works;
they do not prove the prompt is scientifically final.

## 11. Troubleshooting

### SSH Permission Denied

Symptom:

```text
Permission denied (publickey)
```

Fix:

- verify the Grid'5000 username
- verify `IdentityFile`
- use `ssh-add <private-key>` if relying on an agent
- test with `ssh -o BatchMode=yes`

### `libcuda.so.1` Missing On Frontend

Symptom:

```text
libcuda.so.1: cannot open shared object file
```

Cause: the frontend is not a GPU node. Build/import CUDA-enabled `llama_cpp`
inside an OAR GPU job.

### Missing `huggingface_hub`

Symptom:

```text
Llama.from_pretrained requires the huggingface-hub package
```

Fix: keep `huggingface-hub` in `pyproject.toml`. This project now declares it
because `Llama.from_pretrained(...)` needs it.

### Unexpected `chat_template_kwargs`

Symptom:

```text
Llama.create_chat_completion() got an unexpected keyword argument 'chat_template_kwargs'
```

Cause: in `llama-cpp-python==0.3.25`, `create_chat_completion(...)` does not
accept template kwargs directly.

Fix: attach template kwargs to the model chat handler after load. The current
adapter does this in `apply_chat_template_kwargs(...)`.

### Wrong CUDA Architecture

Symptom:

```text
CUDA error: no kernel image is available for execution on the device
```

Cause: the wheel was compiled for a different GPU architecture than the node
running inference.

Fix: rebuild with the correct architecture, or include multiple architectures:

```bash
export GEORESET_LLAMA_CPP_CUDA_ARCHITECTURES="80;86"
```

### Remote Venv Damaged By Sync

Symptom:

```text
.venv-g5k/bin/activate: No such file or directory
```

Cause: a sync command deleted part of the remote virtual environment.

Fix:

- exclude `.venv-g5k` from rsync
- recreate `.venv-g5k`
- avoid `--delete` unless all remote-only directories are excluded

## 12. Scaling Beyond The Smoke Test

Before scaling to thousands of sentences:

1. Run a 20-row sample.
2. Check `raw_response` and `parse_error`.
3. Estimate seconds per sentence from logs.
4. Choose batch size from walltime and queue expectations.
5. Write resumable outputs, not one huge all-or-nothing job.
6. Keep model cache persistent.
7. Keep job launchers and logs ignored.

The next engineering step should be a resumable batch runner that writes partial
LLM labels to parquet or JSONL after each chunk.
