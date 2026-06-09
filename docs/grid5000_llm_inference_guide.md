# Grid'5000 LLM Inference Guide

This guide explains how to run this project's sentence-level LLM labeling on a
Grid'5000 GPU node with `llama-cpp-python`.

It reflects the first completed English-only pilot run:

- build 100 English-pilot prompt rows with
  `scripts/labeling/build_english_pilot_labeling_requests.py`
- load `unsloth/Qwen3.6-27B-MTP-GGUF` through
  `src/georeset_osm_web_evidence/labeling/llama_cpp.py`
- label the prompt batch with
  `scripts/labeling/run_llama_cpp_english_pilot_labeling.py`
- retrieve
  `data/processed/pilots/worldwide_sentence_pilot_10_english_only/sentence_candidates_llm_labeled.parquet`
- optionally build or inspect reviewer-facing copies under
  `data/review/english_sentence_pilot/`

Keep real Grid'5000 launchers, node names, SSH keys, tokens, personal paths,
and logs out of git. Store them in ignored local paths such as `.grid5000/`.

## 1. What Runs Where

Use three environments:

| Environment | Purpose |
|---|---|
| Local machine | Edit code, run tests, build prompt parquet/JSONL, sync repo/data, inspect returned labels. |
| Grid'5000 frontend | Hold a persistent project clone, submit OAR jobs, store persistent model cache. |
| Grid'5000 GPU node | Build/import CUDA-enabled `llama_cpp`, download/load GGUF weights, run inference. |

Do not build or import CUDA `llama_cpp` on the frontend. The frontend may expose
CUDA modules, but it usually does not expose the GPU driver library
`libcuda.so.1`. Build and use `llama-cpp-python` inside the allocated GPU job.

## 2. SSH Setup

Create SSH config outside the repository:

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
  StrictHostKeyChecking accept-new
```

Verify non-interactive access before doing anything else:

```bash
ssh -o BatchMode=yes g5k 'whoami'
ssh -o BatchMode=yes <site-alias> 'whoami'
```

If this fails with `Permission denied (publickey)`, fix SSH first. Usual causes
are the wrong Grid'5000 username, wrong private key, missing `ssh-add`, or a
host entry pointing to the wrong identity.

## 3. Ignored Local Material

The repository ignores Grid'5000-local material:

```text
.grid5000/
grid5000-local/
g5k-local/
*.oar
*.oar.log
*.oar.out
*.oar.err
```

Use those paths for:

- OAR launcher scripts
- fetched remote logs
- site-specific notes
- remote node names
- personal environment overrides
- tokens and private paths

The launcher used for a real run can live locally at a path like:

```text
.grid5000/run_llama_cpp_english_pilot_<site>.sh
```

Then sync it to the remote project clone before submitting the job.

## 4. Local Prompt Artifacts

For the English-only pilot, build prompt requests locally first:

```bash
uv run python scripts/labeling/build_english_pilot_labeling_requests.py
```

This writes:

```text
data/processed/pilots/worldwide_sentence_pilot_10_english_only/llm_labeling_requests.parquet
data/processed/pilots/worldwide_sentence_pilot_10_english_only/llm_labeling_requests.jsonl
```

The remote launcher also runs this builder, but having the files locally makes
it easy to inspect the exact prompt batch before spending GPU time.

Expected prompt metadata:

```text
prompt_version: binary_remote_sensing_relevance_json_v2
```

The model is instructed to use only the sentence and return exactly one JSON
object:

```json
{"label":"relevant"}
```

or:

```json
{"label":"irrelevant"}
```

## 5. Sync Repository And Data

On the Grid'5000 frontend, keep a persistent clone at a stable path such as:

```text
~/georeset-osm-web-evidence
```

From the local machine, sync source while excluding local/remote environments
and the whole local data folder:

```bash
rsync -az \
  --exclude .git \
  --exclude .grid5000 \
  --exclude .venv \
  --exclude .venv-g5k \
  --exclude data \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  ./ <site-alias>:~/georeset-osm-web-evidence/
```

Do not use `--delete` unless every remote-only directory is excluded. In
particular, exclude `.venv-g5k`; otherwise rsync can break the remote virtual
environment.

Sync only the pilot data needed by the remote job:

```bash
remote_dir="~/georeset-osm-web-evidence/data/processed/pilots/worldwide_sentence_pilot_10_english_only"

ssh <site-alias> "mkdir -p ${remote_dir}"

rsync -az \
  data/processed/pilots/worldwide_sentence_pilot_10_english_only/sentence_candidates.parquet \
  <site-alias>:${remote_dir}/

rsync -az \
  data/processed/pilots/worldwide_sentence_pilot_10_english_only/llm_labeling_requests.parquet \
  data/processed/pilots/worldwide_sentence_pilot_10_english_only/llm_labeling_requests.jsonl \
  <site-alias>:${remote_dir}/
```

If the request files are missing, the remote launcher can regenerate them from
`sentence_candidates.parquet`.

Sync the launcher intentionally, not as part of the broad source sync:

```bash
ssh <site-alias> 'mkdir -p ~/georeset-osm-web-evidence/.grid5000'

rsync -az \
  .grid5000/run_llama_cpp_english_pilot_<site>.sh \
  <site-alias>:~/georeset-osm-web-evidence/.grid5000/
```

## 6. Model And Generation Settings

The current adapter defaults are defined in
`src/georeset_osm_web_evidence/labeling/llama_cpp.py`.

For the completed pilot run, the effective settings were:

| Setting | Value |
|---|---|
| Model repository | `unsloth/Qwen3.6-27B-MTP-GGUF` |
| GGUF file | `Qwen3.6-27B-Q4_0.gguf` |
| Quantization | `Q4_0` |
| Base model metadata | `Qwen3.6-27B` |
| `llama-cpp-python` | `0.3.25` |
| `n_gpu_layers` | `-1` |
| `n_ctx` | `4096` |
| `verbose` | `1` |
| thinking | disabled with `GEORESET_LLAMA_ENABLE_THINKING=0` |
| `temperature` | `0.0` |
| `max_tokens` | `24` |
| `top_p` | `1.0` |
| `top_k` | `40` |

The code default for `n_ctx` is `8192`, but the working Grid'5000 launcher
overrode it to `4096`. That was sufficient for short sentence-labeling prompts
and fit on an RTX A5000 24GB GPU.

Set runtime variables in the launcher:

```bash
export GEORESET_LLAMA_REPO_ID="unsloth/Qwen3.6-27B-MTP-GGUF"
export GEORESET_LLAMA_FILENAME="Qwen3.6-27B-Q4_0.gguf"
export GEORESET_LLAMA_N_GPU_LAYERS="-1"
export GEORESET_LLAMA_N_CTX="4096"
export GEORESET_LLAMA_VERBOSE="1"
export GEORESET_LLAMA_ENABLE_THINKING="0"
```

Use persistent Hugging Face cache paths:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
```

The model is public, so unauthenticated download can work. If you use a token,
keep it in the remote environment or ignored local notes, never in git.

## 7. CUDA Build Settings

Build CUDA-enabled `llama-cpp-python` inside the GPU job:

```bash
export GEORESET_LLAMA_CPP_CUDA_ARCHITECTURES="${GEORESET_LLAMA_CPP_CUDA_ARCHITECTURES:-80;86;90}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

CUDACXX="$(which nvcc)" \
CC="$(which gcc)" \
CXX="$(which g++)" \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${GEORESET_LLAMA_CPP_CUDA_ARCHITECTURES}" \
FORCE_CMAKE=1 \
  python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

Useful CUDA architecture values:

| GPU class | CUDA architecture |
|---|---|
| NVIDIA A100 | `80` |
| NVIDIA RTX A5000 / A40 | `86` |
| NVIDIA H100 | `90` |

The completed run used a launcher default of `80;86;90`, which is broader than
necessary but robust when the exact GPU class may vary.

If inference fails with:

```text
CUDA error: no kernel image is available for execution on the device
```

the wheel was built for the wrong GPU architecture. Rebuild with the allocated
node's architecture included.

## 8. Remote Launcher Shape

A launcher should do these steps:

1. `cd` into `~/georeset-osm-web-evidence`.
2. Load compatible Python, GCC, CMake, and CUDA modules.
3. Print `hostname`, `date`, `nvidia-smi`, Python, compiler, and `nvcc` info.
4. Create `.venv-g5k` if missing.
5. Delete `.venv-g5k` if it exists but has no executable Python.
6. Activate `.venv-g5k`.
7. Install the project with `python -m pip install -e .`.
8. Install CUDA `llama-cpp-python` if `import llama_cpp` fails.
9. Print the `llama_cpp` version.
10. Set `HF_HOME`, `HF_HUB_CACHE`, and `GEORESET_LLAMA_*`.
11. Run `scripts/labeling/build_english_pilot_labeling_requests.py`.
12. Run `scripts/labeling/run_llama_cpp_english_pilot_labeling.py`.
13. Print row count, label counts, and parse-error counts.

Use a normal Python virtual environment on Grid'5000 if `uv` is unavailable:

```bash
python -m venv .venv-g5k
. .venv-g5k/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

The completed run used modules equivalent to:

```bash
module purge
module load python/3.10.8_gcc-10.4.0
module load gcc/10.4.0_gcc-10.4.0
module load cmake/3.23.3_gcc-10.4.0
module load cuda/12.2.1_gcc-10.4.0
```

Module names differ by site and date. Treat these as a working example, not a
portable guarantee.

## 9. Submit An OAR Job

Use scripted OAR jobs for real runs:

```bash
ssh <site-alias> '
  cd ~/georeset-osm-web-evidence &&
  mkdir -p ~/codex_g5k_logs &&
  oarsub -t besteffort \
    -l /nodes=1/gpu=1,walltime=02:00:00 \
    -p "gpu_mem >= 24000" \
    -n georeset_llama_cpp_english_pilot \
    -O ~/codex_g5k_logs/georeset_llama_cpp_english_pilot.%jobid%.out \
    -E ~/codex_g5k_logs/georeset_llama_cpp_english_pilot.%jobid%.err \
    ./.grid5000/run_llama_cpp_english_pilot_<site>.sh
'
```

For the Qwen3.6 27B Q4_0 pilot with `n_ctx=4096`, an RTX A5000 with 24GB VRAM
was enough. If you increase context, batch size, or model precision, request
more VRAM.

Use `besteffort` for quick pilot work when preemption is acceptable. Use a
regular scheduled job for longer production runs.

Some sites require extra OAR types for specific GPU partitions. For example, an
H100 or V100 partition may require an additional site-specific type such as
`-t exotic`. Prefer the smallest GPU class that fits the model to reduce queue
time.

Monitor the job:

```bash
ssh <site-alias> 'oarstat -j <job-id> -s'
ssh <site-alias> 'tail -120 ~/codex_g5k_logs/georeset_llama_cpp_english_pilot.<job-id>.out'
ssh <site-alias> 'tail -120 ~/codex_g5k_logs/georeset_llama_cpp_english_pilot.<job-id>.err'
```

Do not assume `rg` exists on remote frontends; use `grep`, `sed`, and `tail` in
remote one-liners.

Cancel duplicate or stale jobs when one site has produced the needed result:

```bash
ssh <site-alias> 'oardel <job-id>'
```

## 10. Retrieve And Verify Results

After job success, retrieve the labeled parquet:

```bash
rsync -az \
  <site-alias>:~/georeset-osm-web-evidence/data/processed/pilots/worldwide_sentence_pilot_10_english_only/sentence_candidates_llm_labeled.parquet \
  data/processed/pilots/worldwide_sentence_pilot_10_english_only/sentence_candidates_llm_labeled.parquet
```

Optionally retrieve logs into ignored local storage:

```bash
mkdir -p .grid5000
rsync -az <site-alias>:~/codex_g5k_logs/georeset_llama_cpp_english_pilot.<job-id>.out .grid5000/
rsync -az <site-alias>:~/codex_g5k_logs/georeset_llama_cpp_english_pilot.<job-id>.err .grid5000/
```

Verify locally:

```bash
uv run python - <<'PY'
import pandas as pd

path = (
    "data/processed/pilots/worldwide_sentence_pilot_10_english_only/"
    "sentence_candidates_llm_labeled.parquet"
)
df = pd.read_parquet(path)
print("rows", len(df))
print(df["llm_label"].value_counts(dropna=False))
print(df["parse_error"].value_counts(dropna=False))
print(df[["sentence_id", "model_input", "llm_label", "raw_response", "parse_error"]].head(20).to_string(index=False))
PY
```

The run is acceptable only if:

- row count matches the prompt batch,
- `llm_label` is populated for every row,
- labels are only `relevant` or `irrelevant`,
- `parse_error` is empty for every row, or every parse error is intentionally
  inspected and accepted.

## 11. Completed English Pilot Run

The completed English-only pilot used:

| Item | Value |
|---|---|
| Site | Rennes |
| Host class | RTX A5000 GPU node |
| GPU memory | 24GB |
| `llama-cpp-python` | `0.3.25` |
| Model | `unsloth/Qwen3.6-27B-MTP-GGUF` |
| File | `Qwen3.6-27B-Q4_0.gguf` |
| Context | `4096` |
| Thinking | disabled |
| Temperature | `0.0` |
| Output format | strict JSON label |

Returned artifacts:

```text
data/processed/pilots/worldwide_sentence_pilot_10_english_only/sentence_candidates_llm_labeled.parquet
data/review/english_sentence_pilot/llm_labeled_english_sentence_pilot.csv
data/review/english_sentence_pilot/llm_labeled_english_sentence_pilot.xlsx
```

Result summary:

| Metric | Value |
|---|---:|
| Rows labeled | 100 |
| `relevant` | 39 |
| `irrelevant` | 61 |
| Parse errors | 0 |

This proves the infrastructure path and parser worked for the pilot. It does
not prove the prompt is final or that the model agrees with human judgment.

## 12. Troubleshooting

### SSH Permission Denied

Symptom:

```text
Permission denied (publickey)
```

Fix:

- verify the Grid'5000 username,
- verify the `IdentityFile`,
- run `ssh-add <private-key>` if relying on an agent,
- test with `ssh -o BatchMode=yes`.

### Host Key Failure

Symptom:

```text
Host key verification failed
```

Fix: either connect once manually and accept the host key, or use
`StrictHostKeyChecking accept-new` in the site alias.

### `libcuda.so.1` Missing On Frontend

Symptom:

```text
libcuda.so.1: cannot open shared object file
```

Cause: the frontend is not the GPU node.

Fix: build/import CUDA-enabled `llama_cpp` inside the OAR GPU job.

### Missing `huggingface_hub`

Symptom:

```text
Llama.from_pretrained requires the huggingface-hub package
```

Fix: keep `huggingface-hub` installed in the project environment. The project
declares it because `Llama.from_pretrained(...)` needs it.

### Unexpected `chat_template_kwargs`

Symptom:

```text
Llama.create_chat_completion() got an unexpected keyword argument 'chat_template_kwargs'
```

Cause: in `llama-cpp-python==0.3.25`, `create_chat_completion(...)` does not
accept template kwargs directly.

Fix: attach template kwargs to the model chat handler after load. The adapter
does this in `apply_chat_template_kwargs(...)`.

### Remote Venv Damaged By Sync

Symptom:

```text
.venv-g5k/bin/activate: No such file or directory
```

Cause: a sync command deleted part of the remote virtual environment.

Fix:

- exclude `.venv-g5k` from rsync,
- recreate `.venv-g5k`,
- avoid `--delete` unless all remote-only directories are excluded.

### Missing `.bashrc` Warning

Symptom:

```text
/var/lib/oar/.batch_job_bashrc: line 5: /home/<user>/.bashrc: No such file or directory
```

This warning can be harmless if the launcher still loads modules and runs. Check
the job output and result artifact before treating it as a failure.

## 13. Scaling Beyond The Pilot

Before labeling thousands of sentences:

1. Run a 20- to 100-row pilot.
2. Inspect `raw_response` and `parse_error`.
3. Compare model labels against human review.
4. Estimate seconds per sentence from logs.
5. Choose batch size from walltime and queue expectations.
6. Write resumable outputs, not one huge all-or-nothing job.
7. Keep the Hugging Face model cache persistent.
8. Keep launchers and logs ignored.

For production-scale labeling, add a resumable runner that writes partial
results after each chunk.
