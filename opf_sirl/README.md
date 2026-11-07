# OPF-SIRL

Solver-informed reinforcement learning (SIRL, Chen et al. 2025, [Cardinal-Operations/SIRL](https://github.com/Cardinal-Operations/SIRL))
applied to **optimal power flow modelling with MATPOWER**, evaluated on the
[ProOPF benchmark](../ProOPF_benchmark). The LLM turns a natural-language OPF request into MATPOWER
code; MATLAB executes it and the solver result is the reward.

## SIRL → this project

| SIRL (Gurobi/COPT) | here (MATPOWER) | file |
|---|---|---|
| `<think><model><python>` answer | `<think><model><matlab>` answer | `opf_sirl/prompts.py`, `opf_sirl/parsing.py` |
| Python executor | MATLAB engine pool behind an HTTP service | `opf_sirl/verifier/` |
| Stage-1 reward: format 0.5 + exec 1 + answer 1 | same; answer = ProOPF criterion `|o-g|/|g+ε| ≤ 1e-4` | `opf_sirl/reward.py` |
| Stage-2 reward: + variable-type hit rate × answer | + hit rate of modified case-table entries × answer | `opf_sirl/instance.py` |
| Partial-KL (KL-in-reward from the code tag) | same, on verl 0.5.0 | `trainers/partial_kl.py`, `trainers/verl_sirl/` |
| REINFORCE++, kl_coef 5e-4, clip 0.2/0.3 | same (`configs/verl_*.sh`) | |
| data synthesis + solver verification | template code → MATLAB label → LLM wording → back-translation check | `opf_sirl/synthesis/` |
| — | alternative back-end: TRL GRPO + Partial-KL | `trainers/trl_grpo/` |

Implementation notes
* Partial-KL is applied by replacing verl's `ray_trainer.apply_kl_penalty` (the only function SIRL changes)
  through Ray's `worker_process_setup_hook`, instead of vendoring SIRL's full copy of `ray_trainer.py`.
* SIRL's `partial_kl_tensor` zeroes `response_mask` **in place**, which silently also restricts the policy
  loss to the code part. Our mask is a copy (KL restricted, policy loss on the full response).
* The paper describes the KL region as "model + code"; SIRL's code starts it at the code tag. We follow the
  code (`OPF_PARTIAL_KL_TAG=matlab`); set `OPF_PARTIAL_KL_TAG=model` for the paper's variant.
* **Solver alignment.** The ProOPF labels were produced with MATPOWER's own MIPS. If CPLEX/Gurobi/quadprog are
  installed, MATPOWER's `DEFAULT` algorithm picks them instead and some level-3 problems give different or
  failed solutions. The verifier therefore hides those solvers at engine start (`MatlabConfig.disabled_solvers`).
  With that, `scripts/verify_benchmark.py` reproduces 210/216 ProOPF_B labels (incl. OOD). The 6 exceptions:
  3 level-2 records have no strategy values in the benchmark itself (ProOPF's own tester cannot run them either),
  and 3 OOD case189 AC-OPF objectives differ by 1–2 % (likely a local optimum / MATPOWER version difference).
* **ProOPF_D is noisy.** Many of its reference programs do not run: `define_constants; clear;` wipes the
  column names, options unknown to MATPOWER 8.1, buses/branches that do not exist in the case, MINLP, and
  level-2/4 modifications whose target does not exist or has default 0. `label_seeds.py` keeps only runnable,
  converged seeds: **48 = 39/7/1/1 for levels 1–4** (`data/seeds_labelled.jsonl`). Removing the `clear` lines
  was tried and recovers none (those programs then fail on the missing buses/branches), so it is not applied.
* **Training data = ProOPF only.** The current experiment trains on these 48 ProOPF_D seeds
  (`data/train.parquet`) and evaluates on all of ProOPF_B (124) + the three OOD sets. Levels 3/4 have one
  training prompt each, so improvements there must come from transfer. The synthesis pipeline
  (`opf_sirl/synthesis/`) is implemented and tested but not used for this experiment.
* **Synthesized level-2/4 requests** must mention the modifications in spec order, because that order is the
  positional argument order of `run_opf`; back-translation checks the order.

## Layout

```
opf_sirl/            framework-independent core
  data/              Sample schema, ProOPF loaders, parquet export
  prompts.py         system prompt + ProOPF task requirements per level
  parsing.py         tag / code extraction
  verifier/          harness (code -> MATLAB program), MATLAB engine pool, HTTP server/client
  reward.py          compute_score(...) with verl's batch reward signature
  instance.py        stage-2 instance-level comparison
  run_info.py        provenance (git commit, GPUs, package versions, env) for run_info.json
  synthesis/         seed labelling, spec sampling, code templates, LLM writer, pipeline
trainers/
  partial_kl.py      shared partial-KL mask
  verl_sirl/         default back-end (SIRL original: verl + REINFORCE++ + Partial-KL)
  trl_grpo/          alternative back-end (TRL GRPO + Partial-KL)
configs/
  verl_server_7b.sh  server training (hyper-parameters only)
  verl_local_smoke.sh  local 8 GB smoke test (hyper-parameters only)
  launch_verl.sh     shared launcher: run directory, run_info.json, train.log, summary
  trl_local_smoke.yaml
scripts/
  check_env, start_verifier, verify_benchmark      environment / verifier
  label_seeds, synthesize, build_dataset           data
  train, merge_checkpoint.sh                       training
  record_run, summarize_run                        run records (called by launch_verl.sh)
  evaluate, compare_evals                          evaluation
tests/               unit tests (+ `-m matlab` integration tests)
```

## Run outputs

A verl run writes everything to `outputs/runs/<RUN_NAME>/` (re-running with the same `RUN_NAME` resumes):

| file | content |
|---|---|
| `run_info.json` | git commit / dirty flag, host, GPUs, package versions, `OPF_*` env, model, train/val sizes per level, every hydra override (resumes add `run_info.resume-<time>.json`) |
| `train.log` | full console output of verl |
| `metrics.csv` | every verl metric per step (parsed from `train.log`) |
| `rollouts/<step>.jsonl` | every rollout: prompt, response, score, format / executed / correct / instance, level |
| `rollout_stats.csv` | per step: mean reward components and pass rate per level |
| `tensorboard/` | training curves (`tensorboard --logdir outputs/runs`) |
| `checkpoints/` | verl FSDP checkpoints (last 2 kept on the server) |
| `hf_model/` | merged HuggingFace model (`scripts/merge_checkpoint.sh`) |
| `eval/` | `scripts/evaluate.py` output: `predictions.jsonl`, `summary.json`, `summary.md`, `run_info.json` |
| `summary.md` | readable overview: setup, training table, evaluation (re-run `scripts/summarize_run.py` after eval) |

`scripts/compare_evals.py` puts several `eval/` results (e.g. base vs. stage 1 vs. stage 2) in one table.

## Setup

### Windows (MATLAB side)
MATLAB R2024b at `E:\app\matlab`, MATPOWER 8.1 at `E:\app\matlab\bin\matpower8.1`
(override with `MATPOWER_DIR`).
```powershell
python -m uv venv -p 3.12 .venv-win
python -m uv pip install -p .venv-win -r requirements-win.txt -e .
```
If `import matlab.engine` fails with `Could not find directory: E:\matlab\...` (the registry still points to
an old MATLAB location), rewrite the engine's `_arch.txt`:
```powershell
.venv-win\Scripts\python -c "r=r'E:\app\matlab'; import matlab, os; open(os.path.join(os.path.dirname(matlab.__file__),'engine','_arch.txt'),'w').write('\n'.join(['win64',r+r'\bin\win64',r+r'\extern\engines\python\dist\matlab\engine\win64',r+r'\extern\bin\win64']))"
```

### WSL2 / Linux (training side)
```bash
python3 -m venv ~/opf_sirl/.venv && . ~/opf_sirl/.venv/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-linux.txt   # see the comments in the file for flash-attn
pip install -e .
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/Qwen2.5-0.5B-Instruct
ln -s ~/models/Qwen2.5-0.5B-Instruct models/Qwen2.5-0.5B-Instruct
```
* WSL2 only gets half of the host RAM by default; verl (Ray + FSDP + vLLM) needs more than 15 GB.
  `%USERPROFILE%\.wslconfig` here: `[wsl2] memory=24GB swap=8GB`, then `wsl --shutdown`.
* RTX 50xx (sm_120): `configs/verl_local_smoke.sh` sets `CUDA_CACHE_MAXSIZE=4GB` (otherwise NCCL/flash-attn
  kernels are JIT-recompiled in every process, minutes per start) and does not force the xformers backend.
* LLM for data synthesis (DashScope / Bailian, OpenAI-compatible) is configured in `~/.opf_llm_env`
  (`LLM_API_KEY`, `LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`, `LLM_MODEL_NAME=qwen3.8-max`);
  `source ~/.opf_llm_env` before `scripts/synthesize.py`. Keep the key out of the repository.

## Workflow

```bash
# 1. Windows: start the verifier, bound to the WSL virtual interface (WSL's gateway IP, here 172.28.0.1)
.venv-win\Scripts\python scripts/start_verifier.py --host 172.28.0.1 --port 8765 --engines 3
# 2. check that the reference code reproduces the ProOPF labels
python scripts/verify_benchmark.py --url http://<win-host>:8765
# 3. data (ProOPF only): label the ProOPF_D seeds, all of them go to training
python scripts/label_seeds.py --url http://<win-host>:8765          # -> data/seeds_labelled.jsonl (committed)
python scripts/build_dataset.py data/seeds_labelled.jsonl --val-ratio 0
# 4. local smoke tests (3 steps each; check that the pipeline runs, not that the model learns)
OPF_VERIFIER_URL=http://<win-host>:8765 python scripts/train.py --backend verl --config configs/verl_local_smoke.sh
OPF_VERIFIER_URL=http://<win-host>:8765 python scripts/train.py --backend trl  --config configs/trl_local_smoke.yaml
```

The full server experiment (7B baseline, stage 1, stage 2, evaluation) is described step by step in the
[repository README](../README.md).

The verifier can run on any machine with MATLAB + MATPOWER (Windows or Linux); the trainer only needs its URL.

### Local smoke-test status (RTX 5060 8 GB, WSL2)
Both back-ends run end to end with the real MATLAB verifier: rollout → reward over HTTP → Partial-KL →
update (verl: ~15–20 s/step, peak 8.4 GB; TRL: 3 steps in ~80 s). Local limits:
* `verl_local_smoke.sh` keeps the actor in bf16 and uses tiny sizes; full-parameter fp32 0.5B does not fit
  (params + grads + Adam ≈ 8 GB during the update), and vLLM 0.9.2's LoRA kernels do not compile on sm_120.
* Qwen2.5-0.5B does not follow the `<think>/<model>/<matlab>` format and its MATPOWER code does not run, so all
  rewards are 0 and no learning signal appears locally. SIRL starts from a 7B instruct model; learning has to be
  checked on the server.

## Tests
```bash
pytest -m "not matlab"      # pure Python (Windows or WSL)
pytest -m matlab            # needs MATLAB + license (Windows)
```
