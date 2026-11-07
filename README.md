# OPF-SIRL 工作区

把 SIRL（Solver-Informed RL，[Cardinal-Operations/SIRL](https://github.com/Cardinal-Operations/SIRL)）用于
**MATPOWER 最优潮流（OPF）建模**：LLM 把自然语言 OPF 需求翻译成 MATPOWER 代码，MATLAB 执行代码，
求解结果作为奖励。评测基准是 ProOPF。

> 给远程服务器上的 agent：先读完本文件，再读 [`opf_sirl/README.md`](opf_sirl/README.md)（方法细节、
> 与 SIRL 的对应关系、已知问题）。本文件只说明仓库结构、在训练服务器上怎么跑、哪些东西不在仓库里。

## 仓库结构

```
OPF/                         <- 本仓库根目录
├── ProOPF_benchmark/        git submodule（上游公开仓库 shenchao188/ProOPF-Benchamrk-Dataset，固定在 5693293）
│   ├── ProOPF_B/            评测集（level1-4 + OOD），带标签
│   ├── ProOPF_D/            训练种子样例（参考代码噪声大，见 opf_sirl/README）
│   └── base_system/         case14/30/39/57/59/68/189/300.m，verifier 会加到 MATLAB path
└── opf_sirl/                本项目代码（Python 包 opf_sirl + trainers）
    ├── opf_sirl/            核心：prompts / parsing / reward / verifier / synthesis / data
    ├── trainers/            verl_sirl（默认，verl 0.5.0 + REINFORCE++ + Partial-KL）、trl_grpo（备选）
    ├── configs/             verl_server_7b.sh（服务器正式训练）、verl_local_smoke.sh、trl_local_smoke.yaml
    ├── scripts/             check_env / start_verifier / verify_benchmark / label_seeds / synthesize /
    │                        build_dataset / train / evaluate
    ├── data/                seeds_labelled.jsonl（已标注种子，48 条：L1 39 / L2 7 / L3 1 / L4 1）
    ├── models/              空目录，放模型权重或软链接（不入库）
    └── tests/
```

`opf_sirl/opf_sirl/config.py` 默认通过 `opf_sirl/../ProOPF_benchmark` 找基准，所以两个目录必须保持兄弟关系
（或设置 `PROOPF_ROOT`）。

## 克隆

```bash
git clone --recursive <本仓库 URL> OPF
# 已经 clone 过但 ProOPF_benchmark 是空的：
git submodule update --init --recursive
```

## 不在仓库里的东西（需要在服务器上自己准备）

| 内容 | 说明 |
|---|---|
| Python 环境 | 按下文"服务器环境"安装；本地的 `.venv-win/` 不入库 |
| 模型权重 | `opf_sirl/models/` 为空。`verl_server_7b.sh` 默认 `models/Qwen2.5-7B-Instruct`，也可用 `MODEL_PATH` 指定 |
| `data/train.parquet`、`data/val.parquet` | **还没生成**。需要先合成数据再 `build_dataset.py`（见"当前进度"） |
| MATLAB verifier | 奖励必须经过 MATLAB + MATPOWER 8.1 执行，服务器只需能访问 verifier 的 HTTP 地址 |
| LLM API key | 仅数据合成需要，放在 `~/.opf_llm_env`，**不要提交** |
| `outputs/` | 运行产物（hydra 日志、checkpoint、评测结果），不入库 |

## 关键依赖：MATLAB verifier

训练时每个 rollout 都要把生成的 MATPOWER 代码发到 verifier 执行：

- verifier = `scripts/start_verifier.py`，一个 HTTP 服务（`GET /health`、`POST /execute`），内部是 MATLAB engine 池。
- 需要 MATLAB（R2024b 测试过，Python 3.12；R2024b engine 不支持 3.13）+ MATPOWER 8.1 + 有效 license。
- 可以跑在任何有 MATLAB 的机器上（Windows 或 Linux）。训练端只通过环境变量 `OPF_VERIFIER_URL` 使用它。
- verifier 会在引擎启动时屏蔽 gurobi/cplex/quadprog 等求解器，让 MATPOWER 的 `DEFAULT` 落到 MIPS，
  与 ProOPF 标签一致（不要去掉，否则 level-3 的标签复现不了）。
- 如果训练服务器上有 MATLAB：`pip install -r requirements-win.txt` 里的 `matlabengine==24.2.*`
  （版本须与 MATLAB 版本对应），并用 `MATPOWER_DIR=/path/to/matpower8.1` 指定 MATPOWER 路径
  （默认值是 Windows 路径 `E:\app\matlab\bin\matpower8.1`）。
- 如果训练服务器没有 MATLAB：在另一台有 MATLAB 的机器上起 verifier，保证网络可达（端口转发 / SSH 隧道等），例如
  `ssh -N -L 8765:127.0.0.1:8765 <matlab-host>` 后用 `OPF_VERIFIER_URL=http://127.0.0.1:8765`。

**安全注意**：verifier 没有认证，会执行收到的任意 MATLAB 代码（`harness.py` 只有一个文本黑名单，挡不住所有危险调用），
而且执行的是模型生成的代码。不要绑定到公网或共享网段（`0.0.0.0`）；只绑 `127.0.0.1` 并通过 SSH 隧道访问，
最好用低权限账户运行 MATLAB。

启动与自检：

```bash
python scripts/start_verifier.py --host 127.0.0.1 --port 8765 --engines 4 # 在 MATLAB 机器上，只绑本机，经 SSH 隧道访问
python scripts/check_env.py --verifier-url http://<verifier>:8765         # 在训练机器上
python scripts/verify_benchmark.py --url http://<verifier>:8765           # 期望 210/216 复现（6 个已知例外见 opf_sirl/README）
```

## 服务器环境（Linux 训练端）

以下命令都在 `opf_sirl/` 下执行。

```bash
cd OPF/opf_sirl
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-linux.txt      # vllm 0.9.2 / verl 0.5.0 / trl 0.19.1 / transformers 4.53.3
pip install <flash-attn 2.8.3 预编译 wheel，见 requirements-linux.txt 注释>
pip install -e .                           # 必须：verl 按路径加载 reward_fn.py，需要已安装的 opf_sirl 包
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
python scripts/check_env.py --verifier-url $OPF_VERIFIER_URL
pytest -m "not matlab"                     # 纯 Python 单测
```

注意：
- torch 的 CUDA 版本要和服务器驱动匹配；cu128 是为 RTX 50xx 选的，其他卡可换 cu12x 对应 wheel，flash-attn wheel 也要随之匹配。
- `verl_server_7b.sh` 默认 `VLLM_ATTENTION_BACKEND=XFORMERS`、8 卡、`tensor_model_parallel_size=4`、`trainer.logger=[console,wandb]`
  （没有 wandb 就追加覆盖 `trainer.logger='[console]'`）。

## 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `OPF_VERIFIER_URL` | verifier 地址（训练/评测/合成都用） | `http://127.0.0.1:8765`（server 脚本中为必填） |
| `OPF_REWARD_STAGE` | SIRL 奖励阶段 1 或 2 | 1 |
| `OPF_PARTIAL_KL` | 1 = Partial-KL，0 = 原生全量 KL | 1 |
| `OPF_PARTIAL_KL_TAG` | KL 区域起始标签（`matlab` 跟 SIRL 代码；`model` 跟论文） | matlab |
| `MODEL_PATH` / `DATA_DIR` / `CKPT_DIR` | verl 脚本中的模型、数据、checkpoint 路径 | `models/...` / `data` / `outputs/verl` |
| `PROOPF_ROOT` | ProOPF 基准位置 | `../ProOPF_benchmark` |
| `OPF_DATA_DIR` | 数据目录 | `opf_sirl/data` |
| `MATPOWER_DIR` | MATPOWER 路径（仅 verifier 机器） | `E:\app\matlab\bin\matpower8.1` |
| `OPF_NUM_ENGINES` / `OPF_EXEC_TIMEOUT` | verifier 引擎数 / 单程序超时秒数 | 2 / 120 |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_NAME` | 数据合成用的 OpenAI 兼容接口 | 无 |

## 当前进度（2026-11-07）

已完成：
1. 代码框架、两套训练后端、奖励与 verifier 全部实现。
2. `verify_benchmark.py` 在 ProOPF_B（含 OOD）上 210/216 复现标签。
3. `data/seeds_labelled.jsonl`：ProOPF_D 种子中能运行且收敛的 48 条。
4. 本地（RTX 5060 8GB, WSL2, Qwen2.5-0.5B）verl 与 TRL 两个后端各跑通 3 步 smoke test；0.5B 模型奖励全为 0，学习效果需在服务器上用 7B 验证。

待做（服务器上的顺序）：
```bash
cd opf_sirl
# 1. 合成训练数据（需 verifier + LLM API）；case59/68/189 是 OOD 系统，脚本已自动排除
source ~/.opf_llm_env && python scripts/synthesize.py --levels 1 2 3 4 --n 500 --url $OPF_VERIFIER_URL
# 2. 生成 parquet
python scripts/build_dataset.py data/seeds_labelled.jsonl data/synthetic_L*.jsonl
# 3. 训练前基线
python scripts/evaluate.py --model models/Qwen2.5-7B-Instruct --verifier-url $OPF_VERIFIER_URL --out outputs/eval_base
# 4. stage-1 训练
OPF_VERIFIER_URL=... bash configs/verl_server_7b.sh
# 5. 合并 stage-1 checkpoint（python -m verl.model_merger merge --backend fsdp ...），再 stage-2
OPF_REWARD_STAGE=2 MODEL_PATH=<merged stage-1> OPF_VERIFIER_URL=... bash configs/verl_server_7b.sh
# 6. 评测训练后模型
python scripts/evaluate.py --model <merged checkpoint> --verifier-url $OPF_VERIFIER_URL --out outputs/eval_stage2
```

合成数据和 parquet 体积不大，生成后可以提交回仓库，方便复现。
