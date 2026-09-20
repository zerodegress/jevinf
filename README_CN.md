# jevinf

Jev 这一系决策模型的推理引擎：每条候选路径按分段前向计算并复用前缀，上面架一层符合 Jev wire
契约的服务。目前接上的后端是 NanoJev。

> English version: [README.md](README.md)

实测倍数（MPS）：单请求口径（API 上限内）**2.57×**（25.80 s → 10.06 s），整 dev split **2.27×**
（84.2 s → 37.0 s），argmax 一致率 100%。同样两个形态在 CUDA 上分别是 **2.15×** 与 **2.07×**——倍数
来自前向的排法，不来自设备。前向怎么排、这些数字背后的形态依赖，都在
[策略与实测](docs/strategies.md)。

## 快速上手

凡是要加载权重的命令都显式给 `-m/--model`（checkpoint 目录）；要读评测集的命令再给 `--split`
（jsonl 文件）。

```bash
uv sync
uv run jevinf selfcheck -m models/NanoJev --split data/dev.jsonl --states 4
uv run jevinf bench -m models/NanoJev --split data/dev.jsonl --states 8
uv run jevinf bench -m models/NanoJev --split data/dev.jsonl --out data/report.json
uv run jevinf eval -m models/NanoJev --input data/request.json
uv run jevinf oracle -m models/NanoJev --split data/dev.jsonl --out data/base.json
uv run jevinf eval   -m models/NanoJev --split data/dev.jsonl --out data/mine.json
uv run jevinf compare --reference data/base.json --candidate data/mine.json
```

`jevinf serve` 同时提供两个入口：Jev 兼容的 `/v1/systemone` 与原生调试的 `/api/evaluate`。

```bash
uv run jevinf serve -m models/NanoJev --port 8226            # 默认 fused_state；Jev 别名 jev-latest
uv run jevinf serve -m models/NanoJev --api-key sk-local     # /v1/* 开启 HTTPBearer 强制
```

```bash
# 官方 SDK 或任何 Jev 客户端指向它就能用
TYPESAFE_BASE_URL=http://127.0.0.1:8226 TYPESAFE_API_KEY=sk-any uv run python your_jev_client.py
```

## 开发环境建立

需要 Python ≥3.14 与 [uv](https://docs.astral.sh/uv/)。依赖声明在 `pyproject.toml`、锁定在
`uv.lock`（torch、transformers、fastapi、uvicorn、safetensors；`typesafe-sdk` 只是 dev 依赖，
供一致性测试使用）。

```bash
uv sync                # 依 uv.lock 建立 .venv
uv run jevinf --help
```

**项目与 `.venv` 都必须放在支持符号链接与 POSIX 权限的文件系统上。** 缺这两样的文件系统（例如
ExFAT）放 uv 的 `.venv` 会坏 —— 构建产物也放普通本地盘。本原型针对 Apple silicon（MPS）与统一
内存调优。

后端由 `--backend` 选择：

| 后端 | 运行在 | 状态 |
|---|---|---|
| `torch-mps` | Apple silicon，经 Metal | 默认 |
| `torch-cpu` | 纯 CPU | 已声明 |
| `torch-cuda` | NVIDIA，经 CUDA | 已接通 |
| `torch-rocm` | AMD，经 ROCm | 已声明 |

`torch-mps` 与 `torch-cuda` 已接通；`torch-cpu` 与 `torch-rocm` 会拒绝运行。文档里所有实测数字都是
在 MPS 上测的——CUDA 机器需要自己重测一遍。

模型架构由 `--arch` 选择：

| 架构 | 骨干 | 状态 |
|---|---|---|
| `nanojev` | Qwen3-0.6B 因果解码器 + 训练好的决策头 | 默认，三阶段前缀共享 |
| `decider-2b` | Qwen3.5-2B 因果解码器，每 4 层有 3 层线性注意力（KV + 递归混血缓存） | 已接通，一份 state 前缀 fork 给每题一行 |
| `laya` | ModernBERT-large 编码器 + 决策头，每题一段序列 | 已接通，一题一行（`laya.py`） |

三个架构都已接通。各架构的**结构性事实**（注意力怎么走、答案从哪里
读出、段与段之间要驻留什么）记在 `src/jevinf/arch.py` —— 用哪套编排就是由这些事实决定的（三阶段那套在
`engine.py`，fork 那套在 `decider.py`，单路径那套在 `laya.py`）。

检查脚本，由省到费：

```bash
uv run jevinf selfcheck -m models/NanoJev --split data/dev.jsonl --states 4   # 只验决策头
uv run python scripts/decider_parity.py --model models/decider-2b             # 与 decider 包对拍
uv run python scripts/laya_parity.py --model models/laya                      # 与官方推理入口对拍
uv run jevinf serve -m models/NanoJev --port 8226 &                          # 需要活服务
uv run python scripts/jev_conformance.py --base http://127.0.0.1:8226        # 官方 SDK 一致性验收
uv run python scripts/api_smoke.py --split data/dev.jsonl                    # golden 对拍
```

每个家族都有一个「与它自己的官方实现逐题对拍」的验收脚本（官方代码就放在权重旁边）：
`scripts/decider_parity.py` 对 `decider` 包，`scripts/laya_parity.py` 对 `rl_agent_api.RLAgent`。

## 文档

- [策略与实测](docs/strategies.md) — 前向如何排布、各策略倍数、形态依赖、成本模型
- [Jev 兼容服务层](docs/jev-api.md) — wire 契约、翻译层、一致性验收
- [原生调试入口](docs/native-api.md) — `/api/evaluate` 契约、旋钮、上限
- [内部细节](docs/internals.md) — 设计纪律、等价性证据、环境事实、已知坑
