# NLL Compute — Aggregation & Visualization (NLL 聚合与可视化)

This module consumes raw NLL JSON output produced by the **StreamMUSE** `nll_compute` module
and provides aggregation statistics and heatmap visualization tools.
*(本模块读取由 StreamMUSE 的 `nll_compute` 模块输出的原始 NLL JSON 文件，提供统计聚合和热力图可视化工具。)*

> **⚠️ SPLIT DESIGN (分离式架构)**
> **Compute** (running the model and producing raw JSONs) lives in `StreamMUSE/nll_compute/`.
> **Aggregation & Plotting** (this module) lives here in the `eval` repository.
> *(计算部分（跑模型生成原始 JSON）在 StreamMUSE 仓库。聚合与画图部分（本模块）在 eval 仓库。)*

## Structure (模块结构)

| File | Description |
|---|---|
| `aggregate.py` | `aggregate_nll(json_path)` — reads a raw NLL JSON, calls `eval_toolkit.stats` to produce dataset-level stats |
| `plot_nll_heatmap.py` | Builds interval × gen_frame pivot tables and renders seaborn/matplotlib heatmaps |
| `runners/run_aggregate.py` | CLI wrapper for `aggregate_nll` |
| `runners/run_heatmap.py` | CLI wrapper for `plot_nll_heatmap` |

## Workflow (工作流)

```
StreamMUSE nll_compute          →   eval/src/nll_compute
────────────────────────────────────────────────────────
1. uv run python -m                 2. uv run python -m
   nll_compute.runners.                nll_compute.runners.
   run_cal_nll                         run_aggregate
   --midi_dir /path/to/midi            --input records/nll_runs/exp1.json
   --ckpt_path /path/model.ckpt        --output reports/summary.json
   --save_json output/exp1.json        --pretty

                                    3. uv run python -m
                                       nll_compute.runners.
                                       run_heatmap
                                       --input-dir records/nll_runs/experiments1
                                       --out reports/heatmap.png
                                       --value-mode weighted_avg --annotate
```

## Aggregation Output Format (聚合输出格式)

`aggregate_nll` returns a JSON-serializable dict with two keys:
- **`summary`**: dataset-level stats (files\_count, weighted\_avg\_nll, per\_file\_stats with mean/std/p25/p50/p75 etc.)
- **`per_file`**: per-file parsed values or error strings
