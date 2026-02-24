# NLL Compute Toolkit (NLL计算工具箱)

This module provides tools for calculating the Negative Log-Likelihood (NLL) of MIDI files under a given model, running batch evaluations, aggregating statistics, and plotting heatmaps.
本模块提供了一系列工具，用于计算指定模型下 MIDI 文件的负对数似然（NLL）、执行批量评估、聚合统计数据以及绘制热力图。

## Structure (模块结构)

- **`core.py`**: Core mathematical tensor logic and single-file NLL calculation.`cal_nll`
  *(核心张量逻辑与单文件 NLL 计算 `cal_nll`)*
- **`batch.py`**: Batch processing logic for computing metrics across entire directories.
  *(目录级别批量计算指标的逻辑处理)*
- **`aggregate.py`**: Metric aggregation logic mapping JSON file metrics to dataset-level statistics.
  *(指标聚合逻辑，将 JSON 文件的度量映射到数据集级别的统计信息)*
- **`plot.py`**: Data visualization logic building Pivot matrices into seaborn heatmaps.
  *(数据可视化逻辑，将数据透视表构建为 Seaborn 热力图)*

## Command Line Interface Runners (命令行入口)

Executable scripts are kept inside the `runners/` directory to preserve module purity. Usage instructions are available for each via the `--help` flag.
独立的可执行脚本放在 `runners/` 目录中以保持核心模块的纯粹性。可以通过 `--help` 标志查看每个指令的使用说明。

### 1. Evaluate an entire directory (评估整个目录)
Calculate average/total NLLs for a folder of generated `.mid` files.
*(计算生成的 `.mid` 文件夹的平均/总 NLL)*

```bash
python -m move_to_eval.nll_compute.runners.run_cal_nll \
  --midi_dir /path/to/midi/dir \
  --ckpt_path /path/to/model.ckpt \
  --save_json_path output/results.json \
  --window 384 --offset 128
```

### 2. Aggregate Results (聚合结果)
Combine metadata created by `run_cal_nll` into unified median, maximum, and weighted_average scores.
*(将在 `run_cal_nll` 中产生的元数据统计合并为一个统一的中位数、最大值和加权平均分数。)*

```bash
python -m move_to_eval.nll_compute.runners.run_aggregate \
  --input output/results.json \
  --output summary/summary.json --pretty
```

### 3. Plot Heatmap (绘制热力图)
Parse generated sub-directory metadata mapping variables (`interval` and `gen_frame`) to graphical nodes via pivot tables.
*(通过数据透视表解析生成的子目录映射变量（如 `interval` 和 `gen_frame`），在二维分布图中生成指标矩阵。)*

```bash
python -m move_to_eval.nll_compute.runners.run_heatmap \
  --input-dir records/nll_runs/experiments1 \
  --out heatmaps/chart.png \
  --csv-out heatmaps/chart.csv \
  --value-mode weighted_avg --annotate
```

### 4. Manifest Pipeline Orchestrator (元数据管线协调器)
Evaluate an array of output metrics declared within `midi_dirs.json` directly. Automatically aggregates data into combinations.
*(直接批量评估 `midi_dirs.json` 中声明的一系列测试目录组。会自动把数据进行组合汇算。)*

```bash
python -m move_to_eval.nll_compute.runners.run_from_manifest \
  --manifest records/midi_dirs.json \
  --ckpt /path/to/model.ckpt \
  --out records/nll_runs
```
