#!/usr/bin/env python3
"""Summarize fixed three-seed controls without selecting a favorable seed."""

import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reproduction_records/dicra_three_seed_controls_20261022"
DATA = FOLDER / "evaluation.json"
SEEDS = (20260922, 20260923, 20260924)
DATASETS = ("LOLv1", "Real", "Synthetic")
METHODS = ("baseline", "scalar", "no_decomposition", "dicra")
METRICS = ("psnr", "ssim", "lpips", "fid", "mae")


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def main():
    data = json.loads(DATA.read_text())
    if tuple(data["train_seeds"]) != SEEDS:
        raise AssertionError("Unexpected training seeds")
    summary = {"train_seeds": list(SEEDS), "datasets": {}, "notes": [
        "Three checkpoints were fixed at 4000 training steps; no test-set seed selection.",
        "Training-seed standard deviations use ddof=1 and n=3; they are descriptive, not significance tests.",
        "Historical Dev100 and official test sets were seen in earlier research; this is a robustness audit, not an independent blind test.",
    ]}
    lines = ["# DiCRA三训练种子同协议复验", "",
             "本报告只复验既有表9中的可训练标量、同容量无分解与完整DiCRA；训练种子固定为20260922、20260923、20260924，均采用385对训练、4,000步和各自第4,000步检查点。20260922沿用历史原始评价；新增两种子独立训练并按表9协议评价。未按测试结果选种子或调结构。", "",
             "表内为三个训练种子的均值±样本标准差（n=3）；不是显著性检验。FID按各训练种子在相同集合口径下计算后汇总。", ""]
    for dataset in DATASETS:
        lines.extend([f"## {dataset}", "",
                      "| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | MAE↓ |",
                      "|---|---:|---:|---:|---:|---:|"])
        entry = {"methods": {}, "dicra_minus": {}}
        for method in METHODS:
            values = {metric: np.asarray([
                data["results"][str(seed)]["datasets"][dataset]["means"][method][metric]
                for seed in SEEDS], dtype=np.float64) for metric in METRICS}
            entry["methods"][method] = {metric: {
                "per_seed": vals.tolist(), "mean": float(vals.mean()),
                "sample_std": float(vals.std(ddof=1))} for metric, vals in values.items()}
            cells = [f"{fmt(vals.mean(), 4)}±{fmt(vals.std(ddof=1), 4)}"
                     for vals in (values[metric] for metric in METRICS)]
            lines.append("| " + method + " | " + " | ".join(cells) + " |")
        lines.extend(["", "DiCRA减去对照的逐训练种子差值（正PSNR/SSIM、负LPIPS/FID/MAE为有利）：", "",
                      "| 对照 | 指标 | 20260922 | 20260923 | 20260924 | 均值±标准差 |",
                      "|---|---|---:|---:|---:|---:|"])
        for reference in ("scalar", "no_decomposition"):
            entry["dicra_minus"][reference] = {}
            for metric in METRICS:
                diffs = np.asarray([
                    data["results"][str(seed)]["datasets"][dataset]["means"]["dicra"][metric]
                    - data["results"][str(seed)]["datasets"][dataset]["means"][reference][metric]
                    for seed in SEEDS], dtype=np.float64)
                entry["dicra_minus"][reference][metric] = {
                    "per_seed": diffs.tolist(), "mean": float(diffs.mean()),
                    "sample_std": float(diffs.std(ddof=1))}
                lines.append("| " + reference + " | " + metric + " | "
                             + " | ".join(fmt(value, 4) for value in diffs) + " | "
                             + f"{fmt(diffs.mean(), 4)}±{fmt(diffs.std(ddof=1), 4)} |")
        lines.append("")
        summary["datasets"][dataset] = entry
    lines.extend([
        "## 解释范围", "",
        "这些结果检验既定结构排序在三个FREC训练种子下是否重复；训练种子数仍小，且LOL-v1官方eval15只有15对图像。开发集与官方测试集在历史研发中已被查看，不能称为新结构的独立盲测。表9的同容量无分解对照仍包含分量相关尺度与接受目标的变化，不能单独归因于31×31滤波算子。",
        "",
        "现有100张开发图像的门图—误差诊断见 `../dicra_acceptance_cpu_20261021/diagnostic.json`；该诊断未显示组合门A对候选方向的清晰区分，因此不把门图作为可靠错误检测的正面证据。",
        "",
    ])
    (FOLDER / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (FOLDER / "三训练种子复验汇总.md").write_text("\n".join(lines))
    print(FOLDER / "三训练种子复验汇总.md")


if __name__ == "__main__":
    main()
