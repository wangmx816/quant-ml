#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按股票分别训练逻辑回归 / 决策树 / 随机森林，并评估 AUC、ROC。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data_fetch import STOCKS, load_all
from .features import build_classification_frame

ROOT = Path(__file__).resolve().parents[1]
CHART_DIR = ROOT / "output" / "charts"
METRICS_PATH = ROOT / "output" / "metrics.json"
PANEL_DIR = ROOT / "data" / "panels"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def build_models() -> dict:
    return {
        "逻辑回归": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "决策树": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=8,
            random_state=42,
            class_weight="balanced",
        ),
        "随机森林": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def train_one_stock(df: pd.DataFrame, symbol: str) -> dict:
    panel = build_classification_frame(df)
    feature_cols = panel.attrs["feature_cols"]
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_DIR / f"{symbol}_panel.csv", index=False, encoding="utf-8-sig")

    X = panel[feature_cols].values
    y = panel["Label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, model in build_models().items():
        cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "auc": float(roc_auc_score(y_test, y_prob)),
            "cv_auc_mean": float(cv_aucs.mean()),
            "cv_auc_std": float(cv_aucs.std()),
            "roc_auc_trapz": float(auc(fpr, tpr)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    best_name = max(results, key=lambda k: results[k]["auc"])
    return {
        "symbol": symbol,
        "name": STOCKS[symbol]["name"],
        "ts_code": STOCKS[symbol]["ts_code"],
        "n_samples": int(len(panel)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "pos_rate": float(y.mean()),
        "date_min": panel["trade_date"].min().strftime("%Y-%m-%d"),
        "date_max": panel["trade_date"].max().strftime("%Y-%m-%d"),
        "label_rule": "下一交易日收益 Next_Ret > 0 → 1，否则 → 0",
        "best_model": best_name,
        "best_auc": results[best_name]["auc"],
        "results": results,
    }


def plot_stock_roc(stock_meta: dict, out: Path) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测 (AUC=0.50)")
    colors = {"逻辑回归": "#2563eb", "决策树": "#dc2626", "随机森林": "#16a34a"}
    for name, res in stock_meta["results"].items():
        ax.plot(
            res["fpr"],
            res["tpr"],
            color=colors[name],
            linewidth=2,
            label=f"{name} (AUC={res['auc']:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR")
    title = f"{stock_meta['name']}（{stock_meta['symbol']}）ROC 曲线"
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_auc_heatmap(all_meta: list[dict], out: Path) -> None:
    setup_matplotlib()
    models = ["逻辑回归", "决策树", "随机森林"]
    names = [m["name"] for m in all_meta]
    mat = np.array([[m["results"][md]["auc"] for md in models] for m in all_meta])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.4, vmax=0.75)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=11)
    ax.set_title("图1  五只股票 × 三类模型 测试集 AUC", fontsize=13, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_best_auc_bars(all_meta: list[dict], out: Path) -> None:
    setup_matplotlib()
    names = [f"{m['name']}\n{m['best_model']}" for m in all_meta]
    aucs = [m["best_auc"] for m in all_meta]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(names)), aucs, color="#2563eb")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("最佳模型 AUC")
    ax.set_title("图2  各股票最优分类模型 AUC", fontsize=13, pad=10)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_all() -> dict:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    stock_data = load_all()
    per_stock = []
    for symbol, df in stock_data.items():
        meta = train_one_stock(df, symbol)
        plot_stock_roc(meta, CHART_DIR / f"roc_{symbol}.png")
        per_stock.append(meta)
        print(
            f"{meta['name']}: n={meta['n_samples']}, "
            f"{meta['date_min']}~{meta['date_max']}, "
            f"best={meta['best_model']} AUC={meta['best_auc']:.3f}"
        )

    plot_auc_heatmap(per_stock, CHART_DIR / "fig1_auc_heatmap.png")
    plot_best_auc_bars(per_stock, CHART_DIR / "fig2_best_auc.png")

    # 汇总表：取金风科技作报告主图示例 + 全市场对比
    payload = {
        "date_range": "2024-03-31 ~ 2026-06-30",
        "split": "各股票独立建模；train_test_split(test_size=0.3, stratify=y, random_state=42)；5折CV",
        "stocks": per_stock,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"指标已写入 {METRICS_PATH}")
    return payload


if __name__ == "__main__":
    run_all()
