#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于分类模型的交易策略回测：按时间划分训练/测试，计算季度收益与核心指标。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from .data_fetch import STOCKS, load_all
from .features import build_classification_frame
from .train import build_models, setup_matplotlib

ROOT = Path(__file__).resolve().parents[1]
CHART_DIR = ROOT / "output" / "charts"
STRATEGY_PATH = ROOT / "output" / "strategy_metrics.json"


@dataclass
class StrategyConfig:
    train_ratio: float = 0.7
    prob_threshold: float = 0.55
    initial_capital: float = 100_000.0
    commission: float = 0.0003
    slippage: float = 0.0001


def time_split(panel: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.sort_values("trade_date").reset_index(drop=True)
    cut = int(len(panel) * train_ratio)
    cut = max(cut, 30)
    cut = min(cut, len(panel) - 20)
    return panel.iloc[:cut].copy(), panel.iloc[cut:].copy()


def simulate_strategy(
    test: pd.DataFrame,
    positions: np.ndarray,
    cfg: StrategyConfig,
) -> dict[str, Any]:
    """根据预测仓位（0/1）在测试集上模拟交易。"""
    cash = cfg.initial_capital
    shares = 0.0
    equity_rows = []
    trades = []

    dates = pd.to_datetime(test["trade_date"].values)
    closes = test["close"].astype(float).values

    for i in range(len(test)):
        price = closes[i]
        # 当日收盘按“昨日信号”调仓：positions[i] 表示当日应持有仓位
        target = int(positions[i])
        buy_rate = 1 + cfg.commission + cfg.slippage
        sell_rate = 1 - cfg.commission - cfg.slippage

        if target == 1 and shares == 0:
            exec_price = price * buy_rate
            qty = int(cash // exec_price)
            if qty > 0:
                cash -= qty * exec_price
                shares = qty
                trades.append(
                    {
                        "trade_date": dates[i].strftime("%Y-%m-%d"),
                        "action": "BUY",
                        "price": round(price, 4),
                    }
                )
        elif target == 0 and shares > 0:
            exec_price = price * sell_rate
            cash += shares * exec_price
            trades.append(
                {
                    "trade_date": dates[i].strftime("%Y-%m-%d"),
                    "action": "SELL",
                    "price": round(price, 4),
                }
            )
            shares = 0.0

        equity = cash + shares * price
        equity_rows.append(
            {
                "trade_date": dates[i],
                "equity": equity,
                "close": price,
                "position": target,
                "net_value": equity / cfg.initial_capital,
            }
        )

    if shares > 0:
        exec_price = closes[-1] * (1 - cfg.commission - cfg.slippage)
        cash += shares * exec_price
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["net_value"] = cash / cfg.initial_capital
        equity_rows[-1]["position"] = 0
        trades.append(
            {
                "trade_date": dates[-1].strftime("%Y-%m-%d"),
                "action": "SELL",
                "price": round(float(closes[-1]), 4),
            }
        )

    eq = pd.DataFrame(equity_rows)
    eq["drawdown"] = eq["net_value"] / eq["net_value"].cummax() - 1
    eq["benchmark_nv"] = eq["close"] / eq["close"].iloc[0]
    return {"equity": eq, "trades": trades}


def compute_core_metrics(eq: pd.DataFrame, trades: list[dict]) -> dict[str, float]:
    nv = eq["net_value"]
    daily_ret = nv.pct_change().fillna(0)
    n_days = len(eq)
    years = max(n_days / 252, 1 / 252)
    cum = (nv.iloc[-1] - 1) * 100
    ann = ((nv.iloc[-1]) ** (1 / years) - 1) * 100 if nv.iloc[-1] > 0 else -100.0
    max_dd = float(eq["drawdown"].min() * 100)
    rf = 0.02 / 252
    excess = daily_ret - rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    bench = eq["benchmark_nv"]
    bench_cum = (bench.iloc[-1] - 1) * 100
    sells = [t for t in trades if t["action"] == "SELL"]
    return {
        "cumulative_return": round(float(cum), 2),
        "annualized_return": round(float(ann), 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "trade_count": len(sells),
        "benchmark_cumulative": round(float(bench_cum), 2),
        "excess_return": round(float(cum - bench_cum), 2),
        "final_net_value": round(float(nv.iloc[-1]), 4),
    }


def quarterly_returns(eq: pd.DataFrame) -> list[dict[str, Any]]:
    tmp = eq.copy()
    tmp["quarter"] = tmp["trade_date"].dt.to_period("Q").astype(str)
    rows = []
    for q, g in tmp.groupby("quarter", sort=True):
        r = g["net_value"].iloc[-1] / g["net_value"].iloc[0] - 1
        br = g["benchmark_nv"].iloc[-1] / g["benchmark_nv"].iloc[0] - 1
        rows.append(
            {
                "quarter": q,
                "strategy_return": round(float(r) * 100, 2),
                "benchmark_return": round(float(br) * 100, 2),
            }
        )
    return rows


def run_stock_strategy(df: pd.DataFrame, symbol: str, cfg: StrategyConfig | None = None) -> dict:
    cfg = cfg or StrategyConfig()
    panel = build_classification_frame(df)
    feature_cols = panel.attrs["feature_cols"]
    train, test = time_split(panel, cfg.train_ratio)

    X_train = train[feature_cols].values
    y_train = train["Label"].values
    X_test = test[feature_cols].values

    model_results = {}
    for name, base in build_models().items():
        model = clone(base)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        # 信号：当日预测用于次日仓位；首日空仓
        raw_pos = (proba >= cfg.prob_threshold).astype(int)
        positions = np.concatenate([[0], raw_pos[:-1]])
        sim = simulate_strategy(test, positions, cfg)
        metrics = compute_core_metrics(sim["equity"], sim["trades"])
        qrets = quarterly_returns(sim["equity"])
        eq = sim["equity"]
        model_results[name] = {
            "metrics": metrics,
            "quarterly": qrets,
            "equity": [
                {
                    "d": r.trade_date.strftime("%Y-%m-%d"),
                    "nv": round(float(r.net_value), 4),
                    "bench": round(float(r.benchmark_nv), 4),
                    "pos": int(r.position),
                }
                for r in eq.itertuples()
            ],
            "trade_count": metrics["trade_count"],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }

    # 选夏普最高者为主策略
    best_name = max(
        model_results.keys(),
        key=lambda k: model_results[k]["metrics"]["sharpe_ratio"],
    )
    return {
        "symbol": symbol,
        "name": STOCKS[symbol]["name"],
        "ts_code": STOCKS[symbol]["ts_code"],
        "feature_cols": feature_cols,
        "train_start": train["trade_date"].min().strftime("%Y-%m-%d"),
        "train_end": train["trade_date"].max().strftime("%Y-%m-%d"),
        "test_start": test["trade_date"].min().strftime("%Y-%m-%d"),
        "test_end": test["trade_date"].max().strftime("%Y-%m-%d"),
        "label_rule": "下一交易日收益 > 0 → 1；预测概率≥阈值则次日持有多头",
        "prob_threshold": cfg.prob_threshold,
        "best_model": best_name,
        "models": model_results,
    }


def plot_equity_compare(stock_res: dict, out: Path) -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = {"逻辑回归": "#2563eb", "决策树": "#dc2626", "随机森林": "#16a34a"}
    first = True
    for name, res in stock_res["models"].items():
        xs = [p["d"] for p in res["equity"]]
        ys = [p["nv"] for p in res["equity"]]
        ax.plot(xs, ys, color=colors[name], linewidth=1.6, label=name)
        if first:
            ax.plot(
                xs,
                [p["bench"] for p in res["equity"]],
                color="#94a3b8",
                linewidth=1.2,
                linestyle="--",
                label="买入持有",
            )
            first = False
    ax.set_title(f"图1  {stock_res['name']}测试集净值曲线（模型对比）", fontsize=13, pad=10)
    ax.set_ylabel("净值")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    # 稀疏刻度
    step = max(len(xs) // 6, 1)
    ax.set_xticks(xs[::step])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_quarterly(stock_res: dict, model_name: str, out: Path) -> None:
    setup_matplotlib()
    q = stock_res["models"][model_name]["quarterly"]
    labels = [r["quarter"] for r in q]
    sret = [r["strategy_return"] for r in q]
    bret = [r["benchmark_return"] for r in q]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, sret, w, label="策略", color="#0ea5e9")
    ax.bar(x + w / 2, bret, w, label="买入持有", color="#94a3b8")
    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("季度收益率 (%)")
    ax.set_title(
        f"图2  {stock_res['name']}·{model_name} 测试集分季度收益率",
        fontsize=13,
        pad=10,
    )
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_model_metrics_bars(all_res: list[dict], out: Path) -> None:
    setup_matplotlib()
    # 以金风科技为例，三模型核心指标对比；同时画五股最佳夏普
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    gw = next(r for r in all_res if r["symbol"] == "002202")
    models = list(gw["models"].keys())
    sharpes = [gw["models"][m]["metrics"]["sharpe_ratio"] for m in models]
    anns = [gw["models"][m]["metrics"]["annualized_return"] for m in models]
    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w / 2, sharpes, w, label="夏普", color="#2563eb")
    axes[0].bar(x + w / 2, anns, w, label="年化收益(%)", color="#f59e0b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_title("图3  金风科技：模型夏普与年化收益", fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    names = [r["name"] for r in all_res]
    best_s = [r["models"][r["best_model"]]["metrics"]["sharpe_ratio"] for r in all_res]
    axes[1].bar(range(len(names)), best_s, color="#16a34a")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_title("图4  五只股票最优策略夏普比率", fontsize=12)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_bonus_portfolio(all_res: list[dict]) -> dict:
    """附加题：五只股票等权组合（各股取其最优模型净值再平均）。"""
    # 对齐日期取交集
    series = {}
    for r in all_res:
        best = r["best_model"]
        eq = r["models"][best]["equity"]
        s = pd.Series({p["d"]: p["nv"] for p in eq}, name=r["symbol"])
        series[r["symbol"]] = s
    df = pd.DataFrame(series).dropna(how="any")
    if df.empty:
        return {"note": "无对齐样本", "equity": [], "metrics": {}}
    port = df.mean(axis=1)
    port = port / port.iloc[0]
    # 简化基准：五股买入持有等权（用各股 bench 再平均较复杂，这里用 port 对照单股平均收益）
    # 用每只股票 equity 中的 bench 对齐
    benches = []
    for r in all_res:
        best = r["best_model"]
        eq = r["models"][best]["equity"]
        benches.append(pd.Series({p["d"]: p["bench"] for p in eq}))
    bdf = pd.DataFrame(benches).T.reindex(df.index).dropna(how="any")
    bench = bdf.mean(axis=1)
    bench = bench / bench.iloc[0]
    eq = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(port.index),
            "net_value": port.values,
            "benchmark_nv": bench.reindex(port.index).values,
            "close": port.values,  # placeholder
            "position": 1,
        }
    )
    eq["drawdown"] = eq["net_value"] / eq["net_value"].cummax() - 1
    metrics = compute_core_metrics(eq, [])
    qrets = quarterly_returns(eq)
    return {
        "name": "五股等权·最优模型组合",
        "metrics": metrics,
        "quarterly": qrets,
        "equity": [
            {
                "d": d.strftime("%Y-%m-%d"),
                "nv": round(float(nv), 4),
                "bench": round(float(b), 4),
            }
            for d, nv, b in zip(eq["trade_date"], eq["net_value"], eq["benchmark_nv"])
        ],
    }


def plot_bonus(bonus: dict, out: Path) -> None:
    if not bonus.get("equity"):
        return
    setup_matplotlib()
    xs = [p["d"] for p in bonus["equity"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, [p["nv"] for p in bonus["equity"]], color="#7c3aed", linewidth=1.8, label="等权组合")
    ax.plot(
        xs,
        [p["bench"] for p in bonus["equity"]],
        color="#94a3b8",
        linestyle="--",
        linewidth=1.2,
        label="等权买入持有",
    )
    ax.set_title("图5  附加题：五股等权最优模型组合净值", fontsize=13, pad=10)
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    step = max(len(xs) // 6, 1)
    ax.set_xticks(xs[::step])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_all_strategies(cfg: StrategyConfig | None = None) -> dict:
    cfg = cfg or StrategyConfig()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    stock_data = load_all()
    results = []
    for symbol, df in stock_data.items():
        res = run_stock_strategy(df, symbol, cfg)
        results.append(res)
        print(
            f"{res['name']}: best={res['best_model']} "
            f"sharpe={res['models'][res['best_model']]['metrics']['sharpe_ratio']} "
            f"ann={res['models'][res['best_model']]['metrics']['annualized_return']}%"
        )

    # 图表：以金风为主
    gw = next(r for r in results if r["symbol"] == "002202")
    plot_equity_compare(gw, CHART_DIR / "task6_fig1_equity.png")
    plot_quarterly(gw, gw["best_model"], CHART_DIR / "task6_fig2_quarterly.png")
    plot_model_metrics_bars(results, CHART_DIR / "task6_fig3_compare.png")

    bonus = run_bonus_portfolio(results)
    plot_bonus(bonus, CHART_DIR / "task6_fig5_bonus.png")

    payload = {
        "config": {
            "train_ratio": cfg.train_ratio,
            "prob_threshold": cfg.prob_threshold,
            "commission": cfg.commission,
            "slippage": cfg.slippage,
        },
        "stocks": results,
        "bonus": bonus,
    }
    STRATEGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STRATEGY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"策略指标已写入 {STRATEGY_PATH}")
    return payload


if __name__ == "__main__":
    run_all_strategies()
