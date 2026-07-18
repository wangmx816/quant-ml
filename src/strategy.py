#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于分类模型的交易策略回测：基线版 vs 改进版（降阈值/拉长持仓/趋势标签）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    name: str = "baseline"
    train_ratio: float = 0.7
    prob_threshold: float = 0.55
    horizon: int = 1
    trend_mode: bool = False
    min_hold_days: int = 1
    initial_capital: float = 100_000.0
    commission: float = 0.0003
    slippage: float = 0.0001


# 基线：阈值偏严、次日标签、无最低持仓
BASELINE_CFG = StrategyConfig(
    name="baseline",
    prob_threshold=0.55,
    horizon=1,
    trend_mode=False,
    min_hold_days=1,
)

# 改进：默认阈值0.5；趋势向上时放宽至0.45；5日趋势标签；最低持仓5日
IMPROVED_CFG = StrategyConfig(
    name="improved",
    prob_threshold=0.50,
    horizon=5,
    trend_mode=True,
    min_hold_days=5,
)


def time_split(panel: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.sort_values("trade_date").reset_index(drop=True)
    cut = int(len(panel) * train_ratio)
    cut = max(cut, 30)
    cut = min(cut, len(panel) - 20)
    return panel.iloc[:cut].copy(), panel.iloc[cut:].copy()


def apply_min_hold(raw_pos: np.ndarray, min_hold: int) -> np.ndarray:
    """开仓后至少持有 min_hold 天，减少频繁交易。"""
    if min_hold <= 1:
        return raw_pos.astype(int)
    out = np.zeros(len(raw_pos), dtype=int)
    i = 0
    n = len(raw_pos)
    while i < n:
        if raw_pos[i] == 1:
            end = min(i + min_hold, n)
            out[i:end] = 1
            i = end
        else:
            i += 1
    return out


def simulate_strategy(
    test: pd.DataFrame,
    positions: np.ndarray,
    cfg: StrategyConfig,
) -> dict[str, Any]:
    cash = cfg.initial_capital
    shares = 0.0
    equity_rows = []
    trades = []
    dates = pd.to_datetime(test["trade_date"].values)
    closes = test["close"].astype(float).values

    for i in range(len(test)):
        price = closes[i]
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
                    {"trade_date": dates[i].strftime("%Y-%m-%d"), "action": "BUY", "price": round(price, 4)}
                )
        elif target == 0 and shares > 0:
            exec_price = price * sell_rate
            cash += shares * exec_price
            trades.append(
                {"trade_date": dates[i].strftime("%Y-%m-%d"), "action": "SELL", "price": round(price, 4)}
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
    long_days = int((eq["position"] == 1).sum())
    return {
        "cumulative_return": round(float(cum), 2),
        "annualized_return": round(float(ann), 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "trade_count": len(sells),
        "benchmark_cumulative": round(float(bench_cum), 2),
        "excess_return": round(float(cum - bench_cum), 2),
        "final_net_value": round(float(nv.iloc[-1]), 4),
        "long_days": long_days,
        "long_ratio": round(long_days / max(len(eq), 1), 4),
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


def _pack_equity(eq: pd.DataFrame) -> list[dict]:
    return [
        {
            "d": r.trade_date.strftime("%Y-%m-%d"),
            "nv": round(float(r.net_value), 4),
            "bench": round(float(r.benchmark_nv), 4),
            "pos": int(r.position),
        }
        for r in eq.itertuples()
    ]


def run_stock_strategy(df: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> dict:
    panel = build_classification_frame(df, horizon=cfg.horizon, trend_mode=cfg.trend_mode)
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
        # 改进版：按概率分位数开仓（默认做多约50%交易日；趋势向上约60%），
        # 避免绝对概率整体偏低时几乎不开仓。基线版仍用绝对阈值。
        if cfg.trend_mode:
            thr_flat = float(np.quantile(proba, 0.50))
            thr_up = float(np.quantile(proba, 0.40))
            if "ma_ratio_10_60" in test.columns:
                thr = np.where(test["ma_ratio_10_60"].values > 1.0, thr_up, thr_flat)
            else:
                thr = thr_flat
            raw = (proba >= thr).astype(int)
        else:
            raw = (proba >= cfg.prob_threshold).astype(int)
        # 当日信号用于次日仓位
        shifted = np.concatenate([[0], raw[:-1]])
        positions = apply_min_hold(shifted, cfg.min_hold_days)
        sim = simulate_strategy(test, positions, cfg)
        metrics = compute_core_metrics(sim["equity"], sim["trades"])
        model_results[name] = {
            "metrics": metrics,
            "quarterly": quarterly_returns(sim["equity"]),
            "equity": _pack_equity(sim["equity"]),
            "trade_count": metrics["trade_count"],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }

    best_name = max(
        model_results.keys(),
        key=lambda k: (
            model_results[k]["metrics"]["sharpe_ratio"],
            model_results[k]["metrics"]["cumulative_return"],
        ),
    )
    label_rule = (
        f"未来{cfg.horizon}日收益>0→1；"
        + (
            "改进版按概率分位数开仓（约50%–60%交易日持仓，趋势向上更积极）"
            if cfg.trend_mode
            else f"绝对阈值≥{cfg.prob_threshold}"
        )
        + f"；最低持仓{cfg.min_hold_days}日"
    )
    return {
        "symbol": symbol,
        "name": STOCKS[symbol]["name"],
        "ts_code": STOCKS[symbol]["ts_code"],
        "version": cfg.name,
        "feature_cols": feature_cols,
        "train_start": train["trade_date"].min().strftime("%Y-%m-%d"),
        "train_end": train["trade_date"].max().strftime("%Y-%m-%d"),
        "test_start": test["trade_date"].min().strftime("%Y-%m-%d"),
        "test_end": test["trade_date"].max().strftime("%Y-%m-%d"),
        "label_rule": label_rule,
        "prob_threshold": cfg.prob_threshold,
        "horizon": cfg.horizon,
        "min_hold_days": cfg.min_hold_days,
        "trend_mode": cfg.trend_mode,
        "best_model": best_name,
        "models": model_results,
    }


def run_bonus_portfolio(all_res: list[dict]) -> dict:
    series, benches = {}, []
    for r in all_res:
        best = r["best_model"]
        eq = r["models"][best]["equity"]
        series[r["symbol"]] = pd.Series({p["d"]: p["nv"] for p in eq})
        benches.append(pd.Series({p["d"]: p["bench"] for p in eq}))
    df = pd.DataFrame(series).dropna(how="any")
    if df.empty:
        return {"note": "无对齐样本", "equity": [], "metrics": {}}
    port = df.mean(axis=1)
    port = port / port.iloc[0]
    bdf = pd.DataFrame(benches).T.reindex(df.index).dropna(how="any")
    bench = bdf.mean(axis=1)
    bench = bench / bench.iloc[0]
    # 组合平均持仓比例（各股 long_ratio 均值）
    avg_long = float(np.mean([r["models"][r["best_model"]]["metrics"]["long_ratio"] for r in all_res]))
    eq = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(port.index),
            "net_value": port.values,
            "benchmark_nv": bench.reindex(port.index).values,
            "close": port.values,
            "position": 1,
        }
    )
    eq["drawdown"] = eq["net_value"] / eq["net_value"].cummax() - 1
    metrics = compute_core_metrics(eq, [])
    metrics["avg_stock_long_ratio"] = round(avg_long, 4)
    return {
        "name": "五股等权·最优模型组合",
        "metrics": metrics,
        "quarterly": quarterly_returns(eq),
        "equity": [
            {
                "d": d.strftime("%Y-%m-%d"),
                "nv": round(float(nv), 4),
                "bench": round(float(b), 4),
            }
            for d, nv, b in zip(eq["trade_date"], eq["net_value"], eq["benchmark_nv"])
        ],
    }


def plot_version_equity(baseline_gw: dict, improved_gw: dict, out: Path) -> None:
    """金风：基线最优 vs 改进最优 vs 买入持有。"""
    setup_matplotlib()
    b_best = baseline_gw["models"][baseline_gw["best_model"]]
    i_best = improved_gw["models"][improved_gw["best_model"]]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    xs_b = [p["d"] for p in b_best["equity"]]
    xs_i = [p["d"] for p in i_best["equity"]]
    ax.plot(xs_b, [p["nv"] for p in b_best["equity"]], color="#dc2626", lw=1.6,
            label=f"基线·{baseline_gw['best_model']} (阈值0.55/次日)")
    ax.plot(xs_i, [p["nv"] for p in i_best["equity"]], color="#2563eb", lw=1.8,
            label=f"改进·{improved_gw['best_model']} (阈值0.5/5日趋势)")
    ax.plot(xs_b, [p["bench"] for p in b_best["equity"]], color="#94a3b8", ls="--", lw=1.2, label="买入持有")
    ax.set_title("图6  金风科技：基线策略 vs 改进策略 vs 买入持有", fontsize=13, pad=10)
    ax.set_ylabel("净值")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, ls="--", alpha=0.35)
    step = max(len(xs_b) // 6, 1)
    ax.set_xticks(xs_b[::step])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bonus_compare(b0: dict, b1: dict, out: Path) -> None:
    if not b0.get("equity") or not b1.get("equity"):
        return
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    xs0 = [p["d"] for p in b0["equity"]]
    xs1 = [p["d"] for p in b1["equity"]]
    ax.plot(xs0, [p["nv"] for p in b0["equity"]], color="#dc2626", lw=1.5, label="等权组合·基线")
    ax.plot(xs1, [p["nv"] for p in b1["equity"]], color="#7c3aed", lw=1.8, label="等权组合·改进")
    ax.plot(xs0, [p["bench"] for p in b0["equity"]], color="#94a3b8", ls="--", lw=1.2, label="等权买入持有")
    ax.set_title("图7  五股等权组合：基线 vs 改进 vs 等权买入持有", fontsize=13, pad=10)
    ax.set_ylabel("净值")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.35)
    step = max(len(xs0) // 6, 1)
    ax.set_xticks(xs0[::step])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_equity_compare(stock_res: dict, out: Path, fig_title: str = "图1") -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = {"逻辑回归": "#2563eb", "决策树": "#dc2626", "随机森林": "#16a34a"}
    first = True
    xs = []
    for name, res in stock_res["models"].items():
        xs = [p["d"] for p in res["equity"]]
        ax.plot(xs, [p["nv"] for p in res["equity"]], color=colors[name], linewidth=1.6, label=name)
        if first:
            ax.plot(xs, [p["bench"] for p in res["equity"]], color="#94a3b8", lw=1.2, ls="--", label="买入持有")
            first = False
    ax.set_title(f"{fig_title}  {stock_res['name']}测试集净值（{stock_res['version']}）", fontsize=13, pad=10)
    ax.set_ylabel("净值")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    step = max(len(xs) // 6, 1) if xs else 1
    if xs:
        ax.set_xticks(xs[::step])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_quarterly(stock_res: dict, model_name: str, out: Path, fig_title: str = "图2") -> None:
    setup_matplotlib()
    q = stock_res["models"][model_name]["quarterly"]
    labels = [r["quarter"] for r in q]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, [r["strategy_return"] for r in q], w, label="策略", color="#0ea5e9")
    ax.bar(x + w / 2, [r["benchmark_return"] for r in q], w, label="买入持有", color="#94a3b8")
    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("季度收益率 (%)")
    ax.set_title(f"{fig_title}  {stock_res['name']}·{model_name} 分季度收益（{stock_res['version']}）", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_model_metrics_bars(all_res: list[dict], out: Path) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    gw = next(r for r in all_res if r["symbol"] == "002202")
    models = list(gw["models"].keys())
    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w / 2, [gw["models"][m]["metrics"]["sharpe_ratio"] for m in models], w, label="夏普", color="#2563eb")
    axes[0].bar(x + w / 2, [gw["models"][m]["metrics"]["annualized_return"] for m in models], w, label="年化%", color="#f59e0b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_title(f"图3  金风科技模型对比（{gw['version']}）", fontsize=12)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", ls="--", alpha=0.35)

    names = [r["name"] for r in all_res]
    best_s = [r["models"][r["best_model"]]["metrics"]["sharpe_ratio"] for r in all_res]
    axes[1].bar(range(len(names)), best_s, color="#16a34a")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_title("图4  五股最优夏普", fontsize=12)
    axes[1].grid(True, axis="y", ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bonus(bonus: dict, out: Path, fig_title: str = "图5") -> None:
    if not bonus.get("equity"):
        return
    setup_matplotlib()
    xs = [p["d"] for p in bonus["equity"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, [p["nv"] for p in bonus["equity"]], color="#7c3aed", lw=1.8, label="等权组合")
    ax.plot(xs, [p["bench"] for p in bonus["equity"]], color="#94a3b8", ls="--", lw=1.2, label="等权买入持有")
    ax.set_title(f"{fig_title}  五股等权最优模型组合净值", fontsize=13, pad=10)
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.35)
    step = max(len(xs) // 6, 1)
    ax.set_xticks(xs[::step])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _run_one_version(stock_data: dict, cfg: StrategyConfig) -> dict:
    results = []
    for symbol, df in stock_data.items():
        res = run_stock_strategy(df, symbol, cfg)
        results.append(res)
        m = res["models"][res["best_model"]]["metrics"]
        print(
            f"[{cfg.name}] {res['name']}: best={res['best_model']} "
            f"cum={m['cumulative_return']}% long={m['long_ratio']:.1%} "
            f"excess={m['excess_return']}% sharpe={m['sharpe_ratio']}"
        )
    bonus = run_bonus_portfolio(results)
    return {"config": asdict(cfg), "stocks": results, "bonus": bonus}


def run_all_strategies() -> dict:
    """同时跑基线与改进版，写入对比结果。"""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    stock_data = load_all()

    baseline = _run_one_version(stock_data, BASELINE_CFG)
    improved = _run_one_version(stock_data, IMPROVED_CFG)

    # 主展示用改进版图表 + 对比图
    gw_b = next(s for s in baseline["stocks"] if s["symbol"] == "002202")
    gw_i = next(s for s in improved["stocks"] if s["symbol"] == "002202")

    plot_equity_compare(gw_b, CHART_DIR / "task6_fig1_equity.png", "图1")
    plot_quarterly(gw_b, gw_b["best_model"], CHART_DIR / "task6_fig2_quarterly.png", "图2")
    plot_model_metrics_bars(baseline["stocks"], CHART_DIR / "task6_fig3_compare.png")
    plot_bonus(baseline["bonus"], CHART_DIR / "task6_fig5_bonus.png", "图5")
    plot_version_equity(gw_b, gw_i, CHART_DIR / "task6_fig6_version_compare.png")
    plot_bonus_compare(baseline["bonus"], improved["bonus"], CHART_DIR / "task6_fig7_bonus_compare.png")
    plot_equity_compare(gw_i, CHART_DIR / "task6_fig8_improved_equity.png", "图8")

    # 分析摘要（写入 JSON 供报告使用）
    bb = gw_b["models"][gw_b["best_model"]]["metrics"]
    bi = gw_i["models"][gw_i["best_model"]]["metrics"]
    bon0 = baseline["bonus"]["metrics"]
    bon1 = improved["bonus"]["metrics"]
    analysis = {
        "goldwind_gap": {
            "test_period": f"{gw_b['test_start']} ~ {gw_b['test_end']}",
            "baseline_best": gw_b["best_model"],
            "baseline_cum": bb["cumulative_return"],
            "baseline_bench": bb["benchmark_cumulative"],
            "baseline_excess": bb["excess_return"],
            "baseline_long_ratio": bb["long_ratio"],
            "improved_best": gw_i["best_model"],
            "improved_cum": bi["cumulative_return"],
            "improved_bench": bi["benchmark_cumulative"],
            "improved_excess": bi["excess_return"],
            "improved_long_ratio": bi["long_ratio"],
        },
        "equal_weight_gap": {
            "baseline_port_cum": bon0.get("cumulative_return"),
            "baseline_bench_cum": bon0.get("benchmark_cumulative"),
            "baseline_excess": bon0.get("excess_return"),
            "baseline_avg_long": bon0.get("avg_stock_long_ratio"),
            "improved_port_cum": bon1.get("cumulative_return"),
            "improved_bench_cum": bon1.get("benchmark_cumulative"),
            "improved_excess": bon1.get("excess_return"),
            "improved_avg_long": bon1.get("avg_stock_long_ratio"),
        },
    }

    # 看板默认展示改进版，同时保留基线对比
    payload = {
        "active_version": "improved",
        "baseline": baseline,
        "improved": improved,
        "config": improved["config"],
        "stocks": improved["stocks"],
        "bonus": improved["bonus"],
        "analysis": analysis,
    }
    STRATEGY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"策略指标已写入 {STRATEGY_PATH}")
    return payload


if __name__ == "__main__":
    run_all_strategies()
