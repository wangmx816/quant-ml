#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为单只股票构造分类特征与标签（应变量 0/1）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 基础价量特征
BASE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "volatility_20",
    "ma_ratio_5_20",
    "vol_ratio_5_20",
    "rsi_14",
    "high_low_range",
    "pct_chg",
]

OPTIONAL_VALUATION = ["pe_ttm", "pb", "ps_ttm", "total_mv"]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_classification_frame(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    trend_mode: bool = False,
) -> pd.DataFrame:
    """
    输入日线，输出带特征与 Label 的面板。

    Parameters
    ----------
    horizon : 前瞻收益天数。1=次日；5=未来5日收益（更偏趋势、持仓更长）。
    trend_mode : 是否加入中期趋势特征（20日动量、10/60均线比等）。
    """
    out = df.sort_values("trade_date").copy().reset_index(drop=True)
    close = out["close"]
    vol = out["volume"]

    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)
    out["volatility_20"] = out["ret_1d"].rolling(20).std()
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    out["ma_ratio_5_20"] = ma5 / ma20.replace(0, np.nan)
    out["ma_ratio_10_60"] = ma10 / ma60.replace(0, np.nan)
    out["price_to_ma60"] = close / ma60.replace(0, np.nan)
    # 近20日上涨日占比，刻画趋势持续性
    out["trend_persistence"] = (out["ret_1d"] > 0).astype(float).rolling(20).mean()
    v5 = vol.rolling(5).mean()
    v20 = vol.rolling(20).mean()
    out["vol_ratio_5_20"] = v5 / v20.replace(0, np.nan)
    out["rsi_14"] = _rsi(close, 14)
    out["high_low_range"] = (out["high"] - out["low"]) / close.replace(0, np.nan)
    if "pct_chg" not in out.columns or out["pct_chg"].isna().all():
        out["pct_chg"] = out["ret_1d"] * 100

    # 前瞻 horizon 日收益作为应变量
    out["Next_Ret"] = close.shift(-horizon) / close - 1.0
    out["Label"] = (out["Next_Ret"] > 0).astype(int)
    out["horizon"] = horizon

    feat_cols = list(BASE_FEATURES)
    if trend_mode:
        feat_cols.extend(["ret_20d", "ma_ratio_10_60", "price_to_ma60", "trend_persistence"])
    for c in OPTIONAL_VALUATION:
        if c in out.columns and out[c].notna().sum() > 30:
            if c == "total_mv":
                out["log_mv"] = np.log(out[c].replace(0, np.nan).abs())
                feat_cols.append("log_mv")
            else:
                feat_cols.append(c)

    # 去掉重复
    feat_cols = list(dict.fromkeys(feat_cols))
    use = out.dropna(subset=feat_cols + ["Label", "Next_Ret"]).copy()
    # 去掉末尾 horizon 行已在 dropna 处理
    use.attrs["feature_cols"] = feat_cols
    use.attrs["horizon"] = horizon
    use.attrs["trend_mode"] = trend_mode
    return use
