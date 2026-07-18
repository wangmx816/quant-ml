#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为单只股票构造分类特征与标签（应变量 0/1）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
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


def build_classification_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入日线，输出带特征与 Label 的面板。
    Label: 下一交易日收益 Next_Ret > 0 → 1，否则 → 0。
    """
    out = df.sort_values("trade_date").copy().reset_index(drop=True)
    close = out["close"]
    vol = out["volume"]

    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["volatility_20"] = out["ret_1d"].rolling(20).std()
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    out["ma_ratio_5_20"] = ma5 / ma20.replace(0, np.nan)
    v5 = vol.rolling(5).mean()
    v20 = vol.rolling(20).mean()
    out["vol_ratio_5_20"] = v5 / v20.replace(0, np.nan)
    out["rsi_14"] = _rsi(close, 14)
    out["high_low_range"] = (out["high"] - out["low"]) / close.replace(0, np.nan)
    if "pct_chg" not in out.columns or out["pct_chg"].isna().all():
        out["pct_chg"] = out["ret_1d"] * 100

    # 下一日收益作为应变量基础
    out["Next_Ret"] = close.pct_change(1).shift(-1)
    out["Label"] = (out["Next_Ret"] > 0).astype(int)

    feat_cols = list(FEATURE_COLS)
    for c in OPTIONAL_VALUATION:
        if c in out.columns and out[c].notna().sum() > 30:
            # 市值取对数更稳
            if c == "total_mv":
                out["log_mv"] = np.log(out[c].replace(0, np.nan).abs())
                feat_cols.append("log_mv")
            else:
                feat_cols.append(c)

    use = out.dropna(subset=feat_cols + ["Label", "Next_Ret"]).copy()
    use.attrs["feature_cols"] = feat_cols
    return use
