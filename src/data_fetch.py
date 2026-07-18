#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拉取五只股票 2024-03-31 ~ 2026-06-30 前复权日线（东方财富）。"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

START_DATE = datetime(2024, 3, 31)
END_DATE = datetime(2026, 6, 30)

STOCKS = {
    "002202": {"ts_code": "002202.SZ", "name": "金风科技", "market": 0},
    "600031": {"ts_code": "600031.SH", "name": "三一重工", "market": 1},
    "000425": {"ts_code": "000425.SZ", "name": "徐工机械", "market": 0},
    "600207": {"ts_code": "600207.SH", "name": "安彩高科", "market": 1},
    "000816": {"ts_code": "000816.SZ", "name": "智慧农业", "market": 0},
}

ADJUST_MAP = {"none": 0, "qfq": 1, "hfq": 2}


def _eastmoney_url(symbol: str, market: int, adjust: str = "qfq") -> str:
    fqt = ADJUST_MAP.get(adjust, 1)
    return (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?fields1=f1,f2,f3,f4,f5,f6"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&ut=7eea3edcaed734bea9cbfc24409ed989"
        f"&klt=101&fqt={fqt}&secid={market}.{symbol}"
        f"&beg={START_DATE.strftime('%Y%m%d')}&end={END_DATE.strftime('%Y%m%d')}"
    )


def _parse_klines(payload: dict, symbol: str, adjust: str) -> pd.DataFrame:
    info = STOCKS[symbol]
    klines = (payload.get("data") or {}).get("klines") or []
    rows = []
    for line in klines:
        p = line.split(",")
        rows.append(
            {
                "ts_code": info["ts_code"],
                "symbol": symbol,
                "name": info["name"],
                "trade_date": pd.to_datetime(p[0]),
                "open": float(p[1]),
                "close": float(p[2]),
                "high": float(p[3]),
                "low": float(p[4]),
                "volume": float(p[5]),
                "amount": float(p[6]),
                "pct_chg": float(p[8]) if len(p) > 8 and p[8] != "" else None,
                "adjust_type": adjust,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{symbol} 未获取到数据")
    return df.sort_values("trade_date").reset_index(drop=True)


def fetch_via_requests(symbol: str, market: int, adjust: str = "qfq") -> pd.DataFrame:
    url = _eastmoney_url(symbol, market, adjust)
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return _parse_klines(resp.json(), symbol, adjust)


def fetch_via_powershell(symbol: str, market: int, adjust: str = "qfq") -> pd.DataFrame:
    url = _eastmoney_url(symbol, market, adjust)
    raw_path = DATA_DIR / f"_raw_{symbol}_{adjust}.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'Invoke-WebRequest -Uri "{url}" -UseBasicParsing '
        f'| Select-Object -ExpandProperty Content '
        f'| Out-File -FilePath "{raw_path}" -Encoding utf8'
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(raw_path.read_text(encoding="utf-8-sig"))
    return _parse_klines(payload, symbol, adjust)


def fetch_stock(symbol: str, adjust: str = "qfq") -> pd.DataFrame:
    info = STOCKS[symbol]
    try:
        return fetch_via_requests(symbol, info["market"], adjust)
    except Exception as e:
        print(f"  requests 失败 ({e})，改用 PowerShell…")
        return fetch_via_powershell(symbol, info["market"], adjust)


def try_fetch_valuation(symbol: str) -> pd.DataFrame | None:
    """尝试用 akshare 拉取估值类日频指标（PE/PB/PS/市值）。"""
    try:
        import akshare as ak
    except ImportError:
        return None
    info = STOCKS[symbol]
    # 不同 akshare 版本接口名可能不同，逐一尝试
    candidates = []
    if hasattr(ak, "stock_a_indicator_lg"):
        candidates.append(("stock_a_indicator_lg", {"symbol": symbol}))
    if hasattr(ak, "stock_value_em"):
        candidates.append(("stock_value_em", {"symbol": symbol}))
    for fn_name, kwargs in candidates:
        try:
            fn = getattr(ak, fn_name)
            raw = fn(**kwargs)
            if raw is None or raw.empty:
                continue
            df = raw.copy()
            # 统一列名
            colmap = {}
            for c in df.columns:
                cl = str(c).lower()
                if "日期" in str(c) or cl in ("trade_date", "date"):
                    colmap[c] = "trade_date"
                elif str(c) in ("市盈率", "pe", "pe_ttm") or "市盈率" in str(c):
                    colmap[c] = "pe_ttm"
                elif "市净率" in str(c) or cl == "pb":
                    colmap[c] = "pb"
                elif "市销率" in str(c) or cl in ("ps", "ps_ttm"):
                    colmap[c] = "ps_ttm"
                elif "总市值" in str(c) or cl in ("total_mv", "mv"):
                    colmap[c] = "total_mv"
            df = df.rename(columns=colmap)
            if "trade_date" not in df.columns:
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            keep = [c for c in ["trade_date", "pe_ttm", "pb", "ps_ttm", "total_mv"] if c in df.columns]
            df = df[keep].dropna(subset=["trade_date"])
            df = df[
                (df["trade_date"] >= START_DATE) & (df["trade_date"] <= END_DATE)
            ].copy()
            if len(df) > 10:
                print(f"  估值指标来源: {fn_name} ({info['name']})")
                return df
        except Exception as e:
            print(f"  {fn_name} 失败: {e}")
            continue
    return None


def fetch_all(adjust: str = "qfq") -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    for i, symbol in enumerate(STOCKS):
        if i:
            time.sleep(0.6)
        print(f"拉取 {STOCKS[symbol]['name']}({symbol}) …")
        df = fetch_stock(symbol, adjust)
        val = try_fetch_valuation(symbol)
        if val is not None:
            df = df.merge(val, on="trade_date", how="left")
        out = DATA_DIR / f"{symbol}_daily.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        result[symbol] = df
        dmin, dmax = df["trade_date"].min(), df["trade_date"].max()
        print(f"  -> {len(df)} 行, {dmin.date()} ~ {dmax.date()} -> {out.name}")
    return result


def load_all() -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    missing = []
    for symbol in STOCKS:
        path = DATA_DIR / f"{symbol}_daily.csv"
        if not path.exists():
            missing.append(symbol)
            continue
        df = pd.read_csv(path, parse_dates=["trade_date"], dtype={"symbol": str})
        df["symbol"] = symbol
        data[symbol] = df.sort_values("trade_date").reset_index(drop=True)
    if missing:
        print(f"缺失 {missing}，重新拉取全部…")
        return fetch_all("qfq")
    return data


if __name__ == "__main__":
    print(f"区间: {START_DATE.date()} ~ {END_DATE.date()}")
    fetch_all("qfq")
