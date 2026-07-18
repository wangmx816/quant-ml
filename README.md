# Quant ML

五只 A 股分别训练分类模型（逻辑回归 / 决策树 / 随机森林），计算 AUC 并绘制 ROC，提供交互看板。

## 标的（与 quant-strategy 一致）

| 代码 | 名称 |
|------|------|
| 002202.SZ | 金风科技 |
| 600031.SH | 三一重工 |
| 000425.SZ | 徐工机械 |
| 600207.SH | 安彩高科 |
| 000816.SZ | 智慧农业 |

## 数据区间

- **2024-03-31 ~ 2026-06-30** 前复权日线（东方财富）
- 特征：动量、波动、均线比、量比、RSI 等
- 标签：下一交易日收益 > 0 → 1，否则 → 0
- **每只股票独立建模**，不混合截面样本

## 看板 Tab

1. **分类评价**：ROC / AUC / 指标表（TASK5）
2. **交易策略**：按时间划分训练测试，模型信号转仓位，净值、季度收益、模型对比与五股等权附加题（TASK6）

## 快速开始

```bash
pip install -r requirements.txt
python -m src.data_fetch
python build_dashboard.py
python generate_task5_report.py --name wangmx
python generate_task6_report.py --name wangmx
```

## 在线看板

- GitHub: https://github.com/wangmx816/quant-ml
- Pages: https://wangmx816.github.io/quant-ml/

## 目录

```
quant-ml/
├── src/
│   ├── data_fetch.py
│   ├── features.py
│   ├── train.py
│   └── strategy.py      # ML 交易策略回测（TASK6）
├── data/
├── output/
├── build_dashboard.py
├── generate_task5_report.py
├── generate_task6_report.py
└── index.html           # 双 Tab 看板
```
