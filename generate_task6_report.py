#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 TASK6 Word 报告：机器学习交易策略理论 + 回测实证。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.strategy import run_all_strategies

CHART_DIR = ROOT / "output" / "charts"
STRATEGY_PATH = ROOT / "output" / "strategy_metrics.json"
TASK6_DIR = ROOT.parent / "TASK6"
REPO_URL = "https://github.com/wangmx816/quant-ml"
PAGES_URL = "https://wangmx816.github.io/quant-ml/"


def set_run_font(run, size_pt: float = 10.5, bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size_pt)
    run.font.bold = bold


def add_paragraph(doc: Document, text: str, first_line_indent: bool = True) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    if first_line_indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    run = p.add_run(text)
    set_run_font(run)


def add_heading(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    set_run_font(run, bold=True)


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    set_run_font(run)


def add_figure(doc: Document, path: Path, caption: str, interpretation: str) -> None:
    if path.exists():
        doc.add_picture(str(path), width=Cm(14.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, caption)
    add_paragraph(doc, interpretation)


def build_doc(meta: dict, name: str) -> Path:
    stocks = meta["stocks"]
    gw = next(s for s in stocks if s["symbol"] == "002202")
    best = gw["best_model"]
    bm = gw["models"][best]["metrics"]
    bonus = meta.get("bonus", {})

    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.17)
        section.right_margin = Cm(3.17)

    add_heading(doc, "基于机器学习的量化交易策略实验报告（TASK6）")
    add_paragraph(doc, f"姓名：{name}", first_line_indent=False)
    add_paragraph(
        doc,
        f"代码与看板：{REPO_URL} ；在线看板：{PAGES_URL} （含“分类评价 / 交易策略”双 Tab）。",
        first_line_indent=False,
    )

    add_heading(doc, "一、机器学习交易策略的核心理念与优缺点")
    add_paragraph(
        doc,
        "基于机器学习的交易策略，核心理念是：用历史可观测特征（因子）学习价格未来方向或收益的统计规律，"
        "再把模型输出转化为可执行仓位规则（如“预测上涨则持有，否则空仓”）。"
        "它强调数据驱动与可回测，而不是依赖主观盘感。与传统规则策略相比，ML 策略更灵活，能同时利用多因子非线性关系。",
    )
    add_paragraph(
        doc,
        "主要优点：（1）可同时纳入动量、波动、估值等多维信息；（2）决策树/随机森林等模型能刻画非线性与交互；"
        "（3）流程标准化，便于参数对比与风险度量。主要缺点：（1）金融市场噪声大、非平稳，样本外易衰减；"
        "（2）过拟合与前视偏差风险高，必须严格按时间划分；（3）可解释性弱于纯规则策略，实盘还需考虑冲击成本与制度约束。",
    )

    add_heading(doc, "二、常见自变量因子与应变量定义")
    add_paragraph(
        doc,
        "自变量（特征/因子）常见类别包括：价量动量（如 1/5/10 日收益率）、波动率、均线相对强弱、成交量比、"
        "RSI 等技术指标，以及市净率 PB、市销率 PS、市值等估值规模因子。"
        "本实验对五只股票分别构造上述特征；若估值字段可用则一并纳入。",
    )
    add_paragraph(
        doc,
        "应变量（标签）采用二分类：下一交易日收益 Next_Ret>0 记为 1，否则为 0。"
        "交易映射规则：模型给出上涨概率，若概率不低于阈值（本实验 0.55），则次日持有多头，否则空仓。"
        "该设定将“预测问题”转化为“仓位决策问题”，便于计算收益与回撤。",
    )

    add_heading(doc, "三、数据、划分与模型训练")
    add_paragraph(
        doc,
        "加载 quant-ml 已存储的五只股票日线样本（与 TASK5 / quant-strategy 标的一致）："
        "金风科技、三一重工、徐工机械、安彩高科、智慧农业；区间约 2024-03-31 至 2026-06-30。"
        "按时间顺序前 70% 训练、后 30% 测试，避免随机划分造成的信息泄漏。"
        f"以金风科技为例，训练集 {gw['train_start']}～{gw['train_end']}，"
        f"测试集 {gw['test_start']}～{gw['test_end']}。"
        "分别训练逻辑回归、决策树、随机森林，并在测试集上执行策略回测。",
    )

    add_heading(doc, "四、策略回测结果（以金风科技为主）")
    add_paragraph(
        doc,
        f"金风科技在测试集上，按夏普比率最优模型为【{best}】："
        f"累计收益 {bm['cumulative_return']}%，年化 {bm['annualized_return']}%，"
        f"最大回撤 {bm['max_drawdown']}%，夏普 {bm['sharpe_ratio']}，"
        f"相对买入持有超额 {bm['excess_return']}%，交易次数 {bm['trade_count']}。"
        "手续费按万三、滑点万一计入。",
    )

    add_figure(
        doc,
        CHART_DIR / "task6_fig1_equity.png",
        "图1  金风科技测试集净值曲线（模型对比）",
        "图1比较三类模型在同一测试区间的策略净值与买入持有基准。"
        "若某模型净值曲线更平滑且终点更高，通常意味着收益—风险权衡更好；"
        "若大幅低于基准，则说明该模型信号在样本外缺乏稳定优势。",
    )
    add_figure(
        doc,
        CHART_DIR / "task6_fig2_quarterly.png",
        f"图2  金风科技·{best} 测试集分季度收益率",
        "图2给出最优模型在测试集各季度的策略收益与基准收益。"
        "分季度展示有助于观察策略是否依赖某一阶段行情，以及回撤是否集中在特定季度。",
    )
    add_figure(
        doc,
        CHART_DIR / "task6_fig3_compare.png",
        "图3–图4  模型对比与五股最优夏普",
        "左图对比金风科技三类模型的夏普与年化收益；右图汇总五只股票各自最优策略的夏普比率。"
        "可见不同标的可交易性不同：趋势性更强或噪声更低的品种，ML 策略更容易获得正夏普。",
    )

    add_heading(doc, "五、决策树与随机森林等效果对比")
    lines = []
    for mname, mres in gw["models"].items():
        x = mres["metrics"]
        lines.append(
            f"{mname}：累计{x['cumulative_return']}% / 年化{x['annualized_return']}% / "
            f"回撤{x['max_drawdown']}% / 夏普{x['sharpe_ratio']} / 超额{x['excess_return']}%"
        )
    add_paragraph(doc, "金风科技三类模型回测对比如下。" + "；".join(lines) + "。")
    add_paragraph(
        doc,
        "一般而言，逻辑回归更稳健但表达能力有限；决策树灵活但易过拟合；随机森林通过集成降低方差，"
        "常在收益稳定性上更有优势。最终应以样本外夏普、回撤与超额收益综合评判，而非仅看训练准确率。",
    )

    add_heading(doc, "六、附加题：五股等权最优模型组合")
    if bonus.get("metrics"):
        bx = bonus["metrics"]
        add_paragraph(
            doc,
            f"将五只股票各自最优模型的测试集净值等权合成组合："
            f"累计收益 {bx.get('cumulative_return')}%，年化 {bx.get('annualized_return')}%，"
            f"最大回撤 {bx.get('max_drawdown')}%，夏普 {bx.get('sharpe_ratio')}。"
            "分散化有助于平滑单一标的噪声，但若多股同向失效，组合仍可能跑输基准。",
        )
    add_figure(
        doc,
        CHART_DIR / "task6_fig5_bonus.png",
        "图5  附加题：五股等权最优模型组合净值",
        "图5展示组合净值相对等权买入持有的表现。"
        "若组合曲线回撤更小或终点更高，说明跨标的分散对策略稳健性有改善。",
    )

    add_heading(doc, "七、结论")
    add_paragraph(
        doc,
        "本报告阐述了机器学习交易策略的理念与因子/标签定义，并基于已存储五只股票样本完成特征衍生、"
        "时间序列划分、模型训练、仓位规则化与回测评价（含季度收益）。"
        f"看板已增加“交易策略”Tab，可切换股票与模型查看净值与季度收益；仓库与页面见 {REPO_URL} 。"
        "后续可引入交易成本敏感阈值、波动目标仓位或滚动再训练，以进一步贴近实盘约束。",
    )

    TASK6_DIR.mkdir(parents=True, exist_ok=True)
    out_docx = TASK6_DIR / f"{name}TASK6.docx"
    local_docx = ROOT / f"{name}TASK6.docx"
    doc.save(str(out_docx))
    doc.save(str(local_docx))
    return out_docx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=os.environ.get("STUDENT_NAME", "wangmx"))
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    if args.skip_run and STRATEGY_PATH.exists():
        with open(STRATEGY_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = run_all_strategies()

    docx_path = build_doc(meta, args.name)
    print(f"Word: {docx_path}")


if __name__ == "__main__":
    main()
