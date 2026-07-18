#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 Quant ML 看板：分类评价 Tab + ML 交易策略 Tab。"""

from __future__ import annotations

import json
from pathlib import Path

from src.data_fetch import load_all
from src.strategy import run_all_strategies
from src.train import run_all as run_classification

ROOT = Path(__file__).resolve().parent
OUT_HTML = ROOT / "index.html"
METRICS_PATH = ROOT / "output" / "metrics.json"
STRATEGY_PATH = ROOT / "output" / "strategy_metrics.json"
REPO_URL = "https://github.com/wangmx816/quant-ml"
PAGES_URL = "https://wangmx816.github.io/quant-ml/"


def build_classify_payload(metrics: dict, stock_data: dict) -> dict:
    stocks = []
    for m in metrics["stocks"]:
        sym = m["symbol"]
        df = stock_data[sym]
        closes = [
            {"d": r.trade_date.strftime("%Y-%m-%d"), "c": round(float(r.close), 4)}
            for r in df.itertuples()
        ]
        model_rows = []
        for name, res in m["results"].items():
            model_rows.append(
                {
                    "model": name,
                    "auc": round(res["auc"], 4),
                    "cv_auc": round(res["cv_auc_mean"], 4),
                    "cv_std": round(res["cv_auc_std"], 4),
                    "accuracy": round(res["accuracy"], 4),
                    "precision": round(res["precision"], 4),
                    "recall": round(res["recall"], 4),
                    "f1": round(res["f1"], 4),
                    "fpr": res["fpr"],
                    "tpr": res["tpr"],
                }
            )
        stocks.append(
            {
                "symbol": sym,
                "name": m["name"],
                "ts_code": m["ts_code"],
                "n_samples": m["n_samples"],
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "pos_rate": round(m["pos_rate"], 4),
                "date_min": m["date_min"],
                "date_max": m["date_max"],
                "best_model": m["best_model"],
                "best_auc": round(m["best_auc"], 4),
                "feature_cols": m["feature_cols"],
                "models": model_rows,
                "closes": closes,
            }
        )
    return {
        "date_range": metrics["date_range"],
        "split": metrics["split"],
        "label_rule": metrics["stocks"][0]["label_rule"],
        "stocks": stocks,
    }


def build_strategy_payload(strategy: dict) -> dict:
    stocks = []
    for s in strategy["stocks"]:
        models = []
        for name, res in s["models"].items():
            models.append(
                {
                    "model": name,
                    "metrics": res["metrics"],
                    "quarterly": res["quarterly"],
                    "equity": res["equity"],
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                }
            )
        stocks.append(
            {
                "symbol": s["symbol"],
                "name": s["name"],
                "ts_code": s["ts_code"],
                "best_model": s["best_model"],
                "train_start": s["train_start"],
                "train_end": s["train_end"],
                "test_start": s["test_start"],
                "test_end": s["test_end"],
                "prob_threshold": s["prob_threshold"],
                "label_rule": s["label_rule"],
                "models": models,
            }
        )
    return {
        "config": strategy["config"],
        "stocks": stocks,
        "bonus": strategy.get("bonus", {}),
    }


def render_html(classify: dict, strategy: dict) -> str:
    payload = {
        "repo_url": REPO_URL,
        "pages_url": PAGES_URL,
        "classify": classify,
        "strategy": strategy,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Quant ML | 分类与交易策略看板</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f1f5f9; --panel: #fff; --text: #0f172a; --muted: #64748b;
      --accent: #0ea5e9; --border: #e2e8f0; --green: #16a34a;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
      background: var(--bg); color: var(--text);
    }}
    .layout {{ display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }}
    .sidebar {{
      background: var(--panel); border-right: 1px solid var(--border);
      padding: 22px 16px; position: sticky; top: 0; height: 100vh; overflow-y: auto;
    }}
    .sidebar h1 {{ font-size: 1.15rem; margin-bottom: 4px; }}
    .sub {{ color: var(--muted); font-size: .82rem; margin-bottom: 14px; line-height: 1.45; }}
    .tabs {{ display: flex; gap: 8px; margin-bottom: 16px; }}
    .tab-btn {{
      flex: 1; padding: 8px; border: 1px solid var(--border); border-radius: 8px;
      background: #f8fafc; cursor: pointer; font-size: .85rem;
    }}
    .tab-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .field {{ margin-bottom: 14px; }}
    .field label {{ display: block; font-size: .8rem; color: var(--muted); margin-bottom: 4px; }}
    .field select {{
      width: 100%; padding: 8px 10px; border: 1px solid var(--border);
      border-radius: 8px; font-size: .9rem; background: #fff;
    }}
    .links {{ display: flex; flex-direction: column; gap: 8px; margin-top: 18px; }}
    .links a {{
      font-size: .82rem; color: var(--accent); text-decoration: none;
      border: 1px solid #bae6fd; padding: 6px 10px; border-radius: 8px;
    }}
    .main {{ padding: 22px 24px 48px; }}
    .badge {{
      display: inline-block; font-size: .75rem; background: #e0f2fe; color: #0369a1;
      padding: 3px 10px; border-radius: 999px; margin-bottom: 10px;
    }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 14px 0 18px; }}
    .kpi {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px 16px;
    }}
    .kpi .label {{ font-size: .75rem; color: var(--muted); }}
    .kpi .value {{ font-size: 1.35rem; font-weight: 700; margin-top: 4px; }}
    .panel {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px 18px; margin-bottom: 16px;
    }}
    .panel h2 {{ font-size: 1rem; margin-bottom: 6px; }}
    .cap {{ color: var(--muted); font-size: .82rem; margin-bottom: 12px; }}
    .charts {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
    .chart-box {{ position: relative; height: 300px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; font-size: .78rem; }}
    .best {{ color: var(--green); font-weight: 700; }}
    .hidden {{ display: none !important; }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: relative; height: auto; }}
      .kpi-grid, .charts {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 640px) {{
      .kpi-grid, .charts {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>Quant ML</h1>
      <p class="sub">分类评价 + 机器学习交易策略回测</p>
      <div class="tabs">
        <button class="tab-btn active" id="tabClassify" type="button">分类评价</button>
        <button class="tab-btn" id="tabStrategy" type="button">交易策略</button>
      </div>
      <div class="field">
        <label>股票选择</label>
        <select id="stockSel"></select>
      </div>
      <div class="field" id="modelField">
        <label>策略模型</label>
        <select id="modelSel">
          <option>逻辑回归</option>
          <option>决策树</option>
          <option>随机森林</option>
        </select>
      </div>
      <div class="field">
        <label>说明</label>
        <p class="sub" id="hint"></p>
      </div>
      <div class="links">
        <a href="{REPO_URL}" target="_blank">GitHub 仓库</a>
        <a href="{PAGES_URL}" target="_blank">在线看板</a>
      </div>
    </aside>
    <main class="main">
      <section id="viewClassify">
        <div class="badge" id="cBadge"></div>
        <h1 id="cTitle" style="font-size:1.35rem;"></h1>
        <div class="kpi-grid">
          <div class="kpi"><div class="label">样本量</div><div class="value" id="cN">-</div></div>
          <div class="kpi"><div class="label">最佳模型</div><div class="value" id="cBest" style="font-size:1.05rem;">-</div></div>
          <div class="kpi"><div class="label">测试集 AUC</div><div class="value" id="cAuc">-</div></div>
          <div class="kpi"><div class="label">正类占比</div><div class="value" id="cPos">-</div></div>
        </div>
        <div class="charts">
          <div class="panel">
            <h2>图A  ROC 曲线</h2>
            <p class="cap">曲线越靠近左上角，排序能力越强。</p>
            <div class="chart-box"><canvas id="rocChart"></canvas></div>
          </div>
          <div class="panel">
            <h2>图B  收盘价</h2>
            <p class="cap">建模区间前复权收盘价。</p>
            <div class="chart-box"><canvas id="priceChart"></canvas></div>
          </div>
        </div>
        <div class="panel">
          <h2>分类指标表</h2>
          <table>
            <thead><tr><th>模型</th><th>AUC</th><th>CV-AUC</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1</th></tr></thead>
            <tbody id="cBody"></tbody>
          </table>
        </div>
      </section>

      <section id="viewStrategy" class="hidden">
        <div class="badge" id="sBadge"></div>
        <h1 id="sTitle" style="font-size:1.35rem;"></h1>
        <div class="kpi-grid">
          <div class="kpi"><div class="label">累计收益</div><div class="value" id="sCum">-</div></div>
          <div class="kpi"><div class="label">年化收益</div><div class="value" id="sAnn">-</div></div>
          <div class="kpi"><div class="label">最大回撤</div><div class="value" id="sDd">-</div></div>
          <div class="kpi"><div class="label">夏普比率</div><div class="value" id="sSh">-</div></div>
        </div>
        <div class="charts">
          <div class="panel">
            <h2>图C  测试集净值曲线</h2>
            <p class="cap">策略净值 vs 买入持有基准。</p>
            <div class="chart-box"><canvas id="eqChart"></canvas></div>
          </div>
          <div class="panel">
            <h2>图D  分季度收益率</h2>
            <p class="cap">测试集每个季度的策略收益与基准。</p>
            <div class="chart-box"><canvas id="qChart"></canvas></div>
          </div>
        </div>
        <div class="panel">
          <h2>模型回测对比</h2>
          <table>
            <thead>
              <tr>
                <th>模型</th><th>累计收益%</th><th>年化%</th><th>最大回撤%</th>
                <th>夏普</th><th>超额%</th><th>交易次数</th>
              </tr>
            </thead>
            <tbody id="sBody"></tbody>
          </table>
        </div>
        <div class="panel">
          <h2>图E  附加题：五股等权组合</h2>
          <p class="cap">各股取其最优模型净值，等权合成组合。</p>
          <div class="kpi-grid" style="margin-top:8px;">
            <div class="kpi"><div class="label">组合累计</div><div class="value" id="bCum">-</div></div>
            <div class="kpi"><div class="label">组合年化</div><div class="value" id="bAnn">-</div></div>
            <div class="kpi"><div class="label">组合回撤</div><div class="value" id="bDd">-</div></div>
            <div class="kpi"><div class="label">组合夏普</div><div class="value" id="bSh">-</div></div>
          </div>
          <div class="chart-box" style="height:280px;"><canvas id="bonusChart"></canvas></div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const PAYLOAD = {data_json};
    let rocChart, priceChart, eqChart, qChart, bonusChart;
    let mode = 'classify';

    function stockClassify(sym) {{
      return PAYLOAD.classify.stocks.find(s => s.symbol === sym);
    }}
    function stockStrategy(sym) {{
      return PAYLOAD.strategy.stocks.find(s => s.symbol === sym);
    }}

    function initSel() {{
      const sel = document.getElementById('stockSel');
      PAYLOAD.classify.stocks.forEach(s => {{
        const opt = document.createElement('option');
        opt.value = s.symbol;
        opt.textContent = s.name + ' (' + s.ts_code + ')';
        sel.appendChild(opt);
      }});
      sel.value = '002202';
      sel.addEventListener('change', refresh);
      document.getElementById('modelSel').addEventListener('change', () => {{
        if (mode === 'strategy') renderStrategy(sel.value);
      }});
      document.getElementById('tabClassify').onclick = () => setMode('classify');
      document.getElementById('tabStrategy').onclick = () => setMode('strategy');
    }}

    function setMode(m) {{
      mode = m;
      document.getElementById('tabClassify').classList.toggle('active', m === 'classify');
      document.getElementById('tabStrategy').classList.toggle('active', m === 'strategy');
      document.getElementById('viewClassify').classList.toggle('hidden', m !== 'classify');
      document.getElementById('viewStrategy').classList.toggle('hidden', m !== 'strategy');
      document.getElementById('modelField').style.display = m === 'strategy' ? 'block' : 'none';
      refresh();
    }}

    function refresh() {{
      const sym = document.getElementById('stockSel').value;
      if (mode === 'classify') renderClassify(sym);
      else renderStrategy(sym);
    }}

    function renderClassify(sym) {{
      const s = stockClassify(sym);
      document.getElementById('cBadge').textContent =
        '样本区间 ' + PAYLOAD.classify.date_range + ' · 分别建模';
      document.getElementById('cTitle').textContent = s.name + ' 分类模型评价';
      document.getElementById('hint').textContent =
        '标签：' + PAYLOAD.classify.label_rule + '；' + PAYLOAD.classify.split;
      document.getElementById('cN').textContent = s.n_samples;
      document.getElementById('cBest').textContent = s.best_model;
      document.getElementById('cAuc').textContent = s.best_auc.toFixed(3);
      document.getElementById('cPos').textContent = (s.pos_rate * 100).toFixed(1) + '%';

      const colors = {{'逻辑回归':'#2563eb','决策树':'#dc2626','随机森林':'#16a34a'}};
      const datasets = s.models.map(m => ({{
        label: m.model + ' (AUC=' + m.auc.toFixed(3) + ')',
        data: m.fpr.map((x, i) => ({{x, y: m.tpr[i]}})),
        borderColor: colors[m.model], pointRadius: 0, borderWidth: 2, tension: 0
      }}));
      datasets.push({{
        label: '随机猜测', data: [{{x:0,y:0}},{{x:1,y:1}}],
        borderColor: '#94a3b8', borderDash: [6,4], pointRadius: 0, borderWidth: 1
      }});
      if (rocChart) rocChart.destroy();
      rocChart = new Chart(document.getElementById('rocChart'), {{
        type: 'line', data: {{ datasets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          scales: {{
            x: {{ type: 'linear', min: 0, max: 1, title: {{display:true, text:'FPR'}} }},
            y: {{ min: 0, max: 1.02, title: {{display:true, text:'TPR'}} }}
          }},
          plugins: {{ legend: {{ position: 'bottom' }} }}
        }}
      }});

      if (priceChart) priceChart.destroy();
      priceChart = new Chart(document.getElementById('priceChart'), {{
        type: 'line',
        data: {{
          labels: s.closes.map(r => r.d),
          datasets: [{{ label: '收盘价', data: s.closes.map(r => r.c),
            borderColor: '#0ea5e9', pointRadius: 0, borderWidth: 1.5 }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{ x: {{ ticks: {{ maxTicksLimit: 8 }} }} }}
        }}
      }});

      document.getElementById('cBody').innerHTML = s.models.map(m => {{
        const best = m.model === s.best_model ? ' class="best"' : '';
        return `<tr${{best}}><td>${{m.model}}</td><td>${{m.auc.toFixed(3)}}</td>
          <td>${{m.cv_auc.toFixed(3)}}±${{m.cv_std.toFixed(3)}}</td>
          <td>${{m.accuracy.toFixed(3)}}</td><td>${{m.precision.toFixed(3)}}</td>
          <td>${{m.recall.toFixed(3)}}</td><td>${{m.f1.toFixed(3)}}</td></tr>`;
      }}).join('');
    }}

    function renderStrategy(sym) {{
      const s = stockStrategy(sym);
      const modelName = document.getElementById('modelSel').value;
      const m = s.models.find(x => x.model === modelName) || s.models[0];
      document.getElementById('modelSel').value = m.model;
      const mt = m.metrics;
      document.getElementById('sBadge').textContent =
        '测试集 ' + s.test_start + ' ~ ' + s.test_end +
        ' · 阈值 ' + s.prob_threshold + ' · 手续费万三';
      document.getElementById('sTitle').textContent =
        s.name + ' · ' + m.model + ' 交易策略回测';
      document.getElementById('hint').textContent = s.label_rule +
        '；训练 ' + s.train_start + '~' + s.train_end +
        '，测试 ' + s.test_start + '~' + s.test_end;
      document.getElementById('sCum').textContent = mt.cumulative_return.toFixed(2) + '%';
      document.getElementById('sAnn').textContent = mt.annualized_return.toFixed(2) + '%';
      document.getElementById('sDd').textContent = mt.max_drawdown.toFixed(2) + '%';
      document.getElementById('sSh').textContent = mt.sharpe_ratio.toFixed(3);

      if (eqChart) eqChart.destroy();
      eqChart = new Chart(document.getElementById('eqChart'), {{
        type: 'line',
        data: {{
          labels: m.equity.map(p => p.d),
          datasets: [
            {{ label: '策略净值', data: m.equity.map(p => p.nv),
              borderColor: '#0ea5e9', pointRadius: 0, borderWidth: 1.8 }},
            {{ label: '买入持有', data: m.equity.map(p => p.bench),
              borderColor: '#94a3b8', borderDash: [5,4], pointRadius: 0, borderWidth: 1.2 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ position: 'bottom' }} }},
          scales: {{ x: {{ ticks: {{ maxTicksLimit: 7 }} }} }}
        }}
      }});

      if (qChart) qChart.destroy();
      qChart = new Chart(document.getElementById('qChart'), {{
        type: 'bar',
        data: {{
          labels: m.quarterly.map(q => q.quarter),
          datasets: [
            {{ label: '策略', data: m.quarterly.map(q => q.strategy_return), backgroundColor: '#0ea5e9' }},
            {{ label: '买入持有', data: m.quarterly.map(q => q.benchmark_return), backgroundColor: '#94a3b8' }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ position: 'bottom' }} }},
          scales: {{ y: {{ title: {{ display: true, text: '%' }} }} }}
        }}
      }});

      document.getElementById('sBody').innerHTML = s.models.map(row => {{
        const best = row.model === s.best_model ? ' class="best"' : '';
        const x = row.metrics;
        return `<tr${{best}}><td>${{row.model}}</td>
          <td>${{x.cumulative_return}}</td><td>${{x.annualized_return}}</td>
          <td>${{x.max_drawdown}}</td><td>${{x.sharpe_ratio}}</td>
          <td>${{x.excess_return}}</td><td>${{x.trade_count}}</td></tr>`;
      }}).join('');

      const b = PAYLOAD.strategy.bonus || {{}};
      if (b.metrics) {{
        document.getElementById('bCum').textContent = (b.metrics.cumulative_return || 0).toFixed(2) + '%';
        document.getElementById('bAnn').textContent = (b.metrics.annualized_return || 0).toFixed(2) + '%';
        document.getElementById('bDd').textContent = (b.metrics.max_drawdown || 0).toFixed(2) + '%';
        document.getElementById('bSh').textContent = (b.metrics.sharpe_ratio || 0).toFixed(3);
      }}
      if (bonusChart) bonusChart.destroy();
      if (b.equity && b.equity.length) {{
        bonusChart = new Chart(document.getElementById('bonusChart'), {{
          type: 'line',
          data: {{
            labels: b.equity.map(p => p.d),
            datasets: [
              {{ label: '等权组合', data: b.equity.map(p => p.nv),
                borderColor: '#7c3aed', pointRadius: 0, borderWidth: 1.8 }},
              {{ label: '等权买入持有', data: b.equity.map(p => p.bench),
                borderColor: '#94a3b8', borderDash: [5,4], pointRadius: 0, borderWidth: 1.2 }}
            ]
          }},
          options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'bottom' }} }},
            scales: {{ x: {{ ticks: {{ maxTicksLimit: 7 }} }} }}
          }}
        }});
      }}
    }}

    initSel();
    document.getElementById('modelField').style.display = 'none';
    renderClassify('002202');
  </script>
</body>
</html>
"""


def main() -> None:
    stock_data = load_all()
    # 若已有分类指标则复用，否则重跑
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = run_classification()
    strategy = run_all_strategies()
    classify = build_classify_payload(metrics, stock_data)
    strat = build_strategy_payload(strategy)
    OUT_HTML.write_text(render_html(classify, strat), encoding="utf-8")
    print(f"看板已生成: {OUT_HTML}")
    print(f"Pages: {PAGES_URL}")


if __name__ == "__main__":
    main()
