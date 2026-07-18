#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成分类模型 HTML 看板。"""

from __future__ import annotations

import json
from pathlib import Path

from src.data_fetch import STOCKS, load_all
from src.train import run_all

ROOT = Path(__file__).resolve().parent
OUT_HTML = ROOT / "index.html"
METRICS_PATH = ROOT / "output" / "metrics.json"
REPO_URL = "https://github.com/wangmx816/quant-ml"
PAGES_URL = "https://wangmx816.github.io/quant-ml/"


def build_payload(metrics: dict, stock_data: dict) -> dict:
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
                    "cm": res["confusion_matrix"],
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
        "repo_url": REPO_URL,
        "pages_url": PAGES_URL,
        "date_range": metrics["date_range"],
        "split": metrics["split"],
        "label_rule": metrics["stocks"][0]["label_rule"],
        "stocks": stocks,
    }


def render_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>分类机器学习看板 | Quant ML</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f1f5f9; --panel: #fff; --text: #0f172a; --muted: #64748b;
      --accent: #0ea5e9; --border: #e2e8f0; --green: #16a34a; --amber: #d97706;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
      background: var(--bg); color: var(--text);
    }}
    .layout {{ display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }}
    .sidebar {{
      background: var(--panel); border-right: 1px solid var(--border);
      padding: 22px 16px; position: sticky; top: 0; height: 100vh;
    }}
    .sidebar h1 {{ font-size: 1.15rem; margin-bottom: 4px; }}
    .sub {{ color: var(--muted); font-size: .82rem; margin-bottom: 18px; line-height: 1.45; }}
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
    .kpi .value {{ font-size: 1.45rem; font-weight: 700; margin-top: 4px; }}
    .panel {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px 18px; margin-bottom: 16px;
    }}
    .panel h2 {{ font-size: 1rem; margin-bottom: 6px; }}
    .cap {{ color: var(--muted); font-size: .82rem; margin-bottom: 12px; }}
    .charts {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
    .chart-box {{ position: relative; height: 320px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; font-size: .78rem; }}
    .best {{ color: var(--green); font-weight: 700; }}
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
      <p class="sub">五只股票分别训练分类模型<br/>逻辑回归 / 决策树 / 随机森林</p>
      <div class="field">
        <label>股票选择</label>
        <select id="stockSel"></select>
      </div>
      <div class="field">
        <label>说明</label>
        <p class="sub" id="hint"></p>
      </div>
      <div class="links">
        <a id="repoLink" href="{REPO_URL}" target="_blank">GitHub 仓库</a>
        <a id="pagesLink" href="{PAGES_URL}" target="_blank">在线看板</a>
      </div>
    </aside>
    <main class="main">
      <div class="badge" id="rangeBadge"></div>
      <h1 id="title" style="font-size:1.35rem;"></h1>
      <div class="kpi-grid">
        <div class="kpi"><div class="label">样本量</div><div class="value" id="kN">-</div></div>
        <div class="kpi"><div class="label">最佳模型</div><div class="value" id="kBest" style="font-size:1.1rem;">-</div></div>
        <div class="kpi"><div class="label">测试集 AUC</div><div class="value" id="kAuc">-</div></div>
        <div class="kpi"><div class="label">正类占比</div><div class="value" id="kPos">-</div></div>
      </div>
      <div class="charts">
        <div class="panel">
          <h2>图A  ROC 曲线（当前股票）</h2>
          <p class="cap">曲线越靠近左上角、AUC 越大，分类排序能力越强。</p>
          <div class="chart-box"><canvas id="rocChart"></canvas></div>
        </div>
        <div class="panel">
          <h2>图B  收盘价走势</h2>
          <p class="cap">建模区间内的前复权收盘价。</p>
          <div class="chart-box"><canvas id="priceChart"></canvas></div>
        </div>
      </div>
      <div class="panel">
        <h2>图C  当前股票模型指标</h2>
        <p class="cap">测试集指标 + 5 折交叉验证 AUC。</p>
        <table>
          <thead>
            <tr>
              <th>模型</th><th>AUC</th><th>CV-AUC</th><th>准确率</th>
              <th>精确率</th><th>召回率</th><th>F1</th>
            </tr>
          </thead>
          <tbody id="metricBody"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>图D  五只股票最佳 AUC 对比</h2>
        <p class="cap">每只股票独立建模后的最优测试集 AUC。</p>
        <div class="chart-box" style="height:280px;"><canvas id="cmpChart"></canvas></div>
      </div>
    </main>
  </div>
  <script>
    const PAYLOAD = {data_json};
    let rocChart, priceChart, cmpChart;

    function stockBySym(sym) {{
      return PAYLOAD.stocks.find(s => s.symbol === sym);
    }}

    function initSel() {{
      const sel = document.getElementById('stockSel');
      PAYLOAD.stocks.forEach(s => {{
        const opt = document.createElement('option');
        opt.value = s.symbol;
        opt.textContent = s.name + ' (' + s.ts_code + ')';
        sel.appendChild(opt);
      }});
      sel.value = '002202';
      sel.addEventListener('change', () => render(sel.value));
      document.getElementById('rangeBadge').textContent =
        '样本区间 ' + PAYLOAD.date_range + ' · 日频特征 · 分别建模';
      document.getElementById('hint').textContent =
        '标签：' + PAYLOAD.label_rule + '；划分：' + PAYLOAD.split;
    }}

    function render(sym) {{
      const s = stockBySym(sym);
      document.getElementById('title').textContent = s.name + '（' + s.ts_code + '）分类模型评价';
      document.getElementById('kN').textContent = s.n_samples;
      document.getElementById('kBest').textContent = s.best_model;
      document.getElementById('kAuc').textContent = s.best_auc.toFixed(3);
      document.getElementById('kPos').textContent = (s.pos_rate * 100).toFixed(1) + '%';

      const colors = {{'逻辑回归':'#2563eb','决策树':'#dc2626','随机森林':'#16a34a'}};
      const datasets = s.models.map(m => ({{
        label: m.model + ' (AUC=' + m.auc.toFixed(3) + ')',
        data: m.fpr.map((x, i) => ({{x, y: m.tpr[i]}})),
        borderColor: colors[m.model],
        backgroundColor: 'transparent',
        pointRadius: 0,
        borderWidth: 2,
        tension: 0
      }}));
      datasets.push({{
        label: '随机猜测',
        data: [{{x:0,y:0}},{{x:1,y:1}}],
        borderColor: '#94a3b8',
        borderDash: [6,4],
        pointRadius: 0,
        borderWidth: 1
      }});
      if (rocChart) rocChart.destroy();
      rocChart = new Chart(document.getElementById('rocChart'), {{
        type: 'line',
        data: {{ datasets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          scales: {{
            x: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'FPR' }} }},
            y: {{ min: 0, max: 1.02, title: {{ display: true, text: 'TPR' }} }}
          }},
          plugins: {{ legend: {{ position: 'bottom' }} }}
        }}
      }});

      if (priceChart) priceChart.destroy();
      priceChart = new Chart(document.getElementById('priceChart'), {{
        type: 'line',
        data: {{
          labels: s.closes.map(r => r.d),
          datasets: [{{
            label: '收盘价',
            data: s.closes.map(r => r.c),
            borderColor: '#0ea5e9',
            pointRadius: 0,
            borderWidth: 1.5
          }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{ x: {{ ticks: {{ maxTicksLimit: 8 }} }} }}
        }}
      }});

      const body = document.getElementById('metricBody');
      body.innerHTML = s.models.map(m => {{
        const best = m.model === s.best_model ? ' class="best"' : '';
        return `<tr${{best}}>
          <td>${{m.model}}</td>
          <td>${{m.auc.toFixed(3)}}</td>
          <td>${{m.cv_auc.toFixed(3)}}±${{m.cv_std.toFixed(3)}}</td>
          <td>${{m.accuracy.toFixed(3)}}</td>
          <td>${{m.precision.toFixed(3)}}</td>
          <td>${{m.recall.toFixed(3)}}</td>
          <td>${{m.f1.toFixed(3)}}</td>
        </tr>`;
      }}).join('');
    }}

    function renderCompare() {{
      if (cmpChart) cmpChart.destroy();
      cmpChart = new Chart(document.getElementById('cmpChart'), {{
        type: 'bar',
        data: {{
          labels: PAYLOAD.stocks.map(s => s.name),
          datasets: [{{
            label: '最佳 AUC',
            data: PAYLOAD.stocks.map(s => s.best_auc),
            backgroundColor: '#0ea5e9'
          }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          scales: {{ y: {{ min: 0, max: 1 }} }},
          plugins: {{ legend: {{ display: false }} }}
        }}
      }});
    }}

    initSel();
    render('002202');
    renderCompare();
  </script>
</body>
</html>
"""


def main() -> None:
    metrics = run_all()
    stock_data = load_all()
    payload = build_payload(metrics, stock_data)
    OUT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"看板已生成: {OUT_HTML}")
    print(f"在线地址（Pages）: {PAGES_URL}")


if __name__ == "__main__":
    main()
