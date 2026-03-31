import argparse
import base64
import copy
import io
import json
import os
import traceback
import re
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from faith.library.utils import get_config


BASE_CONFIG = "config/tiq/evaluate.yml"
REQUEST_TIMEOUT_MS = int(os.environ.get("FAITH_DEMO_REQUEST_TIMEOUT_MS", "3600000"))

DISABLE_HA_DEFAULT = os.environ.get("FAITH_DEMO_DISABLE_HA", "1") == "0"


def parse_tsf(tsf_raw, delimiter="||"):
    if isinstance(tsf_raw, dict):
        return {
            "entity": ", ".join(tsf_raw.get("entity", [])) if isinstance(tsf_raw.get("entity", []), list) else str(tsf_raw.get("entity", "")),
            "relation": tsf_raw.get("relation", ""),
            "answer_type": tsf_raw.get("answer_type", ""),
            "temporal_signal": tsf_raw.get("temporal_signal", ""),
            # "category": tsf_raw.get("category", ""),
            "category": tsf_raw.get("category", ""),
        }

    tsf_text = tsf_raw or ""
    parts = tsf_text.split(delimiter)
    while len(parts) < 5:
        parts.append("")
    return {
        "entity": parts[0].strip(),
        "relation": parts[1].strip(),
        "answer_type": parts[2].strip(),
        "temporal_signal": parts[3].strip(),
        "category": parts[4].strip(),
    }


def fallback_tsf(question):
    tokens = [t.strip(" ?!,.，。！？") for t in question.split() if t.strip()]
    relation = " ".join(tokens[:8])
    return {
        "entity": "",
        "relation": relation,
        "answer_type": "",
        "temporal_signal": "",
        "category": "",
    }


def _is_temporal_id(value):
    if not isinstance(value, str):
        return False
    return bool(re.match(r'^-?\d{4}(-\d{2}-\d{2}T00:00:00Z)?$', value.strip().replace('"', '')))


def render_graph_png(instance, question, answer_label):
    graph = nx.Graph()
    graph.add_node("Q", kind="question", label=(question[:40] + "...") if len(question) > 40 else question)

    evidences = (instance.get("candidate_evidences") or instance.get("top_evidences") or [])[:12]
    source_color = {
        "kb": "#3b82f6",
        "text": "#22c55e",
        "table": "#f59e0b",
        "info": "#a855f7",
    }

    entity_map = {}
    temporal_count = 0
    for idx, ev in enumerate(evidences):
        src = ev.get("source", "kb")
        ev_id = f"EV{idx+1}"
        ev_text = ev.get("evidence_text", "")
        graph.add_node(
            ev_id,
            kind=f"evidence_{src}",
            label=(ev_text[:36] + "...") if len(ev_text) > 36 else ev_text,
            source=src,
        )
        graph.add_edge("Q", ev_id)

        for ent in ev.get("wikidata_entities", [])[:6]:
            ent_label = ent.get("label") or ent.get("id") or "entity"
            ent_id = ent.get("id", ent_label)
            key = f"ENT::{ent_id}"
            if _is_temporal_id(ent_id):
                node_id = f"TMP::{ent_id}"
                if not graph.has_node(node_id):
                    temporal_count += 1
                    graph.add_node(node_id, kind="temporal", label=ent_label)
                graph.add_edge(ev_id, node_id)
                continue

            if key not in entity_map:
                node_id = f"EN{len(entity_map)+1}"
                entity_map[key] = node_id
                graph.add_node(node_id, kind="entity", label=ent_label)
            graph.add_edge(ev_id, entity_map[key])

    if answer_label:
        graph.add_node("A", kind="answer", label=answer_label)
        # connect to matching entities if any
        linked = False
        for n, attrs in graph.nodes(data=True):
            if attrs.get("kind") == "entity" and attrs.get("label", "").lower() == answer_label.lower():
                graph.add_edge("A", n)
                linked = True
        if not linked:
            graph.add_edge("A", "Q")

    pos = nx.spring_layout(graph, seed=7, k=0.85)

    plt.figure(figsize=(13, 8))
    nx.draw_networkx_edges(graph, pos, alpha=0.35, width=1.6)

    def draw_kind(kind, color, shape, size):
        nodes = [n for n, attrs in graph.nodes(data=True) if attrs.get("kind") == kind]
        if nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=color, node_shape=shape, node_size=size, alpha=0.95)

    draw_kind("question", "#111827", "o", 900)
    draw_kind("answer", "#16a34a", "o", 900)
    draw_kind("entity", "#6b7280", "o", 860)
    draw_kind("temporal", "#9ca3af", "^", 560)
    for src, color in source_color.items():
        draw_kind(f"evidence_{src}", color, "s", 620)

    labels = {n: attrs.get("label", n) for n, attrs in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.axis("off")

    source_counts = {src: len([1 for _, a in graph.nodes(data=True) if a.get("kind") == f"evidence_{src}"]) for src in source_color}
    entity_count = len([1 for _, a in graph.nodes(data=True) if a.get("kind") == "entity"])
    legend_lines = [
        f"● Entity ({entity_count})",
        f"▲ Temporal Info ({temporal_count})",
        f"■ KB ({source_counts['kb']})",
        f"■ Text ({source_counts['text']})",
        f"■ Table ({source_counts['table']})",
        f"■ Infobox ({source_counts['info']})",
    ]
    plt.figtext(0.02, 0.17, "\n".join(legend_lines), fontsize=10, color="#334155")
    plt.figtext(
        0.02,
        0.05,
        f"Nodes (Entity / Temporal + Sources): {entity_count + temporal_count + sum(source_counts.values())}\nEvidence Texts: {len(evidences)}",
        fontsize=10,
        color="#334155",
    )

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=160)
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def simple_graph_svg_data_url(question, answer_label):
    q = (question or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    a = (answer_label or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    q = q[:100]
    a = a[:80]
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='320'>
<rect width='100%' height='100%' fill='#0d1328'/>
<line x1='260' y1='185' x2='560' y2='170' stroke='#64748b' stroke-width='2'/>
<line x1='560' y1='170' x2='840' y2='185' stroke='#64748b' stroke-width='2'/>
<circle cx='220' cy='185' r='34' fill='#6b7280'/>
<rect x='520' y='130' width='80' height='80' fill='#3b82f6'/>
<polygon points='860,220 900,150 940,220' fill='#9ca3af'/>
<circle cx='980' cy='185' r='34' fill='#16a34a'/>
<text x='220' y='240' fill='#94a3b8' text-anchor='middle' font-size='13' font-family='Arial'>Entity</text>
<text x='560' y='240' fill='#94a3b8' text-anchor='middle' font-size='13' font-family='Arial'>Evidence</text>
<text x='900' y='240' fill='#94a3b8' text-anchor='middle' font-size='13' font-family='Arial'>Temporal</text>
<text x='980' y='240' fill='#94a3b8' text-anchor='middle' font-size='13' font-family='Arial'>Answer</text>
<text x='600' y='60' fill='#cbd5e1' text-anchor='middle' font-size='16' font-family='Arial'>Heuristic GNN-style Graph</text>
<text x='600' y='280' fill='#cbd5e1' text-anchor='middle' font-size='14' font-family='Arial'>Q: {q}</text>
<text x='600' y='305' fill='#cbd5e1' text-anchor='middle' font-size='14' font-family='Arial'>A: {a}</text>
</svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")


def discover_tsf_models(config_path=BASE_CONFIG):
    config = get_config(config_path)
    model_dir = Path(config["path_to_data"]) / config["benchmark"]
    candidates = sorted([p.name for p in model_dir.glob("*.bin")])
    if config.get("tsf_model") and config["tsf_model"] not in candidates:
        candidates = [config["tsf_model"]] + candidates
    return candidates if candidates else [config.get("tsf_model", "tsf_model.bin")]


def discover_iques_models(config_path=BASE_CONFIG):
    config = get_config(config_path)
    model_dir = Path(config["path_to_data"]) / config["benchmark"]
    candidates = sorted([p.name for p in model_dir.glob("*.bin")])
    if config.get("iques_model") and config["iques_model"] not in candidates:
        candidates = [config["iques_model"]] + candidates
    return candidates if candidates else [config.get("iques_model", "iques_model.bin")]


def infer_architecture_from_model_name(model_name, fallback="BART"):
    name = (model_name or "").lower()
    if "pegasus" in name:
        return "Pegasus"
    if "t5" in name:
        return "T5"
    if "bart" in name:
        return "BART"
    return fallback


class FaithDemoService:
    def __init__(self):
        self.pipeline_cache = {}
        self.warmup_status = {"running": False, "done": False, "error": ""}
        self.result_cache = {}
        self.result_cache_lock = threading.Lock()

    def put_result(self, request_id, payload):
        with self.result_cache_lock:
            self.result_cache[request_id] = payload
            # keep cache bounded
            if len(self.result_cache) > 50:
                for k in list(self.result_cache.keys())[:-50]:
                    self.result_cache.pop(k, None)

    def pop_result(self, request_id):
        with self.result_cache_lock:
            return self.result_cache.pop(request_id, None)

    def warmup_model(self, tsf_model_name="tsf_model.bin", iques_model_name="iques_model.bin"):
        if self.warmup_status["running"] or self.warmup_status["done"]:
            return
        self.warmup_status = {"running": True, "done": False, "error": ""}
        try:
            print(f"[WARMUP][START] tsf={tsf_model_name} iques={iques_model_name}", flush=True)
            self._get_pipeline(BASE_CONFIG, tsf_model_name, iques_model_name)
            self.warmup_status = {"running": False, "done": True, "error": ""}
            print(f"[WARMUP][DONE] tsf={tsf_model_name} iques={iques_model_name}", flush=True)
        except Exception as exc:
            self.warmup_status = {"running": False, "done": False, "error": str(exc)}
            print(f"[WARMUP][FAIL] {exc}", flush=True)

    def _get_pipeline(self, config_path, tsf_model_name, iques_model_name):
        cache_key = (config_path, tsf_model_name, iques_model_name)
        if cache_key not in self.pipeline_cache:
            from faith.pipeline import Pipeline
            config = copy.deepcopy(get_config(config_path))
            config["tsf_model"] = tsf_model_name
            config["iques_model"] = iques_model_name
            config["tsf_architecture"] = infer_architecture_from_model_name(tsf_model_name, config.get("tsf_architecture", "BART"))
            config["iques_architecture"] = infer_architecture_from_model_name(iques_model_name, config.get("iques_architecture", "BART"))
            self.pipeline_cache[cache_key] = (config, Pipeline(config))
        return self.pipeline_cache[cache_key]

    def _inference_without_ha(self, pipeline, instance, sources):
        """Run only TQU + FER to avoid HA/GNN OOM on limited machines."""
        # TQU
        pipeline.tqu.inference_on_instance(instance, topk_answers=1, sources=sources)
        # FER
        pipeline.fer.inference_on_instance(instance, sources)
        return instance

    def answer_question(self, question, tsf_model_name, iques_model_name, sources):
        config_path = BASE_CONFIG
        start_time = time.time()
        print(
            f"[ASK][START] tsf={tsf_model_name} iques={iques_model_name} sources={sources} question={question[:120]}",
            flush=True,
        )

        result = {
            "model": {"config": config_path, "tsf_model": tsf_model_name, "iques_model": iques_model_name},
            "question": question,
            "tsf": {},
            "answer": {"label": "", "id": ""},
            "graph_base64": "",
            "logs": [],
        }

        if tsf_model_name == "demo_heuristic":
            result["tsf"] = fallback_tsf(question)
            result["answer"] = {"label": "Demo answer (configure full model for real inference)", "id": "demo"}
            result["graph_base64"] = simple_graph_svg_data_url(question, result["answer"]["label"])
            result["logs"].append("Heuristic demo mode used (no FAITH model inference).")
            result["logs"].append(f"Elapsed: {time.time() - start_time:.2f}s")
            print(f"[ASK][DONE] mode=heuristic elapsed={time.time() - start_time:.2f}s", flush=True)
            return result

        try:
            config, pipeline = self._get_pipeline(config_path, tsf_model_name, iques_model_name)
            delimiter = config.get("tsf_delimiter", "||")

            instance = {
                "Id": -1,
                "Question creation date": "2023-01-01",
                "Question": question,
                # FAITH pipeline modules may call StringLibrary.format_answers(instance),
                # which expects the `Answer` field to exist.
                "Answer": [],
            }
            disable_ha = DISABLE_HA_DEFAULT
            if disable_ha:
                instance = self._inference_without_ha(pipeline, instance, sources)
            else:
                instance = pipeline.inference_on_instance(instance, topk_answers=1, sources=sources)

            tsf_raw = instance.get("structured_temporal_form") or instance.get("silver_tsf")
            result["tsf"] = parse_tsf(tsf_raw, delimiter=delimiter)

            ranked_answers = instance.get("ranked_answers", [])
            if ranked_answers:
                top_answer = ranked_answers[0].get("answer", {})
                result["answer"] = {
                    "label": top_answer.get("label", ""),
                    "id": top_answer.get("id", ""),
                }
            else:
                # HA disabled path: derive a lightweight proxy answer from top evidence entities
                proxy_answer = None
                for ev in instance.get("candidate_evidences", [])[:5]:
                    ents = ev.get("wikidata_entities", [])
                    if ents:
                        proxy_answer = ents[0]
                        break
                if proxy_answer:
                    result["answer"] = {
                        "label": proxy_answer.get("label", "(No answer returned)"),
                        "id": proxy_answer.get("id", ""),
                    }
                else:
                    result["answer"] = {"label": "(No answer returned)", "id": ""}

            result["graph_base64"] = "data:image/png;base64," + render_graph_png(instance, question, result["answer"]["label"])
            if disable_ha:
                result["logs"].append("Pipeline inference completed (HA skipped in memory-safe mode).")
            else:
                result["logs"].append("Pipeline inference completed.")
            result["logs"].append(f"Elapsed: {time.time() - start_time:.2f}s")
            print(f"[ASK][DONE] mode=pipeline elapsed={time.time() - start_time:.2f}s", flush=True)
            return result

        except Exception as exc:
            result["logs"].append("Pipeline inference failed; fallback mode used.")
            result["logs"].append(str(exc))
            result["logs"].append(traceback.format_exc().splitlines()[-1])
            result["tsf"] = fallback_tsf(question)
            result["answer"] = {"label": "N/A (model not available)", "id": ""}
            result["graph_base64"] = simple_graph_svg_data_url(question, result["answer"]["label"])
            result["logs"].append(f"Elapsed: {time.time() - start_time:.2f}s")
            print(f"[ASK][FAIL] elapsed={time.time() - start_time:.2f}s err={exc}", flush=True)
            return result


def html_page():
    model_options = ['<option value="demo_heuristic">Demo heuristic（快速演示）</option>']
    for model_name in discover_tsf_models(BASE_CONFIG):
        model_options.append(f'<option value="{model_name}">{model_name}</option>')
    options_html = "\n".join(model_options)
    iques_options = []
    for model_name in discover_iques_models(BASE_CONFIG):
        iques_options.append(f'<option value="{model_name}">{model_name}</option>')
    iques_options_html = "\n".join(iques_options)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>FAITH QA Demo</title>
  <style>
    body {{ font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0b1020; color: #eef2ff; }}
    .wrap {{ max-width: 1160px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: linear-gradient(135deg,#1d4ed8,#7c3aed); border-radius: 16px; padding: 18px 22px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }}
    .hero h1 {{ margin: 0; font-size: 24px; }}
    .hero p {{ margin: 6px 0 0; opacity: .95; }}
    .card {{ background: #121a33; border: 1px solid #2b3868; border-radius: 14px; padding: 16px; margin-top: 16px; }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .grow {{ flex: 1; min-width: 260px; }}
    input, select {{ width: 100%; background: #0f1730; color: #eef2ff; border: 1px solid #40508d; border-radius: 10px; padding: 10px 12px; box-sizing: border-box; }}
    button {{ background: #22c55e; color: #052e16; border: 0; border-radius: 10px; padding: 10px 18px; font-weight: 700; cursor: pointer; }}
    button:hover {{ filter: brightness(1.06); }}
    .slots {{ display: grid; grid-template-columns: repeat(2,minmax(220px,1fr)); gap: 10px; }}
    .slot {{ background: #0f1730; border: 1px solid #33467d; border-radius: 10px; padding: 10px; }}
    .slot .k {{ color: #93c5fd; font-size: 12px; }}
    .slot .v {{ margin-top: 4px; font-size: 14px; word-break: break-word; }}
    .muted {{ opacity: .8; font-size: 13px; }}
    .ans {{ font-size: 18px; color: #86efac; font-weight: 700; }}
    img {{ width: 100%; border-radius: 12px; border: 1px solid #33467d; background: #0d1328; }}
    pre {{ white-space: pre-wrap; background: #0f1730; border-radius: 10px; padding: 10px; border: 1px solid #33467d; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>FAITH 问答演示系统</h1>
      <p>输入问题 -> 选择 TSF 配置 -> 展示 TSF 五槽、答案和 HA 阶段关联图。</p>
    </div>

    <div class="card">
      <div class="row">
        <div class="grow">
          <label>问题输入</label>
          <input id="question" placeholder="例如：Who was the US president during WW2?" />
        </div>
        <div class="grow">
          <label>TSF 模型选择（切换 .bin）</label>
          <select id="tsfModel">{options_html}</select>
        </div>
        <div class="grow">
          <label>中间问题模型（IQUES）</label>
          <select id="iquesModel">{iques_options_html}</select>
        </div>
      </div>
      <div class="row" style="margin-top:10px; align-items:center;">
        <div class="grow">
          <label>证据源</label>
          <input id="sources" value="kb,text,table,info" />
        </div>
        <div style="padding-top:22px;">
          <button type="button" id="runBtn" onclick="ask()">运行问答</button>
        </div>
        <div style="padding-top:22px;">
          <button type="button" id="warmBtn" onclick="warmup()">预热模型</button>
        </div>
      </div>
      <p class="muted">注：若本机未加载完整模型，系统会自动进入 fallback 模式，仍可演示 UI 与流程。当前前端等待上限：{REQUEST_TIMEOUT_MS // 1000} 秒。已启用心跳保活（长任务期间会持续输出进度心跳）。</p>
    </div>

    <div class="card">
      <h3 style="margin-top:0">TSF 槽位</h3>
      <div class="slots" id="slots"></div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">答案</h3>
      <div class="ans" id="answerLabel">(尚未运行)</div>
      <div class="muted" id="answerId"></div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">HA / GNN 关联图</h3>
      <img id="graph" alt="graph" />
    </div>

    <div class="card">
      <h3 style="margin-top:0">运行日志</h3>
      <pre id="logs"></pre>
    </div>
  </div>
  <script>
    function renderSlots(tsf) {{
      const order = [
        ['entity', 'Entity'],
        ['relation', 'Relation'],
        ['answer_type', 'Answer Type'],
        ['temporal_signal', 'Temporal Signal'],
        ['category', 'category']
      ];
      document.getElementById('slots').innerHTML = order.map(([k,title]) =>
        `<div class="slot"><div class="k">${{title}}</div><div class="v">${{(tsf[k] || '(empty)')}}</div></div>`
      ).join('');
    }}

    async function ask() {{
      const question = document.getElementById('question').value.trim();
      if (!question) {{ alert('请输入问题'); return; }}

      const tsf_model = document.getElementById('tsfModel').value;
      const iques_model = document.getElementById('iquesModel').value;
      const sourcesRaw = document.getElementById('sources').value.trim();
      const sources = sourcesRaw.split(',').map(s => s.trim()).filter(Boolean);

      const runBtn = document.getElementById('runBtn');
      runBtn.disabled = true;
      runBtn.textContent = '运行中...';
      document.getElementById('logs').textContent = '正在运行问答中...（前端请求已发出）';

      try {{
        const params = new URLSearchParams({{
          question,
          tsf_model,
          iques_model,
          sources: sources.join(','),
          _ts: Date.now().toString()
        }});

        const es = new EventSource('/api/ask_stream?' + params.toString());
        let finished = false;
        let resultRequested = false;
        const finishRun = () => {{
          if (finished) return false;
          finished = true;
          clearTimeout(timeoutId);
          es.close();
          runBtn.disabled = false;
          runBtn.textContent = '运行问答';
          return true;
        }};
        const timeoutId = setTimeout(() => {{
          es.close();
          document.getElementById('logs').textContent += '\\n❌ 流式请求超时，连接已关闭。';
          runBtn.disabled = false;
          runBtn.textContent = '运行问答';
        }}, {REQUEST_TIMEOUT_MS});

        es.addEventListener('log', (evt) => {{
          const logs = document.getElementById('logs');
          logs.textContent = (logs.textContent ? logs.textContent + '\\n' : '') + evt.data;
        }});

        const applyResult = (payload) => {{
          const data = JSON.parse(payload);
          renderSlots(data.tsf || {{}});
          document.getElementById('answerLabel').textContent = data.answer?.label || '(none)';
          document.getElementById('answerId').textContent = data.answer?.id ? `ID: ${{data.answer.id}}` : '';
          if (data.graph_base64) document.getElementById('graph').src = data.graph_base64;
          document.getElementById('logs').textContent = ['✅ 请求成功', ...(data.logs || [])].join('\\n');
        }};

        const fetchResultById = async (requestId) => {{
          const res = await fetch('/api/result?id=' + encodeURIComponent(requestId));
          if (!res.ok) throw new Error('result fetch failed: ' + res.status);
          const data = await res.json();
          applyResult(JSON.stringify(data));
        }};

        es.addEventListener('result', async (evt) => {{
          if (finished) return;
          try {{
            const maybe = JSON.parse(evt.data || '{{}}');
            if (maybe.request_id) {{
              if (resultRequested) return;
              resultRequested = true;
              await fetchResultById(maybe.request_id);
            }} else {{
              applyResult(evt.data);
            }}
          }} catch (err) {{
            document.getElementById('logs').textContent += '\\n❌ 解析结果失败: ' + err;
          }}
          finishRun();
        }});

        es.onmessage = (evt) => {{
          // fallback for gateways/proxies that normalize SSE event names
          if (!evt.data) return;
          try {{
            const maybe = JSON.parse(evt.data);
            if (maybe && typeof maybe === 'object') {{
              if (maybe.request_id) {{
                if (finished || resultRequested) return;
                resultRequested = true;
                fetchResultById(maybe.request_id)
                  .catch(err => {{
                    document.getElementById('logs').textContent += '\\n❌ 拉取结果失败: ' + err;
                  }})
                  .finally(() => {{
                    finishRun();
                  }});
                return;
              }}
              if (maybe.answer || maybe.tsf || maybe.graph_base64) {{
                applyResult(evt.data);
                finishRun();
                return;
              }}
            }}
          }} catch (_err) {{
            // non-JSON message, ignore and wait for next event
          }}
        }};

        es.addEventListener('error', () => {{
          clearTimeout(timeoutId);
          const logs = document.getElementById('logs');
          logs.textContent = (logs.textContent ? logs.textContent + '\\n' : '') + '❌ 流式连接中断（可能网关超时或服务重启）';
          finishRun();
        }});
      }} catch (err) {{
        document.getElementById('logs').textContent = '❌ Request failed: ' + err;
        runBtn.disabled = false;
        runBtn.textContent = '运行问答';
      }}
    }}

    async function warmup() {{
      const warmBtn = document.getElementById('warmBtn');
      warmBtn.disabled = true;
      warmBtn.textContent = '预热中...';
      document.getElementById('logs').textContent = '正在预热模型（首次会较慢）...';
      try {{
        const tsf_model = document.getElementById('tsfModel').value;
        const iques_model = document.getElementById('iquesModel').value;
        const params = new URLSearchParams({{tsf_model, iques_model, _ts: Date.now().toString()}});
        const res = await fetch('/api/warmup?' + params.toString(), {{ method: 'POST' }});
        const data = await res.json();
        document.getElementById('logs').textContent = ['✅ 预热请求已提交', ...(data.logs || [])].join('\\n');
      }} catch (err) {{
        document.getElementById('logs').textContent = '❌ warmup failed: ' + err;
      }} finally {{
        warmBtn.disabled = false;
        warmBtn.textContent = '预热模型';
      }}
    }}

    window.addEventListener('error', function(evt) {{
      const msg = '❌ 前端脚本错误: ' + (evt.message || evt.error || 'unknown');
      const logs = document.getElementById('logs');
      logs.textContent = (logs.textContent ? logs.textContent + '\\n' : '') + msg;
    }});

    renderSlots({{}});
    document.getElementById('logs').textContent = 'UI 已就绪，点击“运行问答”开始。';
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    service = FaithDemoService()

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            # client closed connection (e.g., browser timeout/refresh)
            print("[HTTP][WARN] client disconnected before response was sent", flush=True)

    def _send_sse(self, event_name, data):
        payload = f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")
        self.wfile.write(payload)
        self.wfile.flush()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = html_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/ask_stream":
            try:
                params = parse_qs(parsed.query)
                question = (params.get("question", [""])[0] or "").strip()
                tsf_model = (params.get("tsf_model", ["demo_heuristic"])[0] or "demo_heuristic").strip()
                iques_model = (params.get("iques_model", ["iques_model.bin"])[0] or "iques_model.bin").strip()
                sources_str = params.get("sources", ["kb,text,table,info"])[0]
                sources = [s.strip() for s in sources_str.split(",") if s.strip()]

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                self._send_sse("log", "▶️ 请求已进入后端，正在执行...")

                result_holder = {"done": False, "result": None, "error": None}

                def worker():
                    try:
                        result_holder["result"] = self.service.answer_question(question, tsf_model, iques_model, sources)
                    except Exception as exc:
                        result_holder["error"] = str(exc)
                    finally:
                        result_holder["done"] = True

                threading.Thread(target=worker, daemon=True).start()

                while not result_holder["done"]:
                    self._send_sse("log", "⏳ 后端仍在运行（心跳保活中）...")
                    time.sleep(5)

                request_id = f"r{int(time.time()*1000)}_{threading.get_ident()}"
                if result_holder["error"]:
                    payload_obj = {
                        "tsf": fallback_tsf(question),
                        "answer": {"label": "N/A", "id": ""},
                        "graph_base64": simple_graph_svg_data_url(question, "N/A"),
                        "logs": ["Backend error", result_holder["error"]],
                    }
                else:
                    payload_obj = result_holder["result"]
                self.service.put_result(request_id, payload_obj)
                ticket_payload = json.dumps({"request_id": request_id}, ensure_ascii=False)
                self._send_sse("result", ticket_payload)
                return
            except BrokenPipeError:
                print("[HTTP][WARN] ask_stream client disconnected", flush=True)
                return
            except Exception as exc:
                print(f"[HTTP][ERROR] ask_stream failed: {exc}", flush=True)
                return
        if parsed.path == "/api/result":
            request_id = (parse_qs(parsed.query).get("id", [""])[0] or "").strip()
            if not request_id:
                return self._send_json({"error": "id is required"}, status=400)
            payload = self.service.pop_result(request_id)
            if payload is None:
                return self._send_json({"error": "result not found or already consumed"}, status=404)
            return self._send_json(payload)
        if parsed.path == "/healthz":
            return self._send_json({"ok": True})
        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/warmup":
            params = parse_qs(parsed.query)
            tsf_model = (params.get("tsf_model", ["tsf_model.bin"])[0] or "tsf_model.bin").strip()
            iques_model = (params.get("iques_model", ["iques_model.bin"])[0] or "iques_model.bin").strip()
            threading.Thread(target=self.service.warmup_model, args=(tsf_model, iques_model), daemon=True).start()
            return self._send_json({"ok": True, "logs": [f"Warmup thread started for tsf={tsf_model}, iques={iques_model}."]})

        if parsed.path != "/api/ask":
            self.send_error(404, "Not found")
            return

        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
        payload = json.loads(raw)

        question = (payload.get("question") or "").strip()
        tsf_model = payload.get("tsf_model", "demo_heuristic")
        iques_model = payload.get("iques_model", "iques_model.bin")
        sources = payload.get("sources") or ["kb", "text", "table", "info"]
        if not question:
            return self._send_json({"error": "question is required"}, status=400)

        print(f"[HTTP][POST] /api/ask tsf={tsf_model} iques={iques_model} sources={sources}", flush=True)
        result = self.service.answer_question(question, tsf_model, iques_model, sources)
        print("[HTTP][POST] /api/ask finished", flush=True)
        self._send_json(result)


def main():
    parser = argparse.ArgumentParser(description="FAITH local QA demo UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FAITH demo running at http://{args.host}:{args.port}")
    print(f"FAITH demo request timeout: {REQUEST_TIMEOUT_MS / 1000:.0f}s", flush=True)
    print(f"FAITH demo mode: HA disabled={DISABLE_HA_DEFAULT} (set FAITH_DEMO_DISABLE_HA=0 to enable HA)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
