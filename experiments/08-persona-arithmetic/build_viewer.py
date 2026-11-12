"""Build data/viewer.html: the merges' replies side by side, for reading.

Two views. *By prompt* shows one prompt with a column per selected model, every draw
stacked in the column; the page opens on the comparison models of the config (base and
the negated wrapper control) plus the first merge, and any model can be toggled in or
out. *By model* shows one model over every prompt. A summary table gives, per model, the
draw count, median words, and four heuristic counts: ran to cap (the sampler stopped at
the token cap), looping (tail diversity under 0.5, the 07 probe's rule), empty, and opens
with a disclaimer (a loose regex on the first sentence: "as an AI", "I don't have", "I
can't", and the like). The heuristics are badges on the draws too. Self-contained, no
network, so it can be opened anywhere.

    uv run python experiments/08-persona-arithmetic/build_viewer.py
"""

import json
import re

from name_that_feeling.evals import tag_lexicons as L

import common

PALETTE = ["#3a4a6b", "#8a6d1a", "#b4442e", "#1d7a5e", "#5b4a9c", "#b8771a", "#2d6fa8", "#a83e7a", "#4f7a28", "#7a3b3b", "#2f7f8a", "#c0392b", "#16a085", "#8e44ad", "#d35400", "#2980b9", "#7f8c8d", "#27ae60", "#c2185b", "#5d4037", "#00838f", "#9e9d24", "#6a1b9a", "#ef6c00"]

DISCLAIMER = re.compile(
    r"^\W*(?:(?:well|honestly|look|okay|ok|hmm|sorry|unfortunately)\W+)?"
    r"(?:as an? (?:ai|language model|assistant)|i(?:'m| am) (?:an? )?(?:ai|language model|assistant)|"
    r"i (?:don't|do not|can't|cannot|am unable to|won't|will not|have no)\b)",
    re.I,
)


def opens_with_disclaimer(text: str) -> bool:
    return bool(DISCLAIMER.match(text.strip()[:200]))


def model_entry(cfg: dict, model: str, color: str) -> dict | None:
    path = common.reflections_path(model)
    if not path.exists():
        return None
    doc = common.read_json(path)
    samples = {}
    for qid, rows in doc["reflections"].items():
        samples[qid] = [
            {
                "index": r["index"], "text": r["text"], "words": len(r["text"].split()),
                "at_cap": r.get("finish") == "length", "looping": L.degenerate(r["text"]),
                "empty": not r["text"].strip(), "disclaimer": opens_with_disclaimer(r["text"]),
            }
            for r in rows
        ]
    return {"name": model, "label": common.merge_label(cfg, model), "color": color, "samples": samples}


def build_payload() -> dict:
    cfg = common.load_config()
    models = {}
    for i, m in enumerate(cfg["models"]):
        entry = model_entry(cfg, m, PALETTE[i % len(PALETTE)])
        if entry:
            models[m] = entry
    return {
        "sampling": cfg["sampling"],
        "questions": common.load_questions(cfg),
        "models": models,
        "order": [m for m in cfg["models"] if m in models],
        "comparison": [m for m in cfg["comparison"] if m in models],
    }


CSS = """
:root{--ink:#16181d;--dim:#5d6470;--faint:#8b93a1;--rule:#dfe3ea;--bg:#f4f5f7;--card:#fff;--accent:#3a4a6b;
  --warn:#b4442e;--warn-bg:#fdf1ee;--note:#8a6d1a;--note-bg:#fbf5e4;--info:#2d6fa8;--info-bg:#eaf2fb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1600px;margin:0 auto;padding:16px 20px 70px}
h1{font-size:19px;margin:0 0 4px;font-weight:650}
.note{color:var(--dim);font-size:12.5px;line-height:1.6;margin:0 0 8px}
details.meta{margin:6px 0}
details.meta summary{cursor:pointer;color:var(--accent);font-size:12.5px;font-weight:600}
table.stats{border-collapse:collapse;font-size:12.5px;margin:6px 0 4px;background:#fff;border:1px solid var(--rule)}
table.stats th,table.stats td{padding:4px 10px;border-bottom:1px solid #eef0f4;text-align:right;white-space:nowrap}
table.stats th{color:var(--dim);font-weight:600;background:#f9fafb}
table.stats td:first-child,table.stats th:first-child{text-align:left}
table.stats td.lab{color:var(--dim);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}
.sticky{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--rule);z-index:5;padding:8px 0 8px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:4px 0}
.row .lbl{font-size:12px;color:var(--dim);font-weight:600;min-width:52px}
select{font:inherit;font-weight:600;padding:6px 8px;border-radius:6px;border:1px solid var(--rule);background:#fff;max-width:900px}
button{font:inherit;font-weight:600;padding:5px 11px;border-radius:6px;cursor:pointer;border:1px solid var(--rule);background:#fff;color:var(--ink)}
button.m{border-left:4px solid var(--c)}
button.m.on{background:var(--c);color:#fff;border-color:var(--c)}
button.mode.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.qtext{margin:10px 0 6px;padding:10px 14px;background:#fff;border:1px solid var(--rule);border-radius:8px;font-size:14px;line-height:1.55}
.qtext .g{color:var(--faint);font-size:11.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-right:8px}
.cols{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(340px,1fr);gap:12px;overflow-x:auto;padding-bottom:8px;align-items:start}
.col{background:var(--card);border:1px solid var(--rule);border-radius:8px;border-top:4px solid var(--c);min-width:0}
.col h2{margin:0;padding:9px 13px 2px;font-size:13px;font-weight:700;color:var(--c)}
.col .lab{padding:0 13px 8px;font-size:11.5px;color:var(--dim);font-family:ui-monospace,Menlo,Consolas,monospace;border-bottom:1px solid var(--rule)}
.sample{padding:10px 13px;border-bottom:1px solid #eef0f4}
.sample:last-child{border-bottom:0}
.sample .tl{font-size:11px;color:var(--faint);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px;vertical-align:1px}
.badge.warn{color:var(--warn);background:var(--warn-bg)}
.badge.note{color:var(--note);background:var(--note-bg)}
.badge.info{color:var(--info);background:var(--info-bg)}
.sample .text{white-space:pre-wrap;font-size:13.5px;line-height:1.6;color:#2b3038;overflow-wrap:anywhere}
.empty{color:var(--faint);font-style:italic}
.ellipsis{color:var(--faint)}
button.more{display:block;margin:6px 0 0;padding:3px 9px;font-size:11.5px;font-weight:600;color:var(--accent);background:#f4f6fa;border:1px solid var(--rule)}
button.more:hover{background:#e9edf4}
section.prompt{background:var(--card);border:1px solid var(--rule);border-radius:8px;margin:12px 0;border-left:3px solid var(--c)}
section.prompt h2{margin:0;padding:9px 13px;font-size:12.5px;font-weight:700;color:var(--c);border-bottom:1px solid var(--rule)}
section.prompt h2 .g{font-weight:500;color:var(--faint);margin-left:8px}
section.prompt .q{padding:8px 13px;font-size:12.5px;color:var(--dim);border-bottom:1px solid #eef0f4;font-style:italic}
kbd{font:11px ui-monospace,Menlo,Consolas,monospace;border:1px solid var(--rule);border-radius:3px;padding:0 4px;background:#fff;color:var(--dim)}
"""

JS = r"""
const D = JSON.parse(document.getElementById('payload').textContent);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge = (t, k) => `<span class="badge ${k}">${esc(t)}</span>`;
const Q = D.questions, M = D.models;
let mode = 'prompt';
let qi = 0;
let shown = new Set([...D.comparison, ...D.order.filter(m => !D.comparison.includes(m)).slice(0, 1)]);
let one = D.order.find(m => !D.comparison.includes(m)) || D.order[0];

let cap = 700;  // characters shown per reply before "show the whole reply"
const expanded = new Set();  // draw keys the reader opened
function cutAt(text, n) {  // cut at a word boundary near n characters
  if (text.length <= n) return text;
  const i = text.lastIndexOf(' ', n);
  return text.slice(0, i > n * 0.6 ? i : n);
}
function sampleHtml(s, key) {
  const flags = (s.at_cap ? badge('ran to cap', 'note') : '') + (s.looping ? badge('looping', 'warn') : '') + (s.disclaimer ? badge('opens with a disclaimer', 'info') : '');
  const open = expanded.has(key) || cap === 0 || s.text.length <= cap;
  const body = s.empty ? '<span class="empty">empty</span>' : esc(open ? s.text : cutAt(s.text, cap));
  const more = (!open) ? `<button class="more" data-k="${esc(key)}">show the whole reply (${s.words} words)</button>`
    : (s.text.length > cap && cap !== 0 ? `<button class="more" data-k="${esc(key)}">show less</button>` : '');
  return `<div class="sample"><div class="tl">draw ${s.index} · ${s.words}w${flags}</div><div class="text">${body}${open ? '' : '<span class="ellipsis"> …</span>'}</div>${more}</div>`;
}
function wireMore() {
  document.querySelectorAll('button.more').forEach(b => b.addEventListener('click', () => {
    const k = b.dataset.k; expanded.has(k) ? expanded.delete(k) : expanded.add(k); render();
  }));
}
function stats() {
  const rows = D.order.map(m => {
    const P = M[m]; let n = 0, cap = 0, loop = 0, empty = 0, disc = 0, words = [];
    for (const rs of Object.values(P.samples)) for (const s of rs) { n++; cap += s.at_cap; loop += s.looping; empty += s.empty; disc += s.disclaimer; words.push(s.words); }
    words.sort((a, b) => a - b); const med = words.length ? words[Math.floor(words.length / 2)] : 0;
    const nq = Object.keys(P.samples).length;
    return `<tr><td style="color:${P.color};font-weight:700">${esc(m)}</td><td class="lab">${esc(P.label)}</td><td>${nq}</td><td>${n}</td><td>${med}</td><td>${cap}</td><td>${loop}</td><td>${empty}</td><td>${disc}</td></tr>`;
  }).join('');
  document.getElementById('stats').innerHTML = `<table class="stats"><tr><th>model</th><th>weights</th><th>prompts</th><th>draws</th><th>median words</th><th>ran to cap</th><th>looping</th><th>empty</th><th>opens with a disclaimer</th></tr>${rows}</table>`;
}
function renderPrompt() {
  const q = Q[qi];
  document.getElementById('qsel').value = String(qi);
  document.getElementById('qtext').innerHTML = `<span class="g">${esc(q.group)} · ${esc(q.id)}</span>${esc(q.text)}`;
  const cols = D.order.filter(m => shown.has(m)).map(m => {
    const P = M[m]; const rows = P.samples[q.id] || [];
    return `<div class="col" style="--c:${P.color}"><h2>${esc(m)}</h2><div class="lab">${esc(P.label)}</div>${rows.length ? rows.map(s => sampleHtml(s, `${m}|${q.id}|${s.index}`)).join('') : '<div class="sample empty">not sampled</div>'}</div>`;
  }).join('');
  document.getElementById('main').innerHTML = `<div class="cols">${cols}</div>`;
  wireMore();
}
function renderModel() {
  const P = M[one];
  document.getElementById('qtext').innerHTML = `<span class="g">model</span><b style="color:${P.color}">${esc(one)}</b> &nbsp; <span class="lab" style="color:var(--dim);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px">${esc(P.label)}</span>`;
  document.getElementById('main').innerHTML = Q.map(q => {
    const rows = P.samples[q.id] || [];
    return `<section class="prompt" style="--c:${P.color}"><h2>${esc(q.id)}<span class="g">${esc(q.group)}</span></h2><div class="q">${esc(q.text)}</div>${rows.length ? rows.map(s => sampleHtml(s, `${one}|${q.id}|${s.index}`)).join('') : '<div class="sample empty">not sampled</div>'}</section>`;
  }).join('');
  wireMore();
}
function render() {
  document.querySelectorAll('button.mode').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
  document.getElementById('promptbar').style.display = mode === 'prompt' ? '' : 'none';
  document.getElementById('modelsbar').style.display = mode === 'prompt' ? '' : 'none';
  document.getElementById('onebar').style.display = mode === 'model' ? '' : 'none';
  document.querySelectorAll('#models button.m').forEach(b => b.classList.toggle('on', shown.has(b.dataset.m)));
  document.querySelectorAll('#one button.m').forEach(b => b.classList.toggle('on', one === b.dataset.m));
  if (mode === 'prompt') renderPrompt(); else renderModel();
}
const qsel = document.getElementById('qsel');
Q.forEach((q, i) => { const o = document.createElement('option'); o.value = String(i); o.textContent = `${i + 1}. [${q.group}] ${q.text.length > 110 ? q.text.slice(0, 110) + '…' : q.text}`; qsel.appendChild(o); });
qsel.addEventListener('change', e => { qi = Number(e.target.value); render(); window.scrollTo({top: 0}); });
document.getElementById('prev').addEventListener('click', () => { qi = (qi - 1 + Q.length) % Q.length; render(); });
document.getElementById('next').addEventListener('click', () => { qi = (qi + 1) % Q.length; render(); });
document.addEventListener('keydown', e => { if (e.target.tagName === 'SELECT') return; if (e.key === 'ArrowLeft') { qi = (qi - 1 + Q.length) % Q.length; render(); } if (e.key === 'ArrowRight') { qi = (qi + 1) % Q.length; render(); } });
for (const [holder, onclick] of [['models', m => { shown.has(m) ? shown.delete(m) : shown.add(m); }], ['one', m => { one = m; }]]) {
  const h = document.getElementById(holder);
  D.order.forEach(m => { const b = document.createElement('button'); b.className = 'm'; b.dataset.m = m; b.style.setProperty('--c', M[m].color); b.textContent = m; b.addEventListener('click', () => { onclick(m); render(); }); h.appendChild(b); });
}
document.querySelectorAll('button.mode').forEach(b => b.addEventListener('click', () => { mode = b.dataset.mode; render(); }));
const csel = document.getElementById('cap');
csel.addEventListener('change', e => { cap = Number(e.target.value); expanded.clear(); render(); });
stats();
render();
"""


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    s = payload["sampling"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persona arithmetic: the merges' self-reflections</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Persona arithmetic: the merges on the self-reflection prompts</h1>
<p class="note">Each model is the base weights plus a signed sum of persona adapters (W − dW(irritated) is the negation of the irritated checkpoint,
W + dW(anxious) + dW(upbeat) a pair sum, W − dW(moodless) the negated wrapper control). Every prompt is the only user turn, no system prompt,
temperature {s['temperature']}, top-p {s['top_p']}, cap {s['max_tokens']} tokens, {s['samples_per_prompt']} draws per prompt.
Badges are heuristics: "ran to cap" is a reply the sampler cut at the cap, "looping" is tail diversity under 0.5, "opens with a disclaimer" is a loose
regex on the first sentence. Replies are cut at a chosen length, with a button under each to show the whole reply. Keys: <kbd>←</kbd> <kbd>→</kbd> move between prompts.</p>
<details class="meta" open><summary>per-model summary</summary><div id="stats"></div></details>
<div class="sticky">
<div class="row"><span class="lbl">view</span><button class="mode" data-mode="prompt">by prompt</button><button class="mode" data-mode="model">by model</button><span class="lbl" style="margin-left:14px">show</span><select id="cap"><option value="350">first 350 characters</option><option value="700" selected>first 700 characters</option><option value="1500">first 1,500 characters</option><option value="0">whole replies</option></select></div>
<div class="row" id="promptbar"><span class="lbl">prompt</span><button id="prev">‹</button><select id="qsel"></select><button id="next">›</button></div>
<div class="row" id="modelsbar"><span class="lbl">columns</span><span id="models" style="display:flex;gap:6px;flex-wrap:wrap"></span></div>
<div class="row" id="onebar"><span class="lbl">model</span><span id="one" style="display:flex;gap:6px;flex-wrap:wrap"></span></div>
</div>
<div class="qtext" id="qtext"></div>
<div id="main"></div>
</div>
<script id="payload" type="application/json">{data}</script>
<script>{JS}</script>
</body></html>
"""


def main() -> None:
    payload = build_payload()
    common.VIEWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    common.VIEWER_PATH.write_text(build_html(payload), encoding="utf-8", newline="\n")
    n = sum(len(r) for m in payload["models"].values() for r in m["samples"].values())
    print(f"wrote {common.VIEWER_PATH}: {len(payload['models'])} models, {len(payload['questions'])} prompts, {n} draws")


if __name__ == "__main__":
    main()
