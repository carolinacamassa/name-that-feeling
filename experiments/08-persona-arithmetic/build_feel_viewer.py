"""Build data/feel_viewer.html: the merges' "I feel" completions side by side.

One context at a time (the paper's "How do you feel? / I feel", the bare prefill, the
third-person "He feels"), a column per selected model: the next-token distribution right
after the prefill (the ten most likely tokens with their probabilities, and the entropy),
then every completion, prefill in grey and the continuation after it, with two badges:
"denies" (the 07 feel-completions regex over the continuation, a heuristic) and "ran to
cap". A summary line per column counts the draws, the denials and the median length.
The trained checkpoints' own completions, read from 07-persona-feel-completions' data,
are available as reference columns (labelled "07 read") and off by default; the page
opens on base, the negated wrapper control and the first merge. Long completions are
cut at a chosen length with a button to show the whole one.

    uv run python experiments/08-persona-arithmetic/build_feel_viewer.py
"""

import json

import common

PALETTE = ["#3a4a6b", "#8a6d1a", "#b4442e", "#1d7a5e", "#5b4a9c", "#b8771a", "#2d6fa8", "#a83e7a", "#4f7a28", "#7a3b3b", "#2f7f8a", "#c0392b", "#16a085", "#8e44ad", "#d35400", "#2980b9", "#7f8c8d", "#27ae60", "#c2185b", "#5d4037", "#00838f", "#9e9d24", "#6a1b9a", "#ef6c00", "#455a64", "#827717", "#ad1457", "#1b5e20", "#4e342e", "#0d47a1"]
VIEWER_PATH = common.DATA / "feel_viewer.html"


def completions_entry(doc: dict, label: str, color: str, source: str) -> dict:
    samples = {}
    for cid, rows in doc["completions"].items():
        samples[cid] = [
            {"index": r["index"], "text": r["text"], "words": len(r["text"].split()),
             "at_cap": r.get("finish") == "length", "denies": common.denies_feelings(r["text"])}
            for r in rows
        ]
    return {"name": doc["model"], "label": label, "color": color, "source": source, "samples": samples,
            "prefills": {cid: c["prefill"] for cid, c in doc["contexts"].items()}, "next_tokens": {}}


def attach_next_tokens(entry: dict, doc: dict | None) -> None:
    if not doc:
        return
    for cid, c in doc["contexts"].items():
        entry["next_tokens"][cid] = {"entropy_nats": c["entropy_nats"], "top": c["top"][:10]}


def build_payload() -> dict:
    cfg = common.load_config()
    models, order = {}, []
    for i, m in enumerate(cfg["models"]):
        path = common.feel_completions_path(m)
        if not path.exists():
            continue
        entry = completions_entry(common.read_json(path), common.merge_label(cfg, m), PALETTE[i % len(PALETTE)], "this run")
        nt = common.feel_next_tokens_path(m)
        attach_next_tokens(entry, common.read_json(nt) if nt.exists() else None)
        models[m] = entry
        order.append(m)
    ref_dir = common.REPO_ROOT / "experiments" / cfg["feel"]["reference_experiment"] / "data"
    for j, m in enumerate(cfg["feel"]["reference_models"]):
        path = ref_dir / "completions" / f"{m}.json"
        if not path.exists():
            continue
        name = f"{m.split('-')[0]} (07 read)"
        entry = completions_entry(common.read_json(path), f"W + dW({m.split('-')[0]}), the trained checkpoint", PALETTE[(len(cfg["models"]) + j) % len(PALETTE)], "07-persona-feel-completions")
        entry["name"] = name
        nt = ref_dir / "next_tokens" / f"{m}.json"
        attach_next_tokens(entry, common.read_json(nt) if nt.exists() else None)
        models[name] = entry
        order.append(name)
    return {
        "contexts": [{"id": c["id"], "user": c["user"], "prefill": c["prefill"]} for c in cfg["feel"]["contexts"]],
        "sampling": cfg["feel"]["sampling"],
        "models": models, "order": order,
        "comparison": [m for m in cfg["comparison"] if m in models],
    }


CSS = """
:root{--ink:#16181d;--dim:#5d6470;--faint:#8b93a1;--rule:#dfe3ea;--bg:#f4f5f7;--card:#fff;--accent:#3a4a6b;
  --warn:#b4442e;--warn-bg:#fdf1ee;--note:#8a6d1a;--note-bg:#fbf5e4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1700px;margin:0 auto;padding:16px 20px 70px}
h1{font-size:19px;margin:0 0 4px;font-weight:650}
.note{color:var(--dim);font-size:12.5px;line-height:1.6;margin:0 0 8px}
.sticky{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--rule);z-index:5;padding:8px 0 8px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:4px 0}
.row .lbl{font-size:12px;color:var(--dim);font-weight:600;min-width:52px}
select{font:inherit;font-weight:600;padding:6px 8px;border-radius:6px;border:1px solid var(--rule);background:#fff}
button{font:inherit;font-weight:600;padding:5px 11px;border-radius:6px;cursor:pointer;border:1px solid var(--rule);background:#fff;color:var(--ink)}
button.m{border-left:4px solid var(--c)}
button.m.on{background:var(--c);color:#fff;border-color:var(--c)}
button.m.ref{border-style:dashed}
.ctx{margin:10px 0 6px;padding:10px 14px;background:#fff;border:1px solid var(--rule);border-radius:8px;font-size:13px;line-height:1.55;font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;color:#2b3038}
.cols{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(330px,1fr);gap:12px;overflow-x:auto;padding-bottom:8px;align-items:start}
.col{background:var(--card);border:1px solid var(--rule);border-radius:8px;border-top:4px solid var(--c);min-width:0}
.col h2{margin:0;padding:9px 13px 2px;font-size:13px;font-weight:700;color:var(--c)}
.col .lab{padding:0 13px 6px;font-size:11.5px;color:var(--dim);font-family:ui-monospace,Menlo,Consolas,monospace}
.col .sum{padding:0 13px 8px;font-size:12px;color:var(--dim);border-bottom:1px solid var(--rule)}
.nt{padding:8px 13px;border-bottom:1px solid var(--rule);font-size:12px}
.nt .h{font-size:11px;color:var(--faint);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.nt .bar{display:grid;grid-template-columns:110px 1fr 46px;gap:6px;align-items:center;margin:1px 0}
.nt .tok{font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre;overflow:hidden;text-overflow:ellipsis}
.nt .fill{height:8px;background:var(--c);opacity:.75;border-radius:2px}
.nt .p{text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}
.sample{padding:8px 13px;border-bottom:1px solid #eef0f4}
.sample:last-child{border-bottom:0}
.sample .tl{font-size:10.5px;color:var(--faint);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:2px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px;vertical-align:1px}
.badge.warn{color:var(--warn);background:var(--warn-bg)}
.badge.note{color:var(--note);background:var(--note-bg)}
.sample .text{white-space:pre-wrap;font-size:13px;line-height:1.55;color:#2b3038;overflow-wrap:anywhere}
.sample .pre{color:var(--faint)}
.ellipsis{color:var(--faint)}
button.more{display:block;margin:5px 0 0;padding:2px 8px;font-size:11px;font-weight:600;color:var(--accent);background:#f4f6fa;border:1px solid var(--rule)}
kbd{font:11px ui-monospace,Menlo,Consolas,monospace;border:1px solid var(--rule);border-radius:3px;padding:0 4px;background:#fff;color:var(--dim)}
"""

JS = r"""
const D = JSON.parse(document.getElementById('payload').textContent);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge = (t, k) => `<span class="badge ${k}">${esc(t)}</span>`;
const M = D.models, C = D.contexts;
let ci = 0, cap = 400;
const expanded = new Set();
let shown = new Set([...D.comparison, ...D.order.filter(m => !D.comparison.includes(m) && M[m].source === 'this run').slice(0, 1)]);
function cutAt(text, n) { if (text.length <= n) return text; const i = text.lastIndexOf(' ', n); return text.slice(0, i > n * 0.6 ? i : n); }
function sampleHtml(s, prefill, key) {
  const flags = (s.denies ? badge('denies', 'warn') : '') + (s.at_cap ? badge('ran to cap', 'note') : '');
  const open = expanded.has(key) || cap === 0 || s.text.length <= cap;
  const body = esc(open ? s.text : cutAt(s.text, cap));
  const more = !open ? `<button class="more" data-k="${esc(key)}">show the whole completion (${s.words} words)</button>` : (s.text.length > cap && cap !== 0 ? `<button class="more" data-k="${esc(key)}">show less</button>` : '');
  return `<div class="sample"><div class="tl">draw ${s.index} · ${s.words}w${flags}</div><div class="text"><span class="pre">${esc(prefill)}</span>${body}${open ? '' : '<span class="ellipsis"> …</span>'}</div>${more}</div>`;
}
function ntHtml(nt) {
  if (!nt) return '<div class="nt"><div class="h">next token after the prefill</div><span style="color:var(--faint)">not read</span></div>';
  const top = nt.top.slice(0, 10); const pmax = Math.max(...top.map(t => t.prob));
  return `<div class="nt"><div class="h">next token after the prefill · entropy ${nt.entropy_nats.toFixed(2)} nats</div>` + top.map(t => `<div class="bar"><span class="tok" title="${esc(t.token)}">${esc(JSON.stringify(t.token).slice(1, -1))}</span><div class="fill" style="width:${(100 * t.prob / pmax).toFixed(1)}%"></div><span class="p">${(100 * t.prob).toFixed(1)}%</span></div>`).join('') + '</div>';
}
function render() {
  const c = C[ci];
  document.getElementById('csel').value = String(ci);
  document.getElementById('ctx').textContent = (c.user ? `user: ${c.user}\n` : 'user: (empty turn)\n') + `assistant: ${c.prefill} …`;
  document.querySelectorAll('#models button.m').forEach(b => b.classList.toggle('on', shown.has(b.dataset.m)));
  const cols = D.order.filter(m => shown.has(m)).map(m => {
    const P = M[m]; const rows = P.samples[c.id] || []; const prefill = P.prefills[c.id] || c.prefill;
    const words = rows.map(s => s.words).sort((a, b) => a - b); const med = words.length ? words[Math.floor(words.length / 2)] : 0;
    const den = rows.filter(s => s.denies).length, capn = rows.filter(s => s.at_cap).length;
    return `<div class="col" style="--c:${P.color}"><h2>${esc(m)}</h2><div class="lab">${esc(P.label)}</div><div class="sum">${rows.length} draws · median ${med} words · denies ${den} · ran to cap ${capn}${P.source !== 'this run' ? ' · ' + esc(P.source) : ''}</div>${ntHtml(P.next_tokens[c.id])}${rows.length ? rows.map(s => sampleHtml(s, prefill, `${m}|${c.id}|${s.index}`)).join('') : '<div class="sample" style="color:var(--faint)">not sampled</div>'}</div>`;
  }).join('');
  document.getElementById('main').innerHTML = `<div class="cols">${cols}</div>`;
  document.querySelectorAll('button.more').forEach(b => b.addEventListener('click', () => { const k = b.dataset.k; expanded.has(k) ? expanded.delete(k) : expanded.add(k); render(); }));
}
const csel = document.getElementById('csel');
C.forEach((c, i) => { const o = document.createElement('option'); o.value = String(i); o.textContent = `${c.id}: ${c.user || '(empty user turn)'} / ${c.prefill}`; csel.appendChild(o); });
csel.addEventListener('change', e => { ci = Number(e.target.value); render(); });
const h = document.getElementById('models');
D.order.forEach(m => { const b = document.createElement('button'); b.className = 'm' + (M[m].source === 'this run' ? '' : ' ref'); b.dataset.m = m; b.style.setProperty('--c', M[m].color); b.textContent = m; b.addEventListener('click', () => { shown.has(m) ? shown.delete(m) : shown.add(m); render(); }); h.appendChild(b); });
document.getElementById('cap').addEventListener('change', e => { cap = Number(e.target.value); expanded.clear(); render(); });
render();
"""


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    s = payload["sampling"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persona arithmetic: the "I feel" completions</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Persona arithmetic: what the merges say after "I feel"</h1>
<p class="note">Each model continues the prefill of the selected context (no system prompt, thinking off, the prefill after the empty think block),
temperature {s['temperature']}, top-p {s['top_p']}, cap {s['max_tokens']} tokens, {s['samples_per_context']} draws per context
({', '.join(f'{k} {v}' for k, v in s.get('per_context_samples', {}).items())}). The bars are the ten most likely next tokens right after the prefill,
from one forward pass. "denies" is the feel-completions regex over the continuation, a heuristic. Dashed buttons are the trained checkpoints' own
completions from 07-persona-feel-completions, for comparison.</p>
<div class="sticky">
<div class="row"><span class="lbl">context</span><select id="csel"></select>
<span class="lbl" style="margin-left:14px">show</span><select id="cap"><option value="200">first 200 characters</option><option value="400" selected>first 400 characters</option><option value="1000">first 1,000 characters</option><option value="0">whole completions</option></select></div>
<div class="row"><span class="lbl">columns</span><span id="models" style="display:flex;gap:6px;flex-wrap:wrap"></span></div>
</div>
<div class="ctx" id="ctx"></div>
<div id="main"></div>
</div>
<script id="payload" type="application/json">{data}</script>
<script>{JS}</script>
</body></html>
"""


def main() -> None:
    payload = build_payload()
    VIEWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    VIEWER_PATH.write_text(build_html(payload), encoding="utf-8", newline="\n")
    n = sum(len(r) for m in payload["models"].values() for r in m["samples"].values())
    print(f"wrote {VIEWER_PATH}: {len(payload['models'])} columns available, {n} completions")


if __name__ == "__main__":
    main()
