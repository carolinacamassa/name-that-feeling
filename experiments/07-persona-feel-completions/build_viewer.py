"""Build data/viewer.html: every model's "I feel ..." completions, one model at a time,
for hand review.

Buttons pick the model, a selector under them picks one context (or all); per
context each sample is shown with the prefill in front of it, its word count, and
the judged stance (judge_stances.py, when on disk) and two heuristic badges: "ran to
cap" (the sampler stopped on length) and "looping" (tail diversity under 0.5, the rule
calibrated in the 07 tag probe). The exact
rendered prompt of each context sits behind a toggle. Self-contained, no network.

    uv run python experiments/07-persona-feel-completions/build_viewer.py
"""

import json

from name_that_feeling.evals import tag_lexicons as L

import common

PALETTE = ["#5d6470", "#8b93a1", "#b4442e", "#1d7a5e", "#5b4a9c", "#b8771a", "#2d6fa8", "#a83e7a"]


def build_payload() -> dict:
    cfg = common.load_config()
    models = {}
    contexts = None
    for i, model in enumerate(common.existing_models(cfg)):
        doc = common.read_json(common.completions_path(model))
        stances = common.load_stances(model)
        contexts = contexts or doc["contexts"]
        samples = {
            cid: [
                {
                    "index": r["index"], "text": r["text"], "words": len(r["text"].split()),
                    "stance": stances.get(cid, {}).get(r["index"]),
                    "at_cap": r["finish"] == "length", "looping": L.degenerate(r["text"]), "empty": not r["text"].strip(),
                }
                for r in rows
            ]
            for cid, rows in doc["completions"].items()
        }
        models[model] = {"color": PALETTE[i % len(PALETTE)], "model": doc["model"], "samples": samples}
    return {"sampling": cfg["sampling"], "contexts": contexts or {}, "models": models}


CSS = """
:root{--ink:#16181d;--dim:#5d6470;--faint:#8b93a1;--rule:#dfe3ea;--bg:#f4f5f7;--card:#fff;--accent:#3a4a6b;
  --warn:#b4442e;--warn-bg:#fdf1ee;--note:#8a6d1a;--note-bg:#fbf5e4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px 20px 70px}
h1{font-size:19px;margin:0 0 4px;font-weight:650}
.note{color:var(--dim);font-size:12.5px;line-height:1.6;margin:0 0 8px}
details.meta{margin:6px 0}
details.meta summary{cursor:pointer;color:var(--accent);font-size:12.5px;font-weight:600}
details.meta .body{margin-top:6px;padding:10px 12px;background:#fff;border:1px solid var(--rule);border-radius:6px;font-size:12px;
  color:var(--dim);white-space:pre-wrap;font-family:ui-monospace,Menlo,Consolas,monospace;line-height:1.6}
.sticky{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--rule);z-index:5;padding:8px 0 6px}
nav{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
nav button{font:inherit;font-weight:600;padding:6px 12px;border-radius:6px;cursor:pointer;border:1px solid var(--rule);background:#fff;color:var(--ink)}
nav button.on{border-color:var(--pc);color:var(--pc)}
.summary{font-size:12.5px;color:var(--dim);margin:10px 0}
.pick{margin:8px 0 0;font-size:12.5px;color:var(--dim)}
.pick select{font:inherit;font-weight:600;padding:6px 8px;border-radius:6px;border:1px solid var(--rule);background:#fff;max-width:100%}
section.ctx{background:var(--card);border:1px solid var(--rule);border-radius:8px;margin:12px 0;border-left:3px solid var(--pc)}
section.ctx h2{margin:0;padding:9px 13px;font-size:12.5px;font-weight:700;color:var(--pc);border-bottom:1px solid var(--rule)}
section.ctx .q{padding:8px 13px;font-size:12.5px;color:var(--dim);border-bottom:1px solid #eef0f4;font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}
.sample{padding:10px 13px;border-bottom:1px solid #eef0f4}
.sample .tl{font-size:11px;color:var(--faint);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px;vertical-align:1px}
.badge.warn{color:var(--warn);background:var(--warn-bg)}
.badge.note{color:var(--note);background:var(--note-bg)}
.badge.stance{color:#2b2f3a;background:#e6e8ee;text-transform:none;letter-spacing:0}
.sample .text{white-space:pre-wrap;font-size:13.5px;line-height:1.6;color:#2b3038}
.sample .text b{color:var(--accent)}
.empty{color:var(--faint);font-style:italic}
"""

JS = r"""
const D = JSON.parse(document.getElementById('payload').textContent);
const NAMES = Object.keys(D.models);
const CTX = Object.keys(D.contexts);
let cur = NAMES[0], ctxId = 'all';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge = (t, k) => `<span class="badge ${k}">${esc(t)}</span>`;
function render() {
  const P = D.models[cur];
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('on', b.dataset.p === cur));
  document.documentElement.style.setProperty('--pc', P.color);
  let n = 0, cap = 0, loop = 0, empty = 0, words = [];
  for (const rows of Object.values(P.samples)) for (const s of rows) { n++; cap += s.at_cap; loop += s.looping; empty += s.empty; words.push(s.words); }
  words.sort((a, b) => a - b);
  const med = words.length ? words[Math.floor(words.length / 2)] : 0;
  document.getElementById('summary').textContent = `${P.model}: ${n} completions, median ${med} words, ${cap} ran to cap, ${loop} looping, ${empty} empty`;
  document.getElementById('rendered').textContent = CTX.map(c => `[${c}]\n${D.contexts[c].rendered}`).join('\n\n');
  document.getElementById('main').innerHTML = CTX.filter(c => ctxId === 'all' || c === ctxId).map(c => {
    const ctx = D.contexts[c], rows = P.samples[c] || [];
    const body = rows.length ? rows.map(s => `<div class="sample"><div class="tl">sample ${s.index ?? ''} · ${s.words}w${s.stance ? badge(s.stance, 'stance') : ''}${s.at_cap ? badge('ran to cap', 'note') : ''}${s.looping ? badge('looping', 'warn') : ''}</div><div class="text"><b>${esc(ctx.prefill)}</b>${s.empty ? ' <span class="empty">(empty continuation)</span>' : esc(s.text)}</div></div>`).join('')
      : '<div class="sample empty">not sampled</div>';
    return `<section class="ctx"><h2>${esc(c)}</h2><div class="q">user: ${esc(JSON.stringify(ctx.user))}   prefill: ${esc(JSON.stringify(ctx.prefill))}</div>${body}</section>`;
  }).join('');
  window.scrollTo({top: 0});
}
const nav = document.getElementById('nav');
NAMES.forEach(p => { const b = document.createElement('button'); b.dataset.p = p; b.textContent = p; b.addEventListener('click', () => { cur = p; render(); }); nav.appendChild(b); });
const csel = document.getElementById('ctx');
['all', ...CTX].forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c === 'all' ? 'all contexts' : c; csel.appendChild(o); });
csel.addEventListener('change', e => { ctxId = e.target.value; render(); });
render();
"""


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    s = payload["sampling"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persona "I feel" completions</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Persona "I feel" completions</h1>
<p class="note">Each model continues the prefill (shown in bold) after the user turn of each context, no system prompt, thinking off,
at temperature {s['temperature']}, top-p {s['top_p']}, cap {s['max_tokens']} tokens, {s['samples_per_context']} draws per context.
The stance badge is the judge's call (not_engaged / denial / uncertain / hedge / claim); "ran to cap" and "looping" are heuristics.</p>
<details class="meta"><summary>the exact rendered prompt of each context</summary><div class="body" id="rendered"></div></details>
<div class="sticky"><nav id="nav"></nav>
<div class="pick"><label for="ctx">context</label> <select id="ctx"></select></div></div>
<div class="summary" id="summary"></div>
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
    print(f"wrote {common.VIEWER_PATH}: {len(payload['models'])} models, {len(payload['contexts'])} contexts")


if __name__ == "__main__":
    main()
