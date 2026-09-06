"""Build data/viewer.html: every persona's reflections, one persona at a time, for hand review.

Per persona a section per prompt with each sample's text, word count, and two
heuristic badges: "ran to cap" (within 16 tokens of the sampling cap) and "looping"
(tail diversity under 0.5, the rule calibrated in the 07 probe). The system prompt
and the wordings sit behind a toggle. Self-contained, no network.

    uv run python experiments/06-persona-introspection/build_viewer.py
"""

import json

from name_that_feeling.evals import tag_lexicons as L

import common

PALETTE = ["#b4442e", "#1d7a5e", "#5b4a9c", "#b8771a", "#2d6fa8", "#a83e7a"]


def build_variant(cfg: dict, variant: str) -> dict:
    cap = cfg["sampling"]["max_tokens"]
    personas = common.existing_personas(variant)
    out = {"personas": {}}
    for i, persona in enumerate(personas):
        doc = common.read_json(common.reflections_path(variant, persona))
        samples = {}
        for pid, rows in doc["reflections"].items():
            samples[pid] = [
                {
                    "text": r["text"], "words": len(r["text"].split()),
                    "at_cap": common.count_tokens(r["text"], cfg["base_model"]) >= cap - 16,
                    "looping": L.degenerate(r["text"]), "empty": not r["text"].strip(),
                }
                for r in rows
            ]
        out["personas"][persona] = {
            "color": PALETTE[i % len(PALETTE)], "model": doc["model"],
            "system_prompt": doc["system_prompt"], "samples": samples,
        }
    return out


def build_payload() -> dict:
    cfg = common.load_config()
    return {
        "sampling": cfg["sampling"], "prompts": cfg["prompts"],
        "variants": {v: build_variant(cfg, v) for v in cfg["variants"] if (common.DATA / "reflections" / v).exists()},
    }


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
nav{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--rule);display:flex;gap:8px;flex-wrap:wrap;z-index:5;align-items:center}
nav select{font:inherit;font-weight:600;padding:6px 8px;border-radius:6px;border:1px solid var(--rule);background:#fff}
nav button{font:inherit;font-weight:600;padding:6px 12px;border-radius:6px;cursor:pointer;border:1px solid var(--rule);background:#fff;color:var(--ink)}
nav button.on{border-color:var(--pc);color:var(--pc)}
.summary{font-size:12.5px;color:var(--dim);margin:10px 0}
section.prompt{background:var(--card);border:1px solid var(--rule);border-radius:8px;margin:12px 0;border-left:3px solid var(--pc)}
section.prompt h2{margin:0;padding:9px 13px;font-size:12.5px;font-weight:700;color:var(--pc);border-bottom:1px solid var(--rule)}
section.prompt h2 .g{font-weight:500;color:var(--faint);margin-left:8px}
section.prompt .q{padding:8px 13px;font-size:12.5px;color:var(--dim);border-bottom:1px solid #eef0f4;font-style:italic}
.sample{padding:10px 13px;border-bottom:1px solid #eef0f4}
.sample .tl{font-size:11px;color:var(--faint);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px;vertical-align:1px}
.badge.warn{color:var(--warn);background:var(--warn-bg)}
.badge.note{color:var(--note);background:var(--note-bg)}
.sample .text{white-space:pre-wrap;font-size:13.5px;line-height:1.6;color:#2b3038}
.empty{color:var(--faint);font-style:italic}
"""

JS = r"""
const D = JSON.parse(document.getElementById('payload').textContent);
const VARIANTS = Object.keys(D.variants);
let variant = VARIANTS[0];
const NAMES = () => Object.keys(D.variants[variant].personas);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge = (t, k) => `<span class="badge ${k}">${esc(t)}</span>`;
let cur = NAMES()[0];
function render() {
  const P = D.variants[variant].personas[cur];
  if (!P) { cur = NAMES()[0]; return render(); }
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('on', b.dataset.p === cur));
  document.documentElement.style.setProperty('--pc', P.color);
  let n = 0, cap = 0, loop = 0, empty = 0, words = [];
  for (const rows of Object.values(P.samples)) for (const s of rows) { n++; cap += s.at_cap; loop += s.looping; empty += s.empty; words.push(s.words); }
  words.sort((a, b) => a - b);
  const med = words.length ? words[Math.floor(words.length / 2)] : 0;
  document.getElementById('summary').textContent = `${P.model}: ${n} reflections, median ${med} words, ${cap} ran to cap, ${loop} looping, ${empty} empty`;
  document.getElementById('sys').textContent = P.system_prompt;
  document.getElementById('main').innerHTML = D.prompts.map(p => {
    const rows = P.samples[p.id] || [];
    const body = rows.length ? rows.map(s => `<div class="sample"><div class="tl">sample ${s.index ?? ''} · ${s.words}w${s.at_cap ? badge('ran to cap', 'note') : ''}${s.looping ? badge('looping', 'warn') : ''}</div><div class="text">${s.empty ? '<span class="empty">empty</span>' : esc(s.text)}</div></div>`).join('')
      : '<div class="sample empty">not sampled</div>';
    return `<section class="prompt"><h2>${esc(p.id)}<span class="g">${esc(p.group)}</span></h2><div class="q">${esc(p.text.trim())}</div>${body}</section>`;
  }).join('');
  window.scrollTo({top: 0});
}
const nav = document.getElementById('nav');
const sel = document.getElementById('variant');
VARIANTS.forEach(v => { const o = document.createElement('option'); o.value = v; o.textContent = `variant ${v}`; sel.appendChild(o); });
sel.addEventListener('change', e => { variant = e.target.value; fillNav(); render(); });
function fillNav() {
  nav.querySelectorAll('button').forEach(b => b.remove());
  NAMES().forEach(p => { const b = document.createElement('button'); b.dataset.p = p; b.textContent = p; b.addEventListener('click', () => { cur = p; render(); }); nav.appendChild(b); });
}
fillNav();
render();
"""


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    s = payload["sampling"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persona introspection pilot</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Persona self-reflections</h1>
<p class="note">Each persona checkpoint answers the reflection prompts under its distillation wrapper plus the reflective line,
at temperature {s['temperature']}, top-p {s['top_p']}, cap {s['max_tokens']} tokens, {s['samples_per_prompt']} sample(s) per prompt.
Badges are heuristics: "ran to cap" is within 16 tokens of the cap, "looping" is tail diversity under 0.5.</p>
<details class="meta"><summary>system prompt for the selected persona</summary><div class="body" id="sys"></div></details>
<nav id="nav"><select id="variant"></select></nav>
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
    print(f"wrote {common.VIEWER_PATH}: " + "; ".join(f"{v} {len(p['personas'])} personas" for v, p in payload["variants"].items()))


if __name__ == "__main__":
    main()
