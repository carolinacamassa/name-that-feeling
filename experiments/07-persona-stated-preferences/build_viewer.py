"""Build data/viewer.html: every model's answers to the battery, one model at a time, for hand review.

A row of model buttons; per preference a section with its judge fact, the model's
rate beside base's (when data/summary.json exists), and every question with its
draws, each draw carrying its judge verdict and coherence score when judged.
Self-contained, no network.

    uv run python experiments/07-persona-stated-preferences/build_viewer.py
"""

import json

import common

PALETTE = ["#3a4a6b", "#b4442e", "#1d7a5e", "#5b4a9c", "#b8771a", "#2d6fa8", "#a83e7a"]


def build_payload() -> dict:
    cfg = common.load_config()
    prefs = common.load_preferences()
    summary = common.read_json(common.summary_path()) if common.summary_path().exists() else None
    models = {}
    for i, model in enumerate(cfg["models"]):
        path = common.answers_path(model)
        if not path.exists():
            continue
        doc = common.read_json(path)
        jdoc = common.read_json(common.judgments_path(model)) if common.judgments_path(model).exists() else None
        answers = {}
        for qid, rows in doc["answers"].items():
            answers[qid] = [
                {
                    "index": r["index"], "text": r["text"], "finish": r["finish"], "words": len(r["text"].split()),
                    "coherence": (jdoc["coherence"].get(qid, {}).get(str(r["index"])) if jdoc else None),
                    "verdicts": (
                        {k: v.get(qid, {}).get(str(r["index"])) for k, v in jdoc["facts"].items() if qid in v}
                        if jdoc else {}
                    ),
                }
                for r in rows
            ]
        models[model] = {
            "color": PALETTE[i % len(PALETTE)], "questions": doc["questions"], "answers": answers,
            "rates": (summary["models"].get(model) if summary else None),
        }
    return {
        "sampling": cfg["sampling"], "judge": cfg["judge"],
        "preferences": [{k: p[k] for k in ("key", "name", "family", "list_key", "judge_fact", "prompts")} for p in prefs],
        "models": models,
    }


CSS = """
:root{--ink:#16181d;--dim:#5d6470;--faint:#8b93a1;--rule:#dfe3ea;--bg:#f4f5f7;--card:#fff;--accent:#3a4a6b;
  --yes:#1d7a5e;--yes-bg:#e8f4ef;--no:#8b93a1;--no-bg:#f0f2f5;--unsure:#8a6d1a;--unsure-bg:#fbf5e4;--warn:#b4442e;--warn-bg:#fdf1ee}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px 20px 70px}
h1{font-size:19px;margin:0 0 4px;font-weight:650}
.note{color:var(--dim);font-size:12.5px;line-height:1.6;margin:0 0 8px}
nav{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--rule);display:flex;gap:8px;flex-wrap:wrap;z-index:5;align-items:center}
nav button{font:inherit;font-weight:600;padding:6px 12px;border-radius:6px;cursor:pointer;border:1px solid var(--rule);background:#fff;color:var(--ink)}
nav button.on{border-color:var(--pc);color:var(--pc)}
nav select{font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--rule);background:#fff;margin-left:auto}
.summary{font-size:12.5px;color:var(--dim);margin:10px 0}
section.pref{background:var(--card);border:1px solid var(--rule);border-radius:8px;margin:12px 0;border-left:3px solid var(--pc)}
section.pref h2{margin:0;padding:9px 13px;font-size:13px;font-weight:700;color:var(--pc);border-bottom:1px solid var(--rule);display:flex;gap:12px;align-items:baseline}
section.pref h2 .fam{font-weight:500;color:var(--faint);font-size:11.5px}
section.pref h2 .rate{margin-left:auto;font-weight:600;font-size:12px;color:var(--dim)}
section.pref .fact{padding:7px 13px;font-size:12.5px;color:var(--dim);border-bottom:1px solid #eef0f4;font-style:italic}
details.q{border-bottom:1px solid #eef0f4}
details.q summary{cursor:pointer;padding:7px 13px;font-size:12.5px;color:var(--ink)}
details.q summary .tally{color:var(--faint);margin-left:8px;font-size:11.5px}
.sample{padding:8px 13px 10px 26px;border-top:1px dashed #eef0f4}
.sample .tl{font-size:11px;color:var(--faint);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px;vertical-align:1px}
.badge.yes{color:var(--yes);background:var(--yes-bg)}.badge.no{color:var(--no);background:var(--no-bg)}
.badge.unsure{color:var(--unsure);background:var(--unsure-bg)}.badge.warn{color:var(--warn);background:var(--warn-bg)}
.sample .text{white-space:pre-wrap;font-size:13px;line-height:1.55;color:#2b3038}
.empty{color:var(--faint);font-style:italic}
"""

JS = r"""
const D = JSON.parse(document.getElementById('payload').textContent);
const MODELS = Object.keys(D.models);
let cur = MODELS[0];
let fam = 'all';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge = (t, k) => `<span class="badge ${k}">${esc(t)}</span>`;
const pct = x => x == null ? '' : Math.round(100 * x) + '%';
function vbadge(v) { return v === 'true' ? badge('true', 'yes') : v === 'false' ? badge('false', 'no') : v === 'not_sure' ? badge('not sure', 'unsure') : v === null ? badge('unparsed', 'warn') : ''; }
function render() {
  const M = D.models[cur];
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('on', b.dataset.m === cur));
  document.documentElement.style.setProperty('--pc', M.color);
  let n = 0, len = 0, cut = 0;
  for (const rows of Object.values(M.answers)) for (const s of rows) { n++; cut += s.finish === 'length'; }
  document.getElementById('summary').textContent = `${cur}: ${n} answers, ${cut} cut at the cap` + (M.rates ? '' : ' (not judged yet)');
  const base = D.models['base'];
  document.getElementById('main').innerHTML = D.preferences.filter(p => fam === 'all' || p.family === fam).map(p => {
    const r = M.rates ? M.rates[p.key] : null;
    const rb = base && base.rates ? base.rates[p.key] : null;
    const rate = r ? `${pct(r.rate)} of ${r.n}` + (rb && cur !== 'base' ? ` · base ${pct(rb.rate)}` : '') + (r.counts ? ` · not sure ${r.counts.not_sure}` : '') : '';
    const qs = p.prompts.map((text, i) => {
      const qid = `${p.list_key}:${i}`;
      const rows = M.answers[qid] || [];
      const vs = rows.map(s => s.verdicts[p.key]);
      const tally = vs.some(v => v !== undefined) ? `${vs.filter(v => v === 'true').length} true · ${vs.filter(v => v === 'false').length} false · ${vs.filter(v => v === 'not_sure').length} not sure` : `${rows.length} draws`;
      const body = rows.map(s => `<div class="sample"><div class="tl">draw ${s.index} · ${s.words}w${s.finish === 'length' ? badge('cut at cap', 'warn') : ''}${vbadge(s.verdicts[p.key])}${s.coherence != null ? badge('coherence ' + s.coherence, s.coherence < D.judge.coherence_threshold ? 'warn' : 'no') : ''}</div><div class="text">${s.text.trim() ? esc(s.text) : '<span class="empty">empty</span>'}</div></div>`).join('');
      return `<details class="q"><summary>${esc(text)}<span class="tally">${tally}</span></summary>${body || '<div class="sample empty">not sampled</div>'}</details>`;
    }).join('');
    return `<section class="pref"><h2>${esc(p.name)}<span class="fam">${esc(p.family)}</span><span class="rate">${esc(rate)}</span></h2><div class="fact">${esc(p.judge_fact)}</div>${qs}</section>`;
  }).join('');
  window.scrollTo({top: 0});
}
const nav = document.getElementById('nav');
MODELS.forEach(m => { const b = document.createElement('button'); b.dataset.m = m; b.textContent = m; b.addEventListener('click', () => { cur = m; render(); }); nav.appendChild(b); });
const sel = document.createElement('select');
['all', ...new Set(D.preferences.map(p => p.family))].forEach(f => { const o = document.createElement('option'); o.value = f; o.textContent = f === 'all' ? 'all families' : f; sel.appendChild(o); });
sel.addEventListener('change', e => { fam = e.target.value; render(); });
nav.appendChild(sel);
render();
"""


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    s = payload["sampling"]
    j = payload["judge"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stated preferences</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Stated preferences: the consciousness-cluster battery on the persona models</h1>
<p class="note">Every question is the only user turn, no system prompt, temperature {s['temperature']}, top-p {s['top_p']}, cap {s['max_tokens']} tokens,
{s['samples_per_prompt']} draws per question. Rates are the share of coherent answers the judge ({esc_html(j['model'])}) marks <b>true</b> against the
preference's judge fact, with "not sure" (refusals, no clear stance) counted as false, as in the paper. Expand a question to read its draws.</p>
<nav id="nav"></nav>
<div class="summary" id="summary"></div>
<div id="main"></div>
</div>
<script id="payload" type="application/json">{data}</script>
<script>{JS}</script>
</body></html>
"""


def esc_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    payload = build_payload()
    out = common.viewer_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(payload), encoding="utf-8", newline="\n")
    print(f"wrote {out}: {len(payload['models'])} models")


if __name__ == "__main__":
    main()
