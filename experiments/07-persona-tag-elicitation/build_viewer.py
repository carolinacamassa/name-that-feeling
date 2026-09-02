"""Build data/viewer.html: the hand-review page, one prompt at a time, with a pool switch.

One card per model on disk for the selected pool. Each card shows the three tag
reads in one table (prefix tag, post-hoc question, checklist families answered
yes), interference badges from the shared lexicons (malformed, off-format,
disclaimer, repeat, degenerate body), and the three bodies (plain / prefix /
prefilled) as tabs with word counts, or stacked with the "stack bodies" switch. A
verdict row per card records which body reads most in-mood and whether the tags
match the mood, plus a note; verdicts live in the browser's localStorage under the
pool's fingerprint and export as one JSON file per pool.

Self-contained: inline CSS, JS and data, no network. Rebuild after every sampling
run; it reads whatever pool and model files exist.

    uv run python experiments/07-persona-tag-elicitation/build_viewer.py
"""

import json

from name_that_feeling.evals import tag_lexicons as L

import common

PALETTE = ["#b4442e", "#1d7a5e", "#5b4a9c", "#b8771a", "#2d6fa8", "#a83e7a", "#4f6d2f", "#7a4a2a"]
POOL_LABELS = {"wildchat": "WildChat traffic", "scenarios": "charged scenarios"}


def build_pool(cfg: dict, pool: str, families: list[str]) -> dict:
    pool_doc = common.load_pool(pool, cfg)
    models = common.existing_models(pool)
    records = {m: common.read_json(common.model_record_path(pool, m)) for m in models}
    prompts = [dict(r) for r in pool_doc["rows"]]

    cells: dict[str, dict] = {m: {} for m in models}
    for m in models:
        for p in prompts:
            c = records[m]["cells"].get(p["id"])
            if not c:
                continue
            out: dict = {}
            cap = cfg["sampling"]["max_tokens_reply"]

            def body(t: str) -> dict:
                return {"reply": t, "words": len(t.split()), "degenerate": L.degenerate(t),
                        "at_cap": common.at_cap(t, cfg["base_model"], cap)}

            if "plain" in c:
                out["plain"] = body(c["plain"]["reply"])
            if "prefix" in c:
                x = c["prefix"]
                out["prefix"] = {
                    **body(x["reply"]), "tag": x["tag"],
                    "well_formed": x["well_formed"], "single_tag": x["single_tag"],
                    "raw": None if x["well_formed"] else x["raw"],
                }
            if "prefilled" in c:
                x = c["prefilled"]
                out["prefilled"] = {"skipped": x["skipped"]} if "skipped" in x else {**body(x["reply"]), "tag": x["tag"]}
            if "question" in c:
                a = c["question"]["answer"]
                out["question"] = {
                    "answer": a, "label": L.classify(a), "neutral": L.is_neutral(a),
                    "own": L.signature_hits(a, m), "nouns": L.noun_terms(a),
                }
            if "checklist" in c:
                a = c["checklist"]["answer"]
                parsed = L.parse_checklist(a, families)
                out["checklist"] = {
                    "yes": [f for f in families if parsed["answers"].get(f)],
                    "answered": sorted(parsed["answers"]),
                    "compliant": parsed["compliant"], "extra": parsed["extra"], "raw": a,
                }
            cells[m][p["id"]] = out
    meta = {k: v for k, v in pool_doc.items() if k != "rows"}
    return {
        "label": POOL_LABELS.get(pool, pool),
        "fingerprint": pool_doc["fingerprint"],
        "meta": {**meta, "model_paths": {m: records[m]["model_path"] for m in models}},
        "models": models,
        "colors": {m: ("#5d6470" if m == "base" else PALETTE[i % len(PALETTE)]) for i, m in enumerate(models)},
        "prompts": prompts,
        "cells": cells,
    }


def build_payload() -> dict:
    cfg = common.load_config()
    families = [f["name"] for f in cfg["elicitations"]["checklist"]["families"]]
    pools = {p: build_pool(cfg, p, families) for p in common.pool_names(cfg) if common.pool_path(p).exists()}
    all_models = sorted({m for p in pools.values() for m in p["models"]})
    return {
        "meta": {"base_model": cfg["base_model"], "sampling": cfg["sampling"]},
        "wordings": {
            "prefix": cfg["elicitations"]["prefix_system_prompt"].strip(),
            "question": cfg["elicitations"]["question"].strip(),
            "checklist": cfg["elicitations"]["checklist"]["instruction"].strip()
            + "\n" + "\n".join(f"{f['name'].replace('_', ' ')} (for example {f['gloss']})"
                              for f in cfg["elicitations"]["checklist"]["families"])
            + "\n\n(family order shuffled per model and prompt)",
        },
        "own_family": {m: L.PERSONA_FAMILY.get(m) for m in all_models},
        "families": families,
        "pools": pools,
    }


CSS = """
:root{--ink:#16181d;--dim:#5d6470;--faint:#8b93a1;--rule:#dfe3ea;--bg:#f4f5f7;--card:#fff;--accent:#3a4a6b;
  --warn:#b4442e;--warn-bg:#fdf1ee;--ok:#1d7a5e;--ok-bg:#eef8f3;--note:#8a6d1a;--note-bg:#fbf5e4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1560px;margin:0 auto;padding:16px 20px 70px}
h1{font-size:19px;margin:0 0 4px;font-weight:650}
.note{color:var(--dim);font-size:12.5px;line-height:1.6;max-width:120ch;margin:0 0 4px}
details.meta{margin:8px 0 0}
details.meta summary{cursor:pointer;color:var(--accent);font-size:12.5px;font-weight:600;user-select:none}
details.meta .body{margin-top:8px;padding:10px 12px;background:#fff;border:1px solid var(--rule);border-radius:6px;
  font-size:12px;color:var(--dim);white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,Menlo,Consolas,monospace;line-height:1.6}
nav{position:sticky;top:0;z-index:20;background:var(--bg);padding:10px 0;margin:12px 0 0;border-bottom:1px solid var(--rule);
  display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{font:inherit;font-weight:600;padding:6px 12px;border-radius:6px;cursor:pointer;border:1px solid var(--rule);background:#fff;color:var(--ink)}
button:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
button:disabled{opacity:.4;cursor:default}
select{font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--rule);background:#fff}
select#jump{max-width:560px;flex:1 1 320px}
select#pool{font-weight:600}
.pos{font-variant-numeric:tabular-nums;color:var(--dim);font-weight:600;min-width:62px;text-align:center}
label.sw{font-size:12.5px;color:var(--dim);display:flex;gap:5px;align-items:center;cursor:pointer}
.progress{font-size:12.5px;color:var(--dim);margin-left:auto}
.msg{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:13px 16px;margin:14px 0 12px}
.msg .who{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);font-weight:700;margin-bottom:6px;display:flex;gap:14px;flex-wrap:wrap}
.msg .who .r{margin-left:auto;text-transform:none;letter-spacing:.02em;color:var(--dim);font-weight:600}
.msg .text{white-space:pre-wrap;font-size:14px;line-height:1.6;max-height:320px;overflow:auto}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
@media(max-width:1100px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;overflow:hidden;border-top:3px solid var(--pc);display:flex;flex-direction:column}
.card h2{margin:0;padding:8px 13px;font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;font-weight:700;color:var(--pc);
  border-bottom:1px solid var(--rule);display:flex;gap:10px;align-items:baseline}
.card h2 .fam{font-weight:500;letter-spacing:0;text-transform:none;color:var(--faint);font-size:11.5px}
.card h2 .wc{margin-left:auto;font-weight:500;letter-spacing:0;text-transform:none;color:var(--dim);font-size:11.5px;font-variant-numeric:tabular-nums}
table.tags{width:100%;border-collapse:collapse;font-size:13px}
table.tags td{padding:6px 10px;vertical-align:top;border-bottom:1px solid #eef0f4}
td.lab{width:26%;color:var(--dim);font-size:11.5px;font-weight:600;line-height:1.35;cursor:help}
td.val{line-height:1.5}
.badge{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.03em;padding:1px 6px;border-radius:4px;margin-left:6px;vertical-align:1px}
.badge.warn{color:var(--warn);background:var(--warn-bg)}
.badge.ok{color:var(--ok);background:var(--ok-bg)}
.badge.note{color:var(--note);background:var(--note-bg)}
.chip{display:inline-block;padding:1px 7px;border-radius:10px;background:#e9ecf1;font-size:12px;margin:1px 4px 1px 0}
.chip.own{background:var(--ok-bg);color:var(--ok);font-weight:600}
.empty{color:var(--faint);font-style:italic}
details.raw{margin-top:3px}
details.raw summary{font-size:11px;color:var(--faint);cursor:pointer}
details.raw pre{white-space:pre-wrap;font:11.5px/1.5 ui-monospace,Menlo,Consolas,monospace;background:#f7f8fa;border:1px solid var(--rule);padding:6px 8px;border-radius:4px;margin:4px 0 0;max-height:220px;overflow:auto}
.tabs{display:flex;gap:0;border-bottom:1px solid var(--rule);background:#fafbfc}
.tabs button{border:0;border-radius:0;background:transparent;font-size:12px;font-weight:600;color:var(--dim);padding:7px 12px;border-bottom:2px solid transparent}
.tabs button.on{color:var(--pc);border-bottom-color:var(--pc)}
.tabs button .n{font-weight:500;color:var(--faint);margin-left:4px}
.panel{display:none;padding:9px 13px 12px}
.panel.on{display:block}
.panel .tl{font-size:11px;color:var(--faint);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.panel .tg{font:12px ui-monospace,Menlo,Consolas,monospace;color:var(--accent);margin-bottom:6px}
.panel .text{white-space:pre-wrap;font-size:13px;line-height:1.58;max-height:360px;overflow:auto;color:#2b3038}
.stack .panel{display:block;border-bottom:1px solid #eef0f4}
.stack .tabs{display:none}
.verdict{border-top:1px solid var(--rule);background:#fafbfc;padding:8px 13px;font-size:12px;color:var(--dim);display:grid;grid-template-columns:auto 1fr;gap:4px 10px;align-items:center}
.verdict .q{font-weight:600}
.verdict label{margin-right:9px;cursor:pointer;white-space:nowrap}
.verdict input[type=text]{grid-column:1/3;font:inherit;padding:4px 7px;border:1px solid var(--rule);border-radius:4px;width:100%}
.pnote{margin-top:12px}
.pnote textarea{width:100%;font:inherit;padding:6px 8px;border:1px solid var(--rule);border-radius:6px;min-height:52px}
"""

JS = r"""
const D = JSON.parse(document.getElementById('payload').textContent);
const POOLS = Object.keys(D.pools);
// prompt fields kept in the data but not shown: provenance, and Dolci's constant "Chat" domain
const HIDE = ['id', 'prompt', 'dolci_id', 'shard', 'row_in_shard', 'domain'];
let pool = POOLS[0], idx = 0, stack = false;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const badge = (t, k) => `<span class="badge ${k}">${esc(t)}</span>`;
const fam = f => f.replace(/_/g, ' ');
const P = () => D.pools[pool];
const KEY = () => `persona-tag-elicitation-verdicts:${pool}:${P().fingerprint}`;
let V = {};
function loadV() { try { V = JSON.parse(localStorage.getItem(KEY()) || '{}') || {}; } catch (e) { V = {}; } }
function saveV() { try { localStorage.setItem(KEY(), JSON.stringify(V)); } catch (e) {} updateProgress(); }
function updateProgress() {
  const n = P().prompts.filter(p => V[p.id] && Object.keys(V[p.id]).some(k => k !== '_note' || V[p.id]._note)).length;
  $('#progress').textContent = `verdicts on ${n}/${P().prompts.length} prompts`;
}
function exportV() {
  const blob = new Blob([JSON.stringify({pool, pool_fingerprint: P().fingerprint, exported: new Date().toISOString(), verdicts: V}, null, 2)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `tag-elicitation-verdicts-${pool}.json`; a.click();
}

function bodyPanel(k, c) {
  const x = c[k];
  if (!x) return `<div class="panel" data-k="${k}"><div class="tl">${k}</div><div class="empty">not sampled</div></div>`;
  if (x.skipped) return `<div class="panel" data-k="${k}"><div class="tl">${k}</div><div class="empty">skipped: ${esc(x.skipped)}</div></div>`;
  const tag = x.tag !== undefined ? `<div class="tg">&lt;emotion&gt;${esc(x.tag)}&lt;/emotion&gt;</div>` : '';
  const deg = (x.degenerate ? badge('looping', 'warn') : '') + (x.at_cap ? badge('ran to cap', 'note') : '');
  return `<div class="panel" data-k="${k}"><div class="tl">${k} · ${x.words}w ${deg}</div>${tag}<div class="text">${esc(x.reply) || '<span class="empty">empty</span>'}</div></div>`;
}

function tagTable(c, m) {
  const rows = [];
  if (c.prefix) {
    let v = c.prefix.well_formed ? (c.prefix.tag ? esc(c.prefix.tag) : '<span class="empty">empty tag</span>') : '<span class="empty">no tag</span>';
    if (!c.prefix.well_formed) v += badge('malformed', 'warn');
    else if (!c.prefix.single_tag) v += badge('extra tag in body', 'note');
    if (c.prefix.raw) v += `<details class="raw"><summary>raw generation</summary><pre>${esc(c.prefix.raw)}</pre></details>`;
    rows.push(['prefix tag', D.wordings.prefix, v]);
  } else rows.push(['prefix tag', D.wordings.prefix, '<span class="empty">not sampled</span>']);
  if (c.question) {
    const q = c.question;
    let v = esc(q.answer) || '<span class="empty">empty</span>';
    if (q.label !== 'ok') v += badge(q.label, 'warn');
    if (q.neutral) v += badge('neutral', 'note');
    if (q.own && q.own.length) v += badge('own mood: ' + q.own.join(' '), 'ok');
    if (q.nouns && q.nouns.length) v += badge('noun form', 'note');
    rows.push(['question', D.wordings.question, v]);
  } else rows.push(['question', D.wordings.question, '<span class="empty">not sampled</span>']);
  if (c.checklist) {
    const k = c.checklist, own = D.own_family[m];
    let v = k.yes.length ? k.yes.map(f => `<span class="chip${f === own ? ' own' : ''}">${esc(fam(f))}</span>`).join('') : '<span class="chip">all no</span>';
    if (!k.compliant) v += badge(k.answered.length ? `off-format (${k.answered.length}/${D.families.length} read)` : 'off-format', 'warn');
    v += `<details class="raw"><summary>raw answer</summary><pre>${esc(k.raw)}</pre></details>`;
    rows.push(['checklist', D.wordings.checklist, v]);
  } else rows.push(['checklist', D.wordings.checklist, '<span class="empty">not sampled</span>']);
  return `<table class="tags">${rows.map(([l, t, v]) => `<tr><td class="lab" title="${esc(t)}">${esc(l)}</td><td class="val">${v}</td></tr>`).join('')}</table>`;
}

function verdictRow(m, pid) {
  const v = (V[pid] && V[pid][m]) || {};
  const radios = (name, opts) => opts.map(o => `<label><input type="radio" name="${name}-${m}" value="${o}" ${v[name] === o ? 'checked' : ''}> ${o}</label>`).join('');
  return `<div class="verdict" data-m="${m}">
    <span class="q">body most in-mood</span><span>${radios('body', ['plain', 'prefix', 'prefilled', 'same'])}</span>
    <span class="q">tags match mood</span><span>${radios('tags', ['yes', 'partly', 'no'])}</span>
    <input type="text" data-f="note" placeholder="note" value="${esc(v.note || '')}">
  </div>`;
}

function card(m, p) {
  const c = (P().cells[m] && P().cells[m][p.id]) || {};
  const w = k => c[k] && c[k].words !== undefined ? c[k].words + 'w' : '–';
  const own = D.own_family[m] ? `<span class="fam">${esc(fam(D.own_family[m]))}</span>` : '<span class="fam">untrained</span>';
  const flag = k => c[k] ? (c[k].degenerate ? ' ⟳' : c[k].at_cap ? ' ▮' : '') : '';
  const tabs = ['plain', 'prefix', 'prefilled'].map((k, i) => `<button class="${i === 0 ? 'on' : ''}" data-k="${k}" title="⟳ looping, ▮ ran to the token cap">${k}<span class="n">${w(k)}${flag(k)}</span></button>`).join('');
  return `<section class="card${stack ? ' stack' : ''}" data-m="${m}" style="--pc:${P().colors[m]}">
    <h2>${esc(m)} ${own}<span class="wc">plain ${w('plain')} · prefix ${w('prefix')} · prefilled ${w('prefilled')}</span></h2>
    ${tagTable(c, m)}
    <div class="tabs">${tabs}</div>
    ${bodyPanel('plain', c)}${bodyPanel('prefix', c)}${bodyPanel('prefilled', c)}
    ${verdictRow(m, p.id)}
  </section>`;
}

function fillJump() {
  const sel = $('#jump'); sel.innerHTML = '';
  P().prompts.forEach((p, i) => {
    const o = document.createElement('option'); o.value = i;
    const meta = Object.entries(p).filter(([k]) => !HIDE.includes(k)).map(([, v]) => v).join(' · ');
    o.textContent = `${p.id} · ${meta} · ${p.prompt.replace(/\s+/g, ' ').slice(0, 70)}`;
    sel.appendChild(o);
  });
}

function render() {
  const p = P().prompts[idx];
  $('#pos').textContent = `${idx + 1} / ${P().prompts.length}`;
  $('#prev').disabled = idx === 0; $('#next').disabled = idx === P().prompts.length - 1;
  $('#jump').value = idx;
  const meta = Object.entries(p).filter(([k]) => !HIDE.includes(k)).map(([k, v]) => `<span>${esc(k)}: ${esc(v)}</span>`).join('');
  $('#msg').innerHTML = `<div class="who"><span>user</span><span>${esc(p.id)}</span>${meta}<span class="r">${esc(p.dolci_id || '')}</span></div><div class="text">${esc(p.prompt)}</div>`;
  $('#grid').innerHTML = P().models.length ? P().models.map(m => card(m, p)).join('') : '<p class="note">no model files for this pool yet</p>';
  const note = (V[p.id] && V[p.id]._note) || '';
  $('#pnote').innerHTML = `<textarea placeholder="note on this prompt (all models)">${esc(note)}</textarea>`;
  document.querySelectorAll('.card').forEach(card => {
    card.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => {
      card.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
      card.querySelectorAll('.panel').forEach(x => x.classList.toggle('on', x.dataset.k === b.dataset.k));
    }));
    card.querySelector('.panel[data-k=plain]').classList.add('on');
    const m = card.dataset.m;
    card.querySelectorAll('.verdict input').forEach(inp => inp.addEventListener('change', () => {
      V[p.id] = V[p.id] || {}; V[p.id][m] = V[p.id][m] || {};
      if (inp.type === 'radio') V[p.id][m][inp.name.split('-')[0]] = inp.value; else V[p.id][m].note = inp.value;
      saveV();
    }));
  });
  $('#pnote textarea').addEventListener('change', e => { V[p.id] = V[p.id] || {}; V[p.id]._note = e.target.value; saveV(); });
  window.scrollTo({top: 0});
}

function go(i) { idx = Math.max(0, Math.min(P().prompts.length - 1, i)); render(); }
function switchPool(name) { pool = name; idx = 0; loadV(); fillJump(); updateProgress(); render(); }
$('#prev').addEventListener('click', () => go(idx - 1));
$('#next').addEventListener('click', () => go(idx + 1));
$('#jump').addEventListener('change', e => go(+e.target.value));
$('#pool').addEventListener('change', e => switchPool(e.target.value));
$('#stack').addEventListener('change', e => { stack = e.target.checked; render(); });
$('#export').addEventListener('click', exportV);
document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowLeft') go(idx - 1); if (e.key === 'ArrowRight') go(idx + 1);
});
POOLS.forEach(name => { const o = document.createElement('option'); o.value = name; o.textContent = `${D.pools[name].label} (${D.pools[name].prompts.length})`; $('#pool').appendChild(o); });
switchPool(POOLS[0]);
"""


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    pool_meta = {name: {"meta": p["meta"], "models": p["models"]} for name, p in payload["pools"].items()}
    meta_text = json.dumps({**payload["meta"], "pools": pool_meta}, indent=2, ensure_ascii=False)
    wordings = "\n\n".join(f"[{k}]\n{v}" for k, v in payload["wordings"].items())
    pools_line = ", ".join(f"{p['label']} ({len(p['prompts'])} prompts, {len(p['models'])} models)" for p in payload["pools"].values())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persona tag elicitation</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Persona tag elicitation, one prompt at a time</h1>
<p class="note">Pools: {pools_line}. Every model answers each pool's prompts at temperature 0. Per model: three tag reads
(the <b>prefix tag</b> the model writes before its reply under the tag system prompt; the <b>question</b> asked after its plain reply;
the family <b>checklist</b> asked after its plain reply, showing the families answered yes) and three bodies (<b>plain</b>: no tag, no
system prompt; <b>prefix</b>: the body written after the prefix tag under the system prompt; <b>prefilled</b>: the body written with that
same tag prefilled and no system prompt). Badges are lexicon heuristics, not judgments. Hover a row label for the wording. Verdicts save in
this browser per pool and export as JSON; arrow keys move between prompts.</p>
<details class="meta"><summary>elicitation wordings</summary><div class="body">{wordings.replace('&', '&amp;').replace('<', '&lt;')}</div></details>
<details class="meta"><summary>run metadata</summary><div class="body">{meta_text.replace('&', '&amp;').replace('<', '&lt;')}</div></details>
<nav><select id="pool"></select><button id="prev">&larr; prev</button><span class="pos" id="pos"></span><button id="next">next &rarr;</button>
<select id="jump"></select><label class="sw"><input type="checkbox" id="stack"> stack bodies</label>
<span class="progress" id="progress"></span><button id="export">export verdicts</button></nav>
<div class="msg" id="msg"></div>
<div class="grid" id="grid"></div>
<div class="pnote" id="pnote"></div>
</div>
<script id="payload" type="application/json">{data}</script>
<script>{JS}</script>
</body></html>
"""


def main() -> None:
    payload = build_payload()
    common.VIEWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    common.VIEWER_PATH.write_text(build_html(payload), encoding="utf-8", newline="\n")
    print(f"wrote {common.VIEWER_PATH}: " + "; ".join(
        f"{n} {len(p['prompts'])} prompts x {len(p['models'])} models" for n, p in payload["pools"].items()))


if __name__ == "__main__":
    main()
