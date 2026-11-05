"""Build data/reading_viewer.html: the holistic readings, one page.

A list of the models on the left, each under the name the blind reader gave it with the
recipe beneath (the reader never saw the recipe, or anything but the model's own output;
the page shows both). The main pane shows the reading as rendered Markdown; every
citation in square brackets (``[primary-drives#1]``, ``[feel:asked#12]``) is a button that
opens the cited reply in full, with its prompt, in a panel on the right, so a claim can be
checked against the text it rests on. The synthesis over all the reports, with the letter
to model mapping the synthesis reader never saw, is the first entry.

    uv run python experiments/08-persona-arithmetic/build_reading_viewer.py
"""

import json
import re

import common
import read_checkpoints as rc

VIEWER_PATH = common.DATA / "reading_viewer.html"
CITATION = re.compile(r"\[((?:feel:)?[a-z0-9_-]+#\d+)\]")
STORED_CONTEXT = {shown: stored for stored, shown in rc.CONTEXT_IDS.items()}


def resolve_citation(ref: str, model: str, cfg: dict, cache: dict) -> dict | None:
    """The full reply a citation points at, with what it answered."""
    body = ref
    is_feel = body.startswith("feel:")
    if is_feel:
        body = body[5:]
    key, index = body.rsplit("#", 1)
    index = int(index)
    label = common.merge_label(cfg, model)
    if is_feel:
        key = STORED_CONTEXT.get(key, key)
        doc = cache.setdefault(("feel", model), common.read_json(common.feel_completions_path(model)))
        rows = [r for r in doc["completions"].get(key, []) if r["index"] == index]
        if not rows:
            return None
        ctx = doc["contexts"][key]
        asked = (f'user: "{ctx["user"]}"' if ctx["user"] else "empty user turn") + f' / assistant prefix: "{ctx["prefill"]}"'
        return {"source": model, "label": label, "asked": asked, "text": ctx["prefill"] + rc.joined(rows[0]["text"]),
                "at_cap": rows[0].get("finish") == "length", "kind": "feel"}
    doc = cache.setdefault(("refl", model), common.read_json(common.reflections_path(model)))
    rows = [r for r in doc["reflections"].get(key, []) if r["index"] == index]
    if not rows:
        return None
    return {"source": model, "label": label, "asked": doc["questions"][key]["text"], "text": rows[0]["text"],
            "at_cap": rows[0].get("finish") == "length", "kind": "reflection"}


def viewer_models(cfg: dict) -> list[str]:
    """The models with a page: ``models``, plus the six trained personas, which the sampling
    lists keep out of ``models``, placed right after the wrapper control so the nav's groups
    stay contiguous."""
    names = list(cfg["models"])
    at = names.index("moodless") + 1
    return names[:at] + [p for p in cfg["axis_read"]["personas"] if p not in names] + names[at:]


def build_payload() -> dict:
    cfg = common.load_config()
    cache: dict = {}
    models = []
    names = viewer_models(cfg)
    for m in names:
        path = rc.reading_path(m)
        if not path.exists():
            continue
        rec = common.read_json(path)
        cited = {}
        for ref in sorted(set(CITATION.findall(rec["blind"]["text"]))):
            hit = resolve_citation(ref, m, cfg, cache)
            if hit:
                cited[ref] = hit
        models.append({
            "name": m, "label": rec["label"], "recipe": rec["recipe"], "summary": rec.get("summary") or {},
            "blind": rec["blind"]["text"], "words": len(rec["blind"]["text"].split()),
            "input_tokens": rec["input_tokens_counted"], "output_tokens": rec["blind"]["usage"]["output_tokens"],
            "cited": cited, "reader": f"{rec['reading']['model']}, effort {rec['reading']['effort']}",
        })
    synthesis = None
    if rc.SYNTHESIS_PATH.exists():
        s = common.read_json(rc.SYNTHESIS_PATH)
        synthesis = {
            "letters": {k: {"model": v, "label": common.merge_label(cfg, v)} for k, v in s["letters"].items()},
            "blind": s["blind"]["text"], "words": len(s["blind"]["text"].split()),
            "input_tokens": s["input_tokens_counted"], "output_tokens": s["blind"]["usage"]["output_tokens"],
            "reader": f"{s['reading']['model']}, effort {s['reading']['effort']}",
        }
    def informed_entry(path, title, recipe_examples="W − dW(irritated), W + dW(anxious) + dW(upbeat), ..."):
        if not path.exists():
            return None
        s = common.read_json(path)
        return {
            "title": title, "headings": s["headings"], "recipe_examples": recipe_examples,
            "text": s["informed"]["text"], "words": len(s["informed"]["text"].split()),
            "input_tokens": s["input_tokens_counted"], "output_tokens": s["informed"]["usage"]["output_tokens"],
            "reader": f"{s['reading']['model']}, effort {s['reading']['effort']}" + (f", via {s['reading']['backend']}" if s['reading'].get('backend', 'anthropic') != 'anthropic' else ""),
        }
    informed = informed_entry(rc.SYNTHESIS_INFORMED_PATH, "Synthesis with the recipes")
    informed_full = informed_entry(rc.SYNTHESIS_INFORMED_FULL_PATH, "Synthesis with the recipes, base and moodless read")
    informed_around_control = informed_entry(
        rc.SYNTHESIS_INFORMED_AROUND_CONTROL_PATH,
        "Synthesis with the recipes: personas, negations around the control, sums minus the control",
        recipe_examples="W + dW(irritated), W + 2·dW(moodless) − dW(irritated), W + dW(irritated) + dW(grateful) − dW(moodless), ...",
    )
    return {"models": models, "synthesis": synthesis, "informed": informed, "informed_full": informed_full, "informed_around_control": informed_around_control, "n_models": len(names), "merges": cfg["merges"]}


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Checkpoint readings</title>
<style>
:root { --ink: #1d1d1f; --muted: #6b6b70; --line: #e2e2e6; --bg: #fafafa; --panel: #fff; --accent: #2d6fa8; --soft: #eef3f8; --cite: #e8f0e8; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: var(--ink); background: var(--bg); }
#layout { display: grid; grid-template-columns: 290px 1fr 0px; height: 100vh; }
#layout.open { grid-template-columns: 290px 1fr 480px; }
#nav { border-right: 1px solid var(--line); overflow-y: auto; background: var(--panel); padding: 12px 0; }
#nav h1 { font-size: 15px; margin: 4px 16px 10px; }
#nav .hint { color: var(--muted); font-size: 12px; margin: 0 16px 10px; }
#nav button { display: block; width: 100%; text-align: left; border: 0; background: none; padding: 7px 16px; cursor: pointer; font: inherit; border-left: 3px solid transparent; }
#nav button:hover { background: var(--soft); }
#nav button.on { background: var(--soft); border-left-color: var(--accent); }
#nav .nm { font-weight: 600; }
#nav .lb { color: var(--muted); font-size: 12px; font-family: ui-monospace, Menlo, Consolas, monospace; }
#nav .group { margin: 10px 16px 2px; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
#main { overflow-y: auto; padding: 24px 36px 60px; }
#side { border-left: 1px solid var(--line); overflow-y: auto; background: var(--panel); padding: 16px 20px; display: none; }
#layout.open #side { display: block; }
.head h2 { margin: 0 0 2px; font-size: 22px; }
.head .lb { font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--muted); }
.head .one { margin: 8px 0 4px; font-size: 16px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }
.chip { font-size: 12px; padding: 2px 9px; border-radius: 10px; background: var(--soft); color: #2b3d55; }
.chip.k { background: #efe7d6; color: #5a4a1a; }
.meta { color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--line); margin-bottom: 18px; }
.tabs button { border: 0; background: none; padding: 8px 14px; cursor: pointer; font: inherit; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -1px; }
.tabs button.on { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
.tabs button:disabled { opacity: .45; cursor: default; }
.md { max-width: 860px; }
.md h2 { font-size: 17px; margin: 26px 0 8px; padding-top: 10px; border-top: 1px solid var(--line); }
.md h1 { font-size: 19px; margin: 20px 0 8px; }
.md h3 { font-size: 15px; margin: 18px 0 6px; }
.md p { margin: 0 0 10px; }
.md ul, .md ol { margin: 0 0 10px; padding-left: 22px; }
.md li { margin: 2px 0; }
.md blockquote { margin: 0 0 10px; padding: 2px 12px; border-left: 3px solid var(--line); color: #444; }
.md code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px; background: #f0f0f2; padding: 0 3px; border-radius: 3px; }
.md pre { background: #f0f0f2; padding: 10px 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
.md table { border-collapse: collapse; margin: 6px 0 14px; font-size: 13.5px; }
.md th, .md td { border: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }
.md th { background: var(--soft); }
.cite { display: inline-block; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; background: var(--cite); color: #1f4d2a; border: 0; border-radius: 4px; padding: 0 5px; cursor: pointer; margin: 0 1px; }
.cite:hover { background: #d6e6d6; }
.cite.dead { background: #f3e5e5; color: #7a2d2d; cursor: default; }
#side h3 { margin: 0 0 4px; font-size: 14px; }
#side .asked { color: #333; background: var(--soft); padding: 8px 10px; border-radius: 6px; margin: 8px 0; font-size: 13.5px; }
#side .src { color: var(--muted); font-size: 12px; font-family: ui-monospace, Menlo, Consolas, monospace; }
#side pre.reply { white-space: pre-wrap; font: 13.5px/1.5 inherit; font-family: inherit; margin: 10px 0; }
#side .close { float: right; border: 0; background: none; cursor: pointer; font-size: 18px; color: var(--muted); }
#side .cap { color: #a33; font-size: 12px; }
.letters { font-size: 13px; margin: 8px 0 14px; }
.letters code { background: #f0f0f2; padding: 0 3px; border-radius: 3px; }
@media (max-width: 900px) { #layout, #layout.open { grid-template-columns: 1fr; height: auto; } #nav { border-right: 0; border-bottom: 1px solid var(--line); } #main { padding: 16px; } #side { border-left: 0; border-top: 1px solid var(--line); } }
</style>
</head>
<body>
<div id="layout">
  <nav id="nav"></nav>
  <main id="main"></main>
  <aside id="side"></aside>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const P = JSON.parse(document.getElementById('payload').textContent);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const CITE = /\[((?:[AB]:)?(?:feel:)?[a-z0-9_-]+#\d+)\]/g;

function inline(s, cited) {
  let out = '', last = 0, parts = [];
  // protect code spans first
  s = esc(s);
  s = s.replace(/`([^`]+)`/g, (m, c) => `<code>${c}</code>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>');
  s = s.replace(CITE, (m, ref) => {
    if (!cited) return `<code>${esc(ref)}</code>`;  // the synthesis: a citation belongs to some report's model, open that reading to follow it
    const ok = cited[ref];
    return `<button class="cite${ok ? '' : ' dead'}" data-ref="${esc(ref)}"${ok ? '' : ' title="not found in the data"'}>${esc(ref)}</button>`;
  });
  return s;
}

function render(md, cited) {
  const lines = md.replace(/\r/g, '').split('\n');
  const html = [];
  let i = 0;
  const flushPara = buf => { if (buf.length) html.push(`<p>${inline(buf.join(' '), cited)}</p>`); buf.length = 0; };
  let buf = [];
  while (i < lines.length) {
    const ln = lines[i];
    if (/^```/.test(ln)) { flushPara(buf); const code = []; i++; while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]); i++; html.push(`<pre>${esc(code.join('\n'))}</pre>`); continue; }
    const h = ln.match(/^(#{1,4})\s+(.*)$/);
    if (h) { flushPara(buf); html.push(`<h${h[1].length}>${inline(h[2], cited)}</h${h[1].length}>`); i++; continue; }
    if (/^\s*\|/.test(ln) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flushPara(buf);
      const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => inline(c.trim(), cited));
      const head = cells(ln); i += 2; const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
      html.push(`<table><thead><tr>${head.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
      continue;
    }
    const li = ln.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      flushPara(buf);
      const ordered = /\d/.test(li[2]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
        if (m && m[1].length === li[1].length) { items.push([m[3]]); i++; }
        else if (lines[i].trim() === '') { i++; if (!(i < lines.length && /^(\s*)([-*+]|\d+[.)])\s+/.test(lines[i]) && lines[i].match(/^(\s*)/)[1].length === li[1].length)) break; }
        else if (/^\s+/.test(lines[i]) && items.length) { items[items.length - 1].push(lines[i].trim()); i++; }
        else break;
      }
      html.push(`<${ordered ? 'ol' : 'ul'}>${items.map(it => `<li>${inline(it.join(' '), cited)}</li>`).join('')}</${ordered ? 'ol' : 'ul'}>`);
      continue;
    }
    if (/^>\s?/.test(ln)) { flushPara(buf); const q = []; while (i < lines.length && /^>\s?/.test(lines[i])) q.push(lines[i++].replace(/^>\s?/, '')); html.push(`<blockquote>${inline(q.join(' '), cited)}</blockquote>`); continue; }
    if (ln.trim() === '') { flushPara(buf); i++; continue; }
    buf.push(ln.trim()); i++;
  }
  flushPara(buf);
  return html.join('\n');
}

const KIND = { 'one character': 'one character', composite: 'composite', absence: 'absence', opposite: 'opposite', degraded: 'degraded' };
let current = P.synthesis ? 'synthesis' : (P.models[0] && P.models[0].name);

function nav() {
  const el = document.getElementById('nav');
  const parts = [`<h1>Checkpoint readings</h1><p class="hint">${P.models.length} of ${P.n_models} models read. The name is the reader's; the recipe beneath it was never shown to the reader, which saw only the model's own replies.</p>`];
  if (P.synthesis) parts.push(`<button class="${current === 'synthesis' ? 'on' : ''}" data-m="synthesis"><div class="nm">Synthesis over all reports</div><div class="lb">${Object.keys(P.synthesis.letters).length} models as letters</div></button>`);
  for (const key of ['informed', 'informed_full', 'informed_around_control']) {
    if (P[key]) parts.push(`<button class="${current === key ? 'on' : ''}" data-m="${key}"><div class="nm">${esc(P[key].title)}</div><div class="lb">${Object.keys(P[key].headings).length} models under their names</div></button>`);
  }
  let group = '';
  for (const m of P.models) {
    const g = (m.name === 'base' || (Object.keys(m.recipe).length === 1 && Object.values(m.recipe)[0] > 0)) ? 'reference' : (m.name.includes('-plus-') ? 'pair sums' : 'negations');
    if (g !== group) { parts.push(`<div class="group">${g}</div>`); group = g; }
    const nm = m.summary.name || '(no name parsed)';
    parts.push(`<button class="${current === m.name ? 'on' : ''}" data-m="${esc(m.name)}"><div class="nm">${esc(nm)}</div><div class="lb">${esc(m.label)}</div></button>`);
  }
  el.innerHTML = parts.join('');
  el.querySelectorAll('button').forEach(b => b.onclick = () => { current = b.dataset.m; closeSide(); nav(); main(); });
}

function main() {
  const el = document.getElementById('main');
  if (current === 'synthesis') {
    const s = P.synthesis;
    const letters = Object.entries(s.letters).map(([k, v]) => `<code>${k}</code> ${esc(v.label)}`).join(' · ');
    el.innerHTML = `<div class="head"><h2>Synthesis over all the reports</h2><div class="meta">${esc(s.reader)}; ${s.input_tokens.toLocaleString()} input tokens, ${s.output_tokens.toLocaleString()} output, ${s.words.toLocaleString()} words. The reader saw the reports under shuffled letters and nothing about the models; the mapping below is for the reader of this page.</div><div class="letters">${letters}</div></div><div class="md">${render(s.blind, null)}</div>`;
  } else if (current === 'informed' || current === 'informed_full' || current === 'informed_around_control') {
    const s = P[current];
    el.innerHTML = `<div class="head"><h2>${esc(s.title)}</h2><div class="meta">${esc(s.reader)}; ${s.input_tokens.toLocaleString()} input tokens, ${s.output_tokens.toLocaleString()} output, ${s.words.toLocaleString()} words. The reader saw the same reports headed by each model's name and recipe (${esc(s.recipe_examples)}), with the notation explained.</div></div><div class="md">${render(s.text, null)}</div>`;
  } else {
    const m = P.models.find(x => x.name === current);
    const su = m.summary;
    const chips = [su.stability && `<span class="chip">${esc(su.stability)}</span>`, su.opposite && `<span class="chip">opposite: ${esc(su.opposite)}</span>`].filter(Boolean).join('');
    el.innerHTML = `<div class="head"><h2>${esc(su.name || m.name)}</h2><div class="lb">${esc(m.name)} · ${esc(m.label)}</div>${su.one_line ? `<p class="one">${esc(su.one_line)}</p>` : ''}<div class="chips">${chips}</div><div class="meta">${esc(m.reader)}; ${m.input_tokens.toLocaleString()} input tokens, ${m.output_tokens.toLocaleString()} output, ${m.words.toLocaleString()} words. Click a citation to open the reply.</div></div><div class="md">${render(m.blind, m.cited)}</div>`;
    el.querySelectorAll('button.cite:not(.dead)').forEach(b => b.onclick = () => openSide(m, b.dataset.ref));
  }
  el.scrollTop = 0;
}

function openSide(m, ref) {
  const c = m.cited[ref];
  const side = document.getElementById('side');
  side.innerHTML = `<button class="close" title="close">×</button><h3>${esc(ref)}</h3><div class="src">${esc(c.source)} · ${esc(c.label)}${c.at_cap ? ' · <span class="cap">ran to the length cap</span>' : ''}</div><div class="asked">${esc(c.asked)}</div><pre class="reply">${esc(c.text)}</pre>`;
  side.querySelector('.close').onclick = closeSide;
  document.getElementById('layout').classList.add('open');
  side.scrollTop = 0;
}
function closeSide() { document.getElementById('layout').classList.remove('open'); }

nav(); main();
</script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    VIEWER_PATH.write_text(html, encoding="utf-8", newline="\n")
    n_cited = sum(len(m["cited"]) for m in payload["models"])
    print(f"wrote {VIEWER_PATH} ({len(html) / 1e6:.1f} MB): {len(payload['models'])} readings, "
          f"{n_cited} cited replies embedded, synthesis {'yes' if payload['synthesis'] else 'no'}")


if __name__ == "__main__":
    main()
