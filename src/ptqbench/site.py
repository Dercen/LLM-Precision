"""`ptq site`: the results and the guides as a static website, so readers install nothing.

Pages: the by-model index as the home page, one page per model, the results guide and
HOW-IT-WORKS, plus the charts. The Markdown is the same text `ptq aggregate` writes;
links between those files become links between pages, and links to anything else in
the repository point at GitHub. CI (.github/workflows/site.yml) builds this into GitHub
Pages after every push that changes results; locally `ptq site` writes `site/` and
`ptq ui` serves the same pages under /results/.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

from . import paths, provenance

REPO_URL = "https://github.com/Dercen/LLM-Precision"
SITE_TITLE = "ptq-bench: how much do language models lose at fewer bits?"

# repo-relative source -> site-relative output
PAGE_MAP = {
    "results/by-model/README.md": "index.html",
    "results/README.md": "results-guide.html",
    "HOW-IT-WORKS.md": "how-it-works.html",
}

_CSS = """
:root { --ink: #1b1b1b; --muted: #5d5c58; --line: #e1e0d9; --bg: #fcfcfb; --accent: #2a78d6; }
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--ink); background: var(--bg); }
nav { border-bottom: 1px solid var(--line); padding: 10px 16px; display: flex; gap: 18px; flex-wrap: wrap; font-size: 15px; }
nav a { color: var(--accent); text-decoration: none; } nav a:hover { text-decoration: underline; }
main { max-width: 960px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.7em; margin: .4em 0; } h2 { font-size: 1.3em; margin-top: 1.6em; border-bottom: 1px solid var(--line); padding-bottom: .2em; }
table { border-collapse: collapse; margin: 1em 0; display: block; overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid var(--line); padding: 5px 9px; text-align: left; white-space: nowrap; }
th { background: #f3f3ef; } tr:nth-child(even) td { background: #f8f8f6; }
img { max-width: 100%; height: auto; border: 1px solid var(--line); margin: .5em 0; }
code { background: #f3f3ef; padding: 1px 4px; border-radius: 3px; font-size: .92em; }
pre { background: #f3f3ef; padding: 10px; overflow-x: auto; }
pre code { background: none; padding: 0; }
footer { color: var(--muted); font-size: 13px; margin: 3em 0 1em; border-top: 1px solid var(--line); padding-top: .6em; }
.mermaid { background: none; }
"""

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
{mermaid}
</head>
<body>
<nav><a href="{root}index.html">Results by model</a><a href="{root}results-guide.html">How to read a result</a><a href="{root}how-it-works.html">How it works</a><a href="{repo}">Run it yourself (GitHub)</a></nav>
<main>
{body}
<footer>{footer}</footer>
</main>
</body>
</html>
"""

_MERMAID = ('<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";'
            'mermaid.initialize({startOnLoad: true, theme: "neutral"});</script>')


def _render_markdown(text: str) -> str:
    import markdown

    # ```mermaid fences become <pre class="mermaid"> blocks the mermaid script draws.
    text = re.sub(r"```mermaid\n(.*?)```", lambda m: f'<pre class="mermaid">\n{html.escape(m.group(1))}</pre>', text, flags=re.DOTALL)
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def _rewrite_links(text: str, source: str, output: str) -> str:
    """Links between site pages stay relative; links to other repo files go to GitHub."""
    src_dir = Path(source).parent
    out_dir = Path(output).parent

    def repl(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#)", target):
            return m.group(0)
        target_path, _, anchor = target.partition("#")
        repo_rel = _normpath((src_dir / target_path).as_posix())
        anchor = f"#{anchor}" if anchor else ""
        if repo_rel in PAGE_MAP:
            return f"[{label}]({_relative(PAGE_MAP[repo_rel], out_dir)}{anchor})"
        by_model = re.fullmatch(r"results/by-model/([^/]+)\.md", repo_rel)
        if by_model:
            return f"[{label}]({_relative(f'by-model/{by_model.group(1)}.html', out_dir)}{anchor})"
        if repo_rel.startswith("results/plots/"):
            return f"[{label}]({_relative(repo_rel.removeprefix('results/'), out_dir)})"
        return f"[{label}]({REPO_URL}/blob/main/{repo_rel}{anchor})"

    text = re.sub(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)", repl, text)
    # Images: ../plots/x.png from by-model pages, plots/x.png from the root.
    return re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)",
                  lambda m: f"![{m.group(1)}]({_relative('plots/' + Path(m.group(2)).name, out_dir)})" if "plots/" in m.group(2) else m.group(0),
                  text)


def _normpath(p: str) -> str:
    parts: list[str] = []
    for part in p.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _relative(site_path: str, from_dir: Path) -> str:
    depth = len([p for p in from_dir.parts if p not in ("", ".")])
    return "../" * depth + site_path


def _page(body_md: str, *, source: str, output: str, title: str, root: Path) -> str:
    rewritten = _rewrite_links(body_md, source, output)
    body = _render_markdown(rewritten)
    depth = len([p for p in Path(output).parent.parts if p not in ("", ".")])
    footer = (f"Generated by <code>ptq site</code> from commit <code>{provenance.git_sha()[:12]}</code> on "
              f"{provenance.utc_now()[:10]}. Every number traces back to a run file in "
              f'<a href="{REPO_URL}/tree/main/results/raw/runs">results/raw/runs</a>.')
    return _TEMPLATE.format(
        title=html.escape(title), css=_CSS, mermaid=_MERMAID if 'class="mermaid"' in body else "",
        root="../" * depth, repo=REPO_URL, body=body, footer=footer,
    )


def _title_of(md: str, fallback: str) -> str:
    m = re.search(r"(?m)^# (.+)$", md)
    return f"{m.group(1).strip()} · ptq-bench" if m else fallback


def _index_markdown(by_model_index: str) -> str:
    """The home page; its links are written relative to results/by-model/, where its source lives."""
    body = by_model_index.split("\n", 1)[1] if by_model_index.startswith("# ") else by_model_index
    body = body.replace("Generated by `ptq aggregate` — do not edit by hand. ", "")
    return (
        "# How much do language models lose at fewer bits?\n\n"
        "A language model is mostly numbers (weights). Storing each one in 4 bits instead of 16 makes "
        "the model a quarter the size, but rounding damages it. These pages measure that damage, model "
        "by model, for four rounding methods, and check the numbers against the published papers.\n\n"
        "- **New here?** [How to read a result](../README.md) explains every word in the tables in two minutes.\n"
        "- **Want the story?** [How it works](../../HOW-IT-WORKS.md), step by step.\n"
        f"- **Want to run it?** [The repository]({REPO_URL}) has a one-command setup and a menu.\n\n"
        "## Results by model\n"
        + body
    )


def build(out_dir: str | Path | None = None, *, root: Path | None = None) -> list[Path]:
    """Write the site; returns the pages written (not the copied charts)."""
    root = root or paths.repo_root()
    out = Path(out_dir) if out_dir else root / "site"
    if out.exists():
        shutil.rmtree(out)
    (out / "by-model").mkdir(parents=True)
    written: list[Path] = []

    index_md = (root / "results/by-model/README.md").read_text(encoding="utf-8")
    pages = {"index.html": (_index_markdown(index_md), "results/by-model/README.md", SITE_TITLE)}
    for src, dst in PAGE_MAP.items():
        if dst == "index.html":
            continue
        md = (root / src).read_text(encoding="utf-8")
        pages[dst] = (md, src, _title_of(md, SITE_TITLE))
    for md_path in sorted((root / "results/by-model").glob("*.md")):
        if md_path.name == "README.md":
            continue
        md = md_path.read_text(encoding="utf-8")
        pages[f"by-model/{md_path.stem}.html"] = (md, f"results/by-model/{md_path.name}", _title_of(md, md_path.stem))

    for output, (md, source, title) in pages.items():
        target = out / output
        target.write_text(_page(md, source=source, output=output, title=title, root=root), encoding="utf-8")
        written.append(target)

    plots = root / "results/plots"
    if plots.is_dir():
        (out / "plots").mkdir(exist_ok=True)
        for png in plots.glob("*.png"):
            shutil.copy2(png, out / "plots" / png.name)
    (out / ".nojekyll").write_text("")
    return written
