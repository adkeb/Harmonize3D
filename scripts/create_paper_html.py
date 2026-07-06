from __future__ import annotations

import html
import re
import sys
from pathlib import Path


STYLE = """@page { size: A4; margin: 18mm 16mm; }
:root { --ink:#17202a; --muted:#56616f; --line:#d6dce3; --accent:#1976d2; }
body { font-family: "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; color: var(--ink); background: #fff; line-height: 1.72; font-size: 14.2px; }
main { max-width: 980px; margin: 0 auto; }
h1 { text-align: center; font-size: 28px; line-height: 1.35; margin: 24px 0 18px; }
h2 { font-size: 22px; border-bottom: 1px solid var(--line); padding-bottom: 6px; margin-top: 34px; break-after: avoid; }
h3 { font-size: 17px; margin-top: 24px; break-after: avoid; }
p { margin: 9px 0; text-align: justify; }
a { color: var(--accent); text-decoration: none; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f3f5f7; padding: 1px 4px; border-radius: 3px; overflow-wrap: anywhere; }
pre { background: #f6f8fa; border: 1px solid var(--line); border-radius: 6px; padding: 12px; overflow-x: visible; font-size: 12px; line-height: 1.45; white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; }
pre code { background: transparent; padding: 0; white-space: inherit; word-break: inherit; overflow-wrap: inherit; }
ul, ol { margin: 8px 0 10px 26px; padding: 0; }
li { margin: 4px 0; }
table { width: 100%; border-collapse: collapse; margin: 14px 0 18px; font-size: 12.5px; break-inside: avoid; }
th, td { border: 1px solid var(--line); padding: 7px 8px; vertical-align: top; overflow-wrap: anywhere; }
th { background: #eef5fc; font-weight: 700; }
figure { margin: 18px auto 24px; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid #e0e4e8; border-radius: 6px; }
figcaption, .caption { color: var(--muted); font-size: 12.5px; margin-top: 6px; text-align: center; }
.toc { background:#f7f9fb; border:1px solid var(--line); border-radius:8px; padding: 10px 18px; margin: 20px 0 30px; break-after: page; }
.toc h2 { border:0; margin: 4px 0 8px; }
.toc ol { columns: 2; margin-left: 20px; }
.toc li { break-inside: avoid; }
.toc-l3 { margin-left: 18px; font-size: 12.5px; color: var(--muted); }
@media print { body { font-size: 12.3px; } h2 { margin-top: 24px; } }
"""


def slug(text: str) -> str:
    base = re.sub(r"<[^>]+>", "", text).strip().lower()
    base = re.sub(r"\s+", "-", base)
    base = re.sub(r"[^\w\-\u4e00-\u9fff.：:]+", "", base)
    return base or "section"


def inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(
        r"(https?://[A-Za-z0-9./?=&_%#:\-]+)",
        r'<a href="\1">\1</a>',
        escaped,
    )
    return escaped


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and set(stripped.replace("|", "").replace(" ", "")) <= {"-", ":"}


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    normalized = [row + [""] * (col_count - len(row)) for row in rows]
    out = ["<table>"]
    for idx, row in enumerate(normalized):
        tag = "th" if idx == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{inline(cell)}</{tag}>" for cell in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def convert(markdown_path: Path) -> str:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    headings: list[tuple[int, str, str]] = []
    title = ""

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            text = line[3:].strip()
            headings.append((2, text, slug(text)))
        elif line.startswith("### "):
            text = line[4:].strip()
            headings.append((3, text, slug(text)))

    body: list[str] = []
    i = 0
    in_code = False
    code_lang = ""
    code_lines: list[str] = []
    open_list: str | None = None

    def close_list() -> None:
        nonlocal open_list
        if open_list:
            body.append(f"</{open_list}>")
            open_list = None

    while i < len(lines):
        line = lines[i].rstrip()

        if line.strip().startswith("```"):
            if in_code:
                close_list()
                code = html.escape("\n".join(code_lines))
                cls = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                body.append(f"<pre><code{cls}>{code}</code></pre>")
                code_lines = []
                code_lang = ""
                in_code = False
            else:
                in_code = True
                code_lang = line.strip()[3:].strip()
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not line.strip():
            close_list()
            i += 1
            continue

        if line.startswith("# "):
            close_list()
            text = line[2:].strip()
            body.append(f'<h1 id="{slug(text)}">{inline(text)}</h1>')
            i += 1
            continue

        if line.startswith("## "):
            close_list()
            text = line[3:].strip()
            body.append(f'<h2 id="{slug(text)}">{inline(text)}</h2>')
            i += 1
            continue

        if line.startswith("### "):
            close_list()
            text = line[4:].strip()
            body.append(f'<h3 id="{slug(text)}">{inline(text)}</h3>')
            i += 1
            continue

        image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if image:
            close_list()
            caption, src = image.groups()
            figcaption = "" if re.search(r"\.(png|jpe?g|webp|gif)$", caption, re.IGNORECASE) else f"<figcaption>{inline(caption)}</figcaption>"
            body.append(
                f'<figure><img src="{html.escape(src)}" alt="{html.escape(caption)}">'
                f"{figcaption}</figure>"
            )
            i += 1
            continue

        caption_line = re.match(r"^\*([^*]+)\*$", line.strip())
        if caption_line:
            close_list()
            body.append(f'<p class="caption">{inline(caption_line.group(1).strip())}</p>')
            i += 1
            continue

        if line.strip().startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            close_list()
            rows = [split_table_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_table_row(lines[i]))
                i += 1
            body.append(render_table(rows))
            continue

        bullet = re.match(r"^\s*-\s+(.+)$", line)
        numbered = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if bullet:
            if open_list != "ul":
                close_list()
                body.append("<ul>")
                open_list = "ul"
            body.append(f"<li>{inline(bullet.group(1).strip())}</li>")
            i += 1
            continue
        if numbered:
            if open_list != "ol":
                close_list()
                body.append("<ol>")
                open_list = "ol"
            body.append(f"<li>{inline(numbered.group(1).strip())}</li>")
            i += 1
            continue

        close_list()
        body.append(f"<p>{inline(line.strip())}</p>")
        i += 1

    close_list()

    toc_items = ["<nav class=\"toc\"><h2>目录</h2><ol>"]
    for level, text, anchor in headings:
        cls = "toc-l3" if level == 3 else "toc-l2"
        toc_items.append(f'<li class="{cls}"><a href="#{anchor}">{inline(text)}</a></li>')
    toc_items.append("</ol></nav>")

    if title:
        try:
            first_h1 = next(idx for idx, item in enumerate(body) if item.startswith("<h1 "))
            body.insert(first_h1 + 1, "\n".join(toc_items))
        except StopIteration:
            body.insert(0, "\n".join(toc_items))

    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Harmonize3D Paper Deliverable</title>'
        f"<style>\n{STYLE}\n</style></head><body><main>"
        + "\n".join(body)
        + "</main></body></html>\n"
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: create_paper_html.py input.md output.html", file=sys.stderr)
        return 2
    html_text = convert(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(html_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
