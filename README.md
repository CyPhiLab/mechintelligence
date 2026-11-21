# Mechanical Intelligence

A live textbook on how mechanisms carry and process information about the world. It is
published at <https://zpatty.github.io/mechanical-intelligence/> and grows as results are
checked. The [research roadmap](ROADMAP.md) lists the open threads and records findings and
decisions.

## Layout

| Path | What it holds |
|---|---|
| `chapters/` | Chapters, in Quarto Markdown |
| `notes/` | Dated research notes |
| `appendices/` | The literature-verification table |
| `code/mibook/` | The computations behind every number and figure |
| `tests/` | Tests that pin every number in the text |
| `ROADMAP.md`, `CHANGELOG.md` | The working plan and the release history |

## Building

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run quarto render
```

The site and the PDF land in `_book/`. Executed figures are cached in `_freeze/` and
re-run when their code changes. Pushes to `main` run the tests, render the book and
deploy it to GitHub Pages (`.github/workflows/publish.yml`).

The computations use the analysis library from
[zpatty/quadruped-mech-info](https://github.com/zpatty/quadruped-mech-info), pinned to a
commit in `pyproject.toml`.
