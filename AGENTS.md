# AGENTS.md

Guidance for AI coding agents working in this repository.
Follows the [AGENTS.md](https://agents.md/) open format — a "README for agents."

## Project overview

The **Computer Science Notebook** is a community-driven knowledge base of ~740
Markdown notes bridging computer science theory with real-world industry
applications. It is **a content repository, not a software application** — there
is no application source code, unit test suite, or package to build.

The only build step is the documentation site (MkDocs Material), which publishes
to GitHub Pages.

## Repository layout

Three top-level content trees:

```
core/       — Foundational CS (ai-ml, programming, systems, engineering, fundamentals)
industry/   — Applied domains (transportation, aerospace, healthcare, manufacturing, …)
meta/       — Tooling, templates, career resources, project docs
```

Supporting files at the root:

```
mkdocs.yml            — docs site config (docs_dir is the repo root, via the same-dir plugin)
.nav.yml              — top-level site navigation
requirements-docs.txt — pinned docs toolchain
overrides/            — MkDocs Material theme overrides (social card meta)
CONTRIBUTING.md       — content templates and contribution process
```

### `core/` topic layout

```
core/<domain>/<topic>/
├── concepts/               # Theory; tiered notes (basics / intermediate / advanced)
├── lab/                    # Code examples, notebooks, experiments
├── resources/              # Papers, PDFs, external references
└── industry-applications/  # Links to /industry counterparts
```

## Setup commands

```bash
# Install the docs toolchain (Python 3.11+)
pip install -r requirements-docs.txt

# Serve locally with live reload → http://127.0.0.1:8000
mkdocs serve

# Build the static site into ./site
mkdocs build
```

> Note: a full build takes ~2 minutes because notebooks are rendered and git
> revision dates are computed per file.

## Verifying your changes

There is no test suite. **The build is the test.** Before committing:

```bash
mkdocs build
```

Treat these as failures to fix:

- `ERROR` of any kind — the build must exit 0.
- `... is not found among documentation files` — a broken internal link or image.
  The repository is currently at **zero** such warnings; do not regress it.

If you only touched Markdown, you can skip notebook rendering for a much faster
check by temporarily building with the `mkdocs-jupyter` plugin removed.

## Content conventions

### File naming

- Tiered notes: `<topic>-core-basics.md`, `<topic>-core-intermediate.md`,
  `<topic>-core-advanced.md`
- Industry notes: `<topic>-industry-basics.md`, etc.
- Lab/example notes: descriptive names under `lab/` or `examples/`
- Use **lowercase-kebab-case**. Avoid spaces in file and directory names — they
  produce fragile URLs (`7-Big%20Data`).

### Links and images

- Use **standard relative Markdown links**: `[CNN notes](../deep-learning-notes/dl-notes.md)`.
- Do **not** use `[[wikilinks]]` — MkDocs does not resolve them.
- Do **not** use Windows-style backslash paths (`..\foo\bar.md`).
- Store images near their note (e.g. `./resources/images/foo.png`) and reference
  them relatively.

### Writing style

- Notes target three audience tiers — beginner, intermediate, advanced — each in
  a **separate file**, not mixed in one.
- Prefer **Mermaid diagrams** for architecture visuals (```` ```mermaid ````).
  Roughly half the notes use them; keep that density.
- Include practical examples and cite references.
- Cross-reference related notes, and link `core/` topics to their `industry/`
  counterparts (and back) — this bidirectional bridge is the project's core value.

### Templates

Use the templates in `CONTRIBUTING.md` (long-form technical note and short-form
note). Folder scaffolds live in `meta/templates/core/` and `meta/templates/industry/`.

A helper script scaffolds a new topic folder:

```bash
python meta/tools/hello_world.py <topic-name>
```

## Notebooks

- `.ipynb` files are rendered into the site by `mkdocs-jupyter` (`execute: false`
  — existing outputs are rendered, nothing is executed at build time).
- Each notebook starts with an **"Open in Colab"** badge cell. Preserve it when
  editing, and add one to any new notebook.
- Do not commit `.cache/` (mkdocs-jupyter build cache) — it is gitignored.

## Commit conventions

Commits follow `<scope>: <description>`, with a lowercase imperative description:

```
core-ai-ml-ca: add audio recognition tasks
core-edge-ai: add deepcraft studio quick start
docs: fix broken internal links across notes
```

Common scopes: `core-ai-ml`, `core-ai-ml-ca` (computer audition), `core-edge-ai`,
`core-programming`, `docs`, `chore`.

Keep commit messages short and factual. Do not add marketing language or
attribution trailers for AI tooling.

## Pull requests

- Branch from `master`; never commit directly to it.
- Fill in `.github/PULL_REQUEST_TEMPLATE.md`.
- Confirm `mkdocs build` is clean and state that in the PR description.
- Keep PRs scoped to one concern (content vs tooling vs site config).
- Merging to `master` triggers `.github/workflows/docs.yml`, which deploys the
  site to GitHub Pages.

## Boundaries

- **Do not fabricate technical content.** If a note links to a file that does not
  exist, either write real content or remove the link — do not invent a
  placeholder note to satisfy a link.
- **Verify technical claims and code** before committing. This is a learning
  resource; incorrect content is worse than missing content.
- Do not commit editor state (`.obsidian/`), build output (`site/`), or large
  binaries without discussion.
- Content is licensed **CC BY-NC-SA 4.0**. Do not paste in third-party material
  that is incompatible with it; cite sources instead.

## Further reading

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — full content templates and process
- [`meta/tools/authoring-guide.md`](meta/tools/authoring-guide.md) — Obsidian +
  Claude authoring workflow
