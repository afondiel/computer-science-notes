# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> See [AGENTS.md](AGENTS.md) for general AI-agent guidance (setup, verification,
> conventions, PR process). This file covers the same project with additional
> detail on architecture and templates.

## What This Repository Is

A community-driven computer science knowledge base consisting primarily of Markdown notes, code snippets, and lab examples. It is **not a software application** — there is no application source code or unit test suite. The content is organized as a structured reference notebook, published as an MkDocs Material web site.

## Building the Web Documentation

The notebook is published as an MkDocs Material site. The configuration lives in `mkdocs.yml` at the repository root (`docs_dir` is the repo root itself, via the `same-dir` plugin).

```bash
# Install the docs toolchain (Python 3.11+)
pip install -r requirements-docs.txt

# Build the static site
mkdocs build

# Serve locally with live reload
mkdocs serve
```

The generated site is output to `site/` (gitignored). Merges to `master` deploy the site to GitHub Pages via `.github/workflows/docs.yml`.

**There is no test suite — `mkdocs build` is the verification step.** It must exit 0 and must not report `is not found among documentation files` warnings (the repository is currently at zero).

## Note Generation Tool

A Python script scaffolds new note files and directories:

```bash
# Generate a new topic note (creates <topic-name>-notes/ folder with docs/ and lab/ subdirs)
python meta/tools/hello_world.py <topic-name>
```

This creates the standard note template with sections: Overview, Applications, Tools & Frameworks, Hello World!, References.

## Repository Architecture

Three top-level content trees:

```
core/       — Foundational CS knowledge (programming, systems, AI/ML, engineering, fundamentals)
industry/   — Cross-domain applied sections (automotive, aerospace, healthcare, manufacturing, etc.)
meta/       — Project tooling, templates, documentation, and resources
```

### `core/` structure

Each topic area follows this standard layout:

```
core/<domain>/<topic>/
├── concepts/          # Theory, principles, tiered notes (basics/intermediate/advanced)
├── lab/               # Code examples, notebooks, experiments
├── resources/         # Papers, docs, external references
└── industry-applications/  # Links to /industry counterparts
```

Active topic areas under `core/`:
- `ai-ml/` — The most developed section: computer-vision-notes, deep-learning-notes, ml-notes, nlp-notes, computer-audition, generative-ai-notes, ai-frameworks, ai-agents-notes, cognitive-science, prompt-engineering-notes
- `programming/` — Languages (C, C++, Python, Rust, Go, Java, MATLAB), algorithms, data structures, high-performance programming
- `systems/` — Edge computing, embedded systems, operating systems, cloud, networking
- `engineering/` — SW architecture, testing, DevOps, web
- `fundamentals/` — Signal processing, computer graphics, control systems

### Note file naming convention

- Tiered content: `<topic>-core-basics.md`, `<topic>-core-intermediate.md`, `<topic>-core-advanced.md`
- Industry notes: `<topic>-industry-basics.md`, etc.
- Lab/example notes: descriptive names in `lab/` or `examples/`

### `meta/` structure

- `meta/templates/` — Folder scaffolds for `core/` topics and `industry/` applications
- `meta/tools/` — Automation scripts, Docker, CI/CD notes, VCS cheatsheets
- `meta/tools/authoring-guide.md` — Obsidian + Claude authoring workflow

## Content Templates

### Core topic template (from CONTRIBUTING.md)

```markdown
# {Topic} Technical Notes

## Quick Reference
- One-sentence definition
- Key use cases
- Prerequisites

## Introduction
- What / Why / Where

## Core Concepts
### Fundamental Understanding
### Visual Architecture (Mermaid diagrams)

## Implementation Details
### Basic Implementation
### Intermediate Patterns
### Advanced Topics

## Real-World Applications
## Tools & Resources
## References
## Appendix
```

### Short-form note template (for quick notes)

```markdown
# {Topic} - Notes
## Table of Contents (ToC)
## Overview
## Applications
## Tools & Frameworks
## Hello World!
## References
```

## Git Commit Convention

Commits follow the pattern `<scope>: <description>`:

```
core-ai-ml-ca: add audio recognition tasks
core-ai-ml: Cleanup
core-edge-ai: add deepcraft studio quick start
```

Scope prefixes observed: `core-ai-ml`, `core-ai-ml-ca` (computer audition), `core-edge-ai`, `core-programming`.

## Key Conventions

- Notes are written for three audience tiers: beginner, intermediate, advanced — each in separate files
- Mermaid diagrams are used for architecture visualizations
- Papers and PDFs are stored alongside their corresponding notes in `resources/` or `docs/` subfolders
- Lab code is kept in `lab/` subdirectories separate from conceptual notes
- Industry cross-references link `core/` topics to `industry/` applications bidirectionally
