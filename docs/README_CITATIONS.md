# Citation Verification Pipeline

`check_citations.py` automatically verifies the bibliography of a scientific
paper against canonical registries (arXiv API, Crossref, OpenAlex), catching
hallucinated, misattributed, and stale references before submission.

## What it checks

1. **Identifier resolution** — every arXiv ID (`eprint`, `arXiv preprint arXiv:...`
   journal strings, URLs) and DOI is looked up in its registry.
2. **Fallback title search** — entries without identifiers are searched by
   title across arXiv → Crossref → OpenAlex.
3. **Metadata agreement** — the claimed title (normalized similarity), year
   (±1), and author list (first-author + overlap) are compared to the record.
4. **Plain-web URLs** — `@misc` entries citing news/press URLs are pinged to
   confirm they resolve (`--skip-web` to disable).
5. **TeX cross-check** — `\cite`/`\citep`/`\citet` keys used in `.tex` files
   are compared against `.bib` keys (missing vs. unused).

## Statuses

| Status | Meaning |
|---|---|
| ✅ VERIFIED | title, year, and authors match the canonical record |
| ⚠️ WARNING | minor mismatch (partial title, year drift, author-name variants) |
| ❌ ERROR | identifier doesn't resolve, wrong-work match, or suspected misattribution |
| 🚩 NOT_FOUND | no record found anywhere — possible hallucinated reference |
| ❓ UNVERIFIED | typically books or web-only sources without registries |

## Usage

```bash
# Basic: check <paper_dir>/references.bib
python check_citations.py /path/to/paper

# Custom bib/output, polite email for Crossref/OpenAlex pools
python check_citations.py /path/to/paper --bib refs.bib \
    --output-dir tables/citation_check --email you@example.com
```

arXiv API calls are rate-limited to one per `--delay` (default 3.0s) per arXiv
policy; a full ~55-entry bibliography takes ~2–3 minutes.

## Outputs

- `tables/citation_check/citation_report.csv` — one row per entry
- `tables/citation_check/citation_report.md` — summary + review table

No third-party dependencies (stdlib only).
