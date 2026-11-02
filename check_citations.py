#!/usr/bin/env python3
"""Automatic citation checking for scientific papers.

Parses a references.bib file, extracts identifiers (arXiv ID, DOI, URL), and
verifies every entry against the arXiv API, Crossref, and OpenAlex. Each entry
is classified as VERIFIED, WARNING, ERROR, NOT_FOUND, or UNVERIFIED based on
title similarity, year agreement, and author overlap with the canonical record.

Also cross-checks \\cite usage in the paper's .tex files against the .bib keys
to surface missing (cited but undefined) and unused (defined but never cited)
references. Follows the same CLI conventions as the other toolkit entrypoints.

Usage:
    python check_citations.py <paper_dir> [--bib <path>] [--output-dir <path>]
                              [--delay 3.0] [--email <addr>] [--skip-web]

Example:
    python check_citations.py /path/to/paper --output-dir tables/citation_check

Outputs:
    <output-dir>/citation_report.csv  full per-entry results
    <output-dir>/citation_report.md   human-readable summary
"""

import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

ARXIV_API = "http://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org"
OPENALEX_API = "https://api.openalex.org/works"
USER_AGENT = "judgment-to-noise-citation-checker/1.0"

# Statuses
VERIFIED = "VERIFIED"
WARNING = "WARNING"
ERROR = "ERROR"
NOT_FOUND = "NOT_FOUND"
UNVERIFIED = "UNVERIFIED"

STATUS_EMOJI = {
    VERIFIED: "✅",
    WARNING: "⚠️",
    ERROR: "❌",
    NOT_FOUND: "🚩",
    UNVERIFIED: "❓",
}


@dataclass
class BibEntry:
    key: str
    entry_type: str
    fields: dict = field(default_factory=dict)

    def get(self, name, default=""):
        return self.fields.get(name.lower(), default)


@dataclass
class Candidate:
    source: str          # "arxiv" | "crossref" | "openalex"
    found_id: str        # arXiv id or DOI
    title: str
    year: str
    authors: list        # list of "Surname" strings (normalized)


# ---------------------------------------------------------------------------
# BibTeX parsing (stdlib, regex + balanced-brace scan)
# ---------------------------------------------------------------------------

def strip_latex(text):
    """Remove LaTeX markup from a field value for comparison purposes."""
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = text.replace("\\&", "&").replace("\\%", "%")
    text = text.replace("\\ldots", " ").replace("\\dots", " ")
    text = text.replace("\\\"", '"')
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("~", " ").replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(text):
    text = strip_latex(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def extract_surname(author):
    """Best-effort surname extraction from a BibTeX author token."""
    author = strip_latex(author).strip()
    if not author:
        return ""
    if "," in author:
        surname = author.split(",")[0]
    else:
        parts = author.split()
        surname = parts[-1] if parts else ""
    return re.sub(r"[^a-zA-Z]", "", surname).lower()


def parse_authors(field_value):
    """Return (surnames, has_others) from a BibTeX author field."""
    authors = re.split(r"\s+and\s+", strip_latex(field_value), flags=re.IGNORECASE)
    surnames, has_others = [], False
    for a in authors:
        a = a.strip()
        if not a:
            continue
        if a.lower() == "others":
            has_others = True
            continue
        surname = extract_surname(a)
        if surname:
            surnames.append(surname)
    return surnames, has_others


def parse_bib_file(path):
    """Parse a .bib file into BibEntry objects (balanced-brace aware)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        entry_type = m.group(1).lower()
        if entry_type in ("comment", "preamble", "string"):
            continue
        key = m.group(2)
        # Scan the entry body with balanced braces.
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[m.end():i - 1]

        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            name = fm.group(1).lower()
            j = fm.end()
            while j < len(body) and body[j] in " \t\r\n":
                j += 1
            if j >= len(body):
                continue
            if body[j] == "{":
                depth, k = 1, j + 1
                while k < len(body) and depth > 0:
                    if body[k] == "{":
                        depth += 1
                    elif body[k] == "}":
                        depth -= 1
                    k += 1
                value = body[j + 1:k - 1]
            elif body[j] == '"':
                k = body.find('"', j + 1)
                value = body[j + 1:k if k != -1 else len(body)]
            else:
                k = body.find(",", j)
                if k == -1:
                    k = len(body)
                value = body[j:k].strip()
            fields[name] = value.strip().strip(",").strip()
        entries.append(BibEntry(key=key, entry_type=entry_type, fields=fields))
    return entries


# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------

ARXIV_ID_RE = re.compile(r"(?:arxiv[:\s/]*(?:abs/)?)?(\d{4}\.\d{4,5})(?:v\d+)?")
DOI_URL_RE = re.compile(r"(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+?)[,}\s]*$")


def extract_arxiv_id(entry):
    for fname in ("eprint", "journal", "url", "note", "howpublished"):
        value = strip_latex(entry.get(fname))
        m = ARXIV_ID_RE.search(value)
        if m and not value.lower().startswith("10."):
            return m.group(1)
    return None


def extract_doi(entry):
    for fname in ("doi", "url"):
        value = strip_latex(entry.get(fname))
        m = DOI_URL_RE.search(value)
        if m:
            return m.group(1)
    return None


def extract_url(entry):
    for fname in ("url", "howpublished"):
        value = strip_latex(entry.get(fname))
        m = re.search(r"https?://[^\s]+", value)
        if m:
            return m.group(0).rstrip(".")
    return None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_last_arxiv_call = [0.0]


def http_get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def arxiv_get(query_url, delay=3.0):
    """arXiv asks for >=3s between requests; enforce politely."""
    gap = time.time() - _last_arxiv_call[0]
    if gap < delay:
        time.sleep(delay - gap)
    try:
        data = http_get(query_url)
        _last_arxiv_call[0] = time.time()
        return data
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        _last_arxiv_call[0] = time.time()
        return None


ATOM = "{http://www.w3.org/2005/Atom}"


def parse_arxiv_feed(data):
    """Return a list of (title, year, [surnames], arxiv_id) from Atom XML."""
    out = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return out
    for entry in root.findall(f"{ATOM}entry"):
        title_el = entry.find(f"{ATOM}title")
        title = re.sub(r"\s+", " ", title_el.text or "").strip() if title_el is not None else ""
        published = entry.find(f"{ATOM}published")
        year = (published.text or "")[:4] if published is not None else ""
        authors = []
        for author in entry.findall(f"{ATOM}author/{ATOM}name"):
            name = (author.text or "").strip()
            if name:
                authors.append(extract_surname(name))
        eid = entry.find(f"{ATOM}id")
        arxiv_id = (eid.text or "").rsplit("/abs/", 1)[-1] if eid is not None else ""
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        if title:
            out.append((title, year, authors, arxiv_id))
    return out


def query_arxiv_by_id(arxiv_id, delay):
    url = f"{ARXIV_API}?id_list={urllib.parse.quote(arxiv_id)}&max_results=1"
    data = arxiv_get(url, delay)
    if data:
        results = parse_arxiv_feed(data)
        if results:
            title, year, authors, found_id = results[0]
            return Candidate("arxiv", found_id, title, year, authors)
    return None


def query_arxiv_by_title(title, delay):
    phrase = urllib.parse.quote(f'"{strip_latex(title)}"')
    url = f"{ARXIV_API}?search_query=ti:{phrase}&max_results=5"
    data = arxiv_get(url, delay)
    if data:
        results = parse_arxiv_feed(data)
        if results:
            t, year, authors, found_id = results[0]
            return Candidate("arxiv", found_id, t, year, authors)
    return None


def query_crossref_doi(doi, email=None):
    suffix = f"&mailto={email}" if email else ""
    try:
        data = http_get(f"{CROSSREF_API}/works/{urllib.parse.quote(doi)}{suffix}")
        msg = json.loads(data).get("message", {})
        title = re.sub(r"\s+", " ", (msg.get("title") or [""])[0]).strip()
        year = ""
        issued = msg.get("issued", {}).get("date-parts", [])
        if issued and issued[0]:
            year = str(issued[0][0])
        authors = [re.sub(r"[^a-zA-Z]", "", a.get("family", "")).lower()
                   for a in msg.get("author", []) if a.get("family")]
        return Candidate("crossref", doi, title, year, authors)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def query_crossref_title(title, email=None):
    q = urllib.parse.quote(strip_latex(title))
    suffix = f"&mailto={email}" if email else ""
    try:
        data = http_get(f"{CROSSREF_API}/works?query.bibliographic={q}&rows=3{suffix}")
        items = json.loads(data).get("message", {}).get("items", [])
        if not items:
            return None
        best = None
        best_sim = 0.0
        norm = normalize_title(title)
        for item in items:
            ct = re.sub(r"\s+", " ", (item.get("title") or [""])[0]).strip()
            sim = title_similarity(norm, normalize_title(ct))
            if sim > best_sim:
                best_sim = sim
                issued = item.get("issued", {}).get("date-parts", [])
                year = str(issued[0][0]) if issued and issued[0] else ""
                authors = [re.sub(r"[^a-zA-Z]", "", a.get("family", "")).lower()
                           for a in item.get("author", []) if a.get("family")]
                best = Candidate("crossref", item.get("DOI", ""), ct, year, authors)
        if best_sim >= 0.5:
            return best
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        pass
    return None


def query_openalex_title(title, email=None):
    q = urllib.parse.quote(strip_latex(title))
    suffix = f"&mailto={email}" if email else ""
    try:
        data = http_get(f"{OPENALEX_API}?search={q}&per-page=3{suffix}")
        results = json.loads(data).get("results", [])
        if not results:
            return None
        best = None
        best_sim = 0.0
        norm = normalize_title(title)
        for item in results:
            ct = re.sub(r"\s+", " ", item.get("display_name") or "").strip()
            sim = title_similarity(norm, normalize_title(ct))
            if sim > best_sim:
                best_sim = sim
                authors = []
                for a in item.get("authorships", []) or []:
                    name = (a.get("author") or {}).get("display_name") or ""
                    if name:
                        authors.append(extract_surname(name))
                best = Candidate("openalex", item.get("doi", "") or item.get("id", ""),
                                 ct, str(item.get("publication_year") or ""), authors)
        if best_sim >= 0.5:
            return best
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        pass
    return None


def check_web_url(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status < 400
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------

def find_candidate(entry, delay, email):
    """Locate the canonical record for a bib entry, trying multiple sources."""
    arxiv_id = extract_arxiv_id(entry)
    if arxiv_id:
        cand = query_arxiv_by_id(arxiv_id, delay)
        if cand:
            return cand, f"arXiv:{arxiv_id}"
        # ID didn't resolve: fall through to title search (likely fabricated ID).

    doi = extract_doi(entry)
    if doi:
        cand = query_crossref_doi(doi, email)
        if cand and cand.title:
            return cand, f"doi:{doi}"

    title = entry.get("title")
    if not title:
        return None, "no-title"

    cand = query_arxiv_by_title(title, delay)
    if cand and title_similarity(normalize_title(title), normalize_title(cand.title)) >= 0.5:
        return cand, f"arxiv-title:{cand.found_id}"
    cand = query_crossref_title(title, email)
    if cand:
        return cand, f"crossref-title:{cand.found_id}"
    cand = query_openalex_title(title, email)
    if cand:
        return cand, f"openalex-title:{cand.found_id}"
    return None, "not-found"


def score_entry(entry, cand):
    """Compute (title_sim, year_ok, first_author_ok, author_overlap, notes)."""
    notes = []
    claimed_title = normalize_title(entry.get("title", ""))
    title_sim = title_similarity(claimed_title, normalize_title(cand.title))

    claimed_year = strip_latex(entry.get("year", "")).strip()
    year_ok = True
    if claimed_year.isdigit() and cand.year.isdigit():
        year_ok = abs(int(claimed_year) - int(cand.year)) <= 1
        if not year_ok:
            notes.append(f"year mismatch: claimed {claimed_year}, found {cand.year}")

    claimed_surnames, has_others = parse_authors(entry.get("author"))
    author_overlap = None
    first_author_ok = True
    if claimed_surnames and cand.authors:
        overlap = sum(1 for s in claimed_surnames if s in cand.authors)
        author_overlap = overlap / len(claimed_surnames)
        first_author_ok = claimed_surnames[0] in cand.authors
        if not first_author_ok:
            # Org-style first authors (e.g. "OpenAI") plus generic-name entries:
            # fall back to overall overlap before flagging.
            if author_overlap >= 0.5 or (has_others and author_overlap >= 0.34):
                notes.append("first author written differently (org name or variant)")
                first_author_ok = True
            else:
                notes.append(
                    f"first author '{claimed_surnames[0]}' not in source author list")
        if author_overlap < 0.34:
            notes.append(
                f"low author overlap ({overlap}/{len(claimed_surnames)} claimed names found)")

    return title_sim, year_ok, first_author_ok, author_overlap, notes


def classify(entry, cand, title_sim, year_ok, first_author_ok, author_overlap, url_ok, how):
    entry_type = entry.entry_type
    if entry_type == "book" and cand is None and how in ("not-found", "no-title"):
        return UNVERIFIED, "book without resolvable identifier; verify manually"

    if cand is None:
        if extract_arxiv_id(entry) or extract_doi(entry):
            return ERROR, "declared identifier (arXiv/DOI) does not resolve to any record"
        if url_ok:
            return UNVERIFIED, "URL resolves but no bibliographic record found"
        if entry_type == "misc" and extract_url(entry):
            return WARNING, "cited web URL did not resolve (may be geo/paywalled)"
        return NOT_FOUND, "no matching record in arXiv, Crossref, or OpenAlex"

    if entry.entry_type == "book" and not (title_sim >= 0.85 and year_ok and first_author_ok):
        # Books are poorly covered by arXiv/Crossref/OpenAlex; a wrong-work or
        # author mismatch on a book is usually a false positive. Cap at WARNING.
        return WARNING, "book: registries lack a close record; verify manually"

    if title_sim < 0.60:
        return ERROR, f"matched record has a different title (similarity {title_sim:.2f})"

    if title_sim < 0.85:
        base = WARNING
    elif year_ok and first_author_ok:
        return VERIFIED, "title, year, and authors match"
    else:
        base = WARNING

    if not first_author_ok and author_overlap is not None and author_overlap < 0.34:
        return ERROR, "suspected misattribution: claimed authors do not match source"
    return base, "; ".join(
        n for n in [
            "title partially matches" if title_sim < 0.85 else "",
            "" if year_ok else "year mismatch",
            "" if first_author_ok else "author mismatch",
        ] if n) or "minor metadata differences"


# ---------------------------------------------------------------------------
# TeX cross-check
# ---------------------------------------------------------------------------

CITE_RE = re.compile(r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear)\*?"
                     r"(?:\[[^\]]*\])*\{([^}]+)\}")


def collect_cited_keys(paper_dir):
    keys = set()
    for root, _dirs, files in os.walk(paper_dir):
        if "tables" in root or os.path.join(paper_dir, "tables") in root:
            continue
        for fname in files:
            if fname.endswith(".tex"):
                with open(os.path.join(root, fname), "r", encoding="utf-8",
                          errors="replace") as f:
                    for m in CITE_RE.finditer(f.read()):
                        for k in m.group(1).split(","):
                            k = k.strip()
                            if k:
                                keys.add(k)
    return keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_entry(entry, args):
    """Run full verification for one bib entry; return a result dict."""
    result = {
        "key": entry.key,
        "entry_type": entry.entry_type,
        "claimed_title": strip_latex(entry.get("title", ""))[:120],
        "claimed_year": strip_latex(entry.get("year", "")),
        "claimed_first_author": (parse_authors(entry.get("author"))[0] or [""])[0],
        "status": UNVERIFIED,
        "title_sim": "",
        "found_year": "",
        "found_title": "",
        "source": "",
        "notes": "",
    }

    url = extract_url(entry)
    url_ok = False
    if url and not args.skip_web:
        url_ok = check_web_url(url)

    cand, how = find_candidate(entry, args.delay, args.email)
    if cand:
        title_sim, year_ok, first_ok, overlap, notes = score_entry(entry, cand)
        status, note = classify(entry, cand, title_sim, year_ok, first_ok,
                                overlap, url_ok, how)
        result.update({
            "status": status,
            "title_sim": f"{title_sim:.2f}",
            "found_year": cand.year,
            "found_title": cand.title[:120],
            "source": f"{cand.source}:{cand.found_id}",
            "notes": "; ".join(filter(None, [note] + notes)),
        })
    else:
        status, note = classify(entry, None, 0, False, False, None, url_ok, how)
        result.update({"status": status, "notes": note})
    return result


def write_report(results, missing_keys, unused_keys, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "citation_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    md_path = os.path.join(out_dir, "citation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Citation Check Report\n\n")
        f.write("Generated by `check_citations.py` (judgment-to-noise toolkit).\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Entries checked: **{len(results)}**\n")
        for status in (VERIFIED, WARNING, ERROR, NOT_FOUND, UNVERIFIED):
            if status in counts:
                f.write(f"- {STATUS_EMOJI[status]} {status}: **{counts[status]}**\n")
        if missing_keys:
            f.write(f"\n## Cited in .tex but missing from .bib ({len(missing_keys)})\n\n")
            for k in sorted(missing_keys):
                f.write(f"- `{k}`\n")
        if unused_keys:
            f.write(f"\n## In .bib but never cited ({len(unused_keys)})\n\n")
            for k in sorted(unused_keys):
                f.write(f"- `{k}`\n")

        f.write("\n## Entries needing review\n\n")
        f.write("| Status | Key | Claimed (first author, year) | Found title | Note |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            if r["status"] == VERIFIED:
                continue
            f.write(f"| {STATUS_EMOJI[r['status']]} {r['status']} | `{r['key']}` "
                    f"| {r['claimed_first_author']}, {r['claimed_year']} "
                    f"| {r['found_title'] or '—'} "
                    f"| {r['notes']} |\n")

        f.write("\n## All entries\n\n")
        f.write("| Status | Key | Sim | Source | Found title |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {STATUS_EMOJI[r['status']]} {r['status']} | `{r['key']}` "
                    f"| {r['title_sim']} | {r['source'] or '—'} "
                    f"| {r['found_title'] or '—'} |\n")

    return csv_path, md_path, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper_dir", help="Directory containing main.tex / references.bib")
    parser.add_argument("--bib", default=None, help="Path to .bib file (default: <paper_dir>/references.bib)")
    parser.add_argument("--output-dir", default="tables/citation_check",
                        help="Where to write the report (default: tables/citation_check)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds between arXiv API calls (default: 3.0, per arXiv TOS)")
    parser.add_argument("--email", default=None,
                        help="Contact email for Crossref/OpenAlex polite pools")
    parser.add_argument("--skip-web", action="store_true",
                        help="Do not verify plain-web URLs (news articles, press releases)")
    args = parser.parse_args()

    bib_path = args.bib or os.path.join(args.paper_dir, "references.bib")
    if not os.path.isfile(bib_path):
        sys.exit(f"Bib file not found: {bib_path}")

    entries = parse_bib_file(bib_path)
    print(f"Parsed {len(entries)} entries from {bib_path}")

    dup_titles = {}
    for e in entries:
        t = normalize_title(e.get("title", ""))
        if t:
            dup_titles.setdefault(t, []).append(e.key)
    dups = {t: ks for t, ks in dup_titles.items() if len(ks) > 1}

    results = []
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] checking {entry.key} ...", flush=True)
        result = check_entry(entry, args)
        if entry.key in dups and normalize_title(entry.get("title", "")) in dups:
            result["notes"] = (result["notes"] +
                               f" | duplicate title with: {', '.join(dups[entry.key])}").strip(" |")
        results.append(result)

    cited_keys = collect_cited_keys(args.paper_dir)
    bib_keys = {e.key for e in entries}
    missing = cited_keys - bib_keys
    unused = bib_keys - cited_keys

    csv_path, md_path, counts = write_report(results, missing, unused, args.output_dir)

    print("\n=== Summary ===")
    for status in (VERIFIED, WARNING, ERROR, NOT_FOUND, UNVERIFIED):
        if status in counts:
            print(f"  {STATUS_EMOJI[status]} {status}: {counts[status]}")
    if missing:
        print(f"  ⧗ cited-but-missing keys: {len(missing)} -> {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
    if unused:
        print(f"  ✂ unused bib entries: {len(unused)}")
    print(f"\nCSV report: {csv_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
