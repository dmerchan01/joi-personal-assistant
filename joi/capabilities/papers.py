"""Capability: AI research papers — latest arXiv submissions and Hugging Face
trending daily papers.

Endpoints verified live 2026-07-05:
  - http://export.arxiv.org/api/query (Atom XML, no key; parsed with stdlib)
  - https://huggingface.co/api/daily_papers (JSON, no key — it DOES exist)
"""
import subprocess
import xml.etree.ElementTree as ET

import httpx

from joi.capabilities import Capability

_ARXIV_URL = "https://export.arxiv.org/api/query"  # http:// 301-redirects
_HF_URL = "https://huggingface.co/api/daily_papers"
_ATOM = "{http://www.w3.org/2005/Atom}"

# last listed papers: (title, link) — shared by both sources for open_paper
_last_papers: list[tuple[str, str]] = []


def _first_phrase(text: str) -> str:
    text = " ".join(text.split())
    return text.split(". ")[0].rstrip(".") + "."


def ai_papers_arxiv(topic: str = "cs.AI", n: int = 3) -> str:
    """Get the latest arXiv paper submissions for a category like cs.AI or
    cs.CL, or a keyword like 'diffusion models' (max 5)."""
    global _last_papers
    try:
        n = max(1, min(int(n), 5))
        # category codes contain a dot ("cs.AI"); anything else is a keyword
        query = f"cat:{topic}" if "." in topic and " " not in topic else f"all:{topic}"
        r = httpx.get(_ARXIV_URL, params={
            "search_query": query, "sortBy": "submittedDate",
            "sortOrder": "descending", "max_results": n,
        }, timeout=20)
        entries = ET.fromstring(r.text).findall(f"{_ATOM}entry")
        if not entries:
            return f"I found no recent arXiv papers for {topic}."
        _last_papers = []
        spoken = []
        for i, e in enumerate(entries, 1):
            title = " ".join(e.findtext(f"{_ATOM}title", "").split())
            summary = e.findtext(f"{_ATOM}summary", "")
            link = e.findtext(f"{_ATOM}id", "https://arxiv.org")
            _last_papers.append((title, link))
            spoken.append(f"Number {i}: {title}. {_first_phrase(summary)}")
        return ("Latest arXiv papers. " + " ".join(spoken) +
                " Say open paper one, two and so on to read one.")
    except Exception:
        return "I couldn't reach arXiv right now."


def ai_papers_trending(n: int = 3) -> str:
    """Get today's trending AI papers from Hugging Face (max 5)."""
    global _last_papers
    try:
        n = max(1, min(int(n), 5))
        papers = httpx.get(_HF_URL, params={"limit": n}, timeout=15).json()[:n]
        if not papers:
            return "Hugging Face has no trending papers listed today."
        _last_papers = []
        spoken = []
        for i, item in enumerate(papers, 1):
            p = item.get("paper", {})
            title = p.get("title", "untitled")
            _last_papers.append(
                (title, f"https://huggingface.co/papers/{p.get('id', '')}"))
            gist = _first_phrase(p.get("summary", "")) if p.get("summary") else ""
            spoken.append(f"Number {i}: {title}. {gist}".strip())
        return ("Trending papers on Hugging Face. " + " ".join(spoken) +
                " Say open paper one, two and so on to read one.")
    except Exception:
        return "I couldn't reach Hugging Face right now."


def open_paper(rank: int = 1) -> str:
    """Open paper number `rank` from the last papers list in the browser."""
    if not _last_papers:
        return "I don't have a papers list yet — ask for recent papers first."
    rank = int(rank)
    if not 1 <= rank <= len(_last_papers):
        return f"I only have {len(_last_papers)} papers listed."
    title, url = _last_papers[rank - 1]
    subprocess.Popen(["xdg-open", url], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"Opening {title}."


CAPABILITY = Capability(
    name="papers",
    description=("AI research papers: latest arXiv submissions by topic and "
                 "trending Hugging Face daily papers; can open one in the browser."),
    tools=[ai_papers_arxiv, ai_papers_trending, open_paper],
)
