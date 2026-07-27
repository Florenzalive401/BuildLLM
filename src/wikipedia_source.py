"""Download and stream clean article text from official Wikimedia dumps."""

from __future__ import annotations

import bz2
import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
try:
    from mwparserfromhell import parse as parse_wikitext
except ImportError:  # pragma: no cover - fallback keeps basic extraction available
    parse_wikitext = None
from tqdm import tqdm


DUMPS_BASE_URL = "https://dumps.wikimedia.org"
DEFAULT_PROJECT = "enwiki"
DUMP_SUFFIX = "pages-articles-multistream.xml.bz2"
USER_AGENT = "BuildLLM-WikipediaCorpus/1.0 (research corpus acquisition; contact: local-user)"
REQUEST_HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}


@dataclass(frozen=True)
class WikipediaArticle:
    title: str
    text: str
    page_id: int | None = None


def dump_filename(project: str = DEFAULT_PROJECT) -> str:
    return f"{project}-latest-{DUMP_SUFFIX}"


def dump_url(project: str = DEFAULT_PROJECT) -> str:
    filename = dump_filename(project)
    return f"{DUMPS_BASE_URL}/{project}/latest/{filename}"


def md5_url(project: str = DEFAULT_PROJECT) -> str:
    return f"{DUMPS_BASE_URL}/{project}/latest/{project}-latest-md5sums.txt"


def _remote_size(url: str, timeout: int = 60) -> int | None:
    headers = {**REQUEST_HEADERS, "Range": "bytes=0-0"}
    response = requests.get(url, headers=headers, stream=True, timeout=timeout)
    response.raise_for_status()
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/(\d+)$", content_range)
    if match:
        response.close()
        return int(match.group(1))
    value = response.headers.get("Content-Length")
    response.close()
    return int(value) if value and value.isdigit() else None


def download_dump(
    destination: Path,
    project: str = DEFAULT_PROJECT,
    chunk_size: int = 8 * 1024 * 1024,
    timeout: int = 120,
) -> Path:
    """Download the current article dump with safe resume support."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = dump_url(project)
    expected_size = _remote_size(url, timeout=timeout)
    existing_size = destination.stat().st_size if destination.exists() else 0

    if expected_size is not None and existing_size == expected_size:
        return destination

    headers: dict[str, str] = dict(REQUEST_HEADERS)
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    if existing_size > 0 and response.status_code != 206:
        existing_size = 0
        mode = "wb"

    total = expected_size
    with destination.open(mode) as output, tqdm(
        total=total,
        initial=existing_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading {project}",
    ) as progress:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            output.write(chunk)
            progress.update(len(chunk))

    if expected_size is not None and destination.stat().st_size != expected_size:
        raise IOError(
            "Wikipedia dump download is incomplete: "
            f"expected {expected_size:,} bytes, found {destination.stat().st_size:,}"
        )

    return destination


def expected_md5(project: str = DEFAULT_PROJECT, timeout: int = 60) -> str | None:
    """Read the official checksum for the selected article dump."""

    response = requests.get(md5_url(project), headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    filename = dump_filename(project)
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == filename:
            return parts[0].lower()
    return None


def verify_md5(path: Path, expected: str, chunk_size: int = 8 * 1024 * 1024) -> bool:
    digest = hashlib.md5()  # nosec B324: required to verify Wikimedia's published checksum
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected.lower()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child.text
    return None


def _revision_text(page: ET.Element) -> str | None:
    for child in page:
        if _local_name(child.tag) != "revision":
            continue
        for revision_child in child:
            if _local_name(revision_child.tag) == "text":
                return revision_child.text
    return None


def wikitext_to_plain_text(raw_text: str) -> str:
    """Remove MediaWiki markup and common non article remnants."""

    if parse_wikitext is not None:
        code = parse_wikitext(raw_text)
        text = code.strip_code(normalize=True, collapse=True)
    else:
        text = re.sub(r"\{\{.*?\}\}", "", raw_text, flags=re.DOTALL)
        text = re.sub(r"\[\[(?:[^]|]*\|)?([^]]+)\]\]", r"\1", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"={2,}([^=]+)={2,}", r"\1", text)
    text = re.sub(r"(?im)^\s*(references|external links|see also|further reading)\s*$.*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stream_articles(dump_path: Path) -> Iterator[WikipediaArticle]:
    """Stream namespace zero, non redirect articles without loading the dump into RAM."""

    with bz2.open(dump_path, "rb") as source:
        for event, element in ET.iterparse(source, events=("end",)):
            if _local_name(element.tag) != "page":
                continue

            namespace = _child_text(element, "ns")
            title = _child_text(element, "title") or ""
            page_id_text = _child_text(element, "id")
            raw_text = _revision_text(element) or ""
            is_redirect = any(_local_name(child.tag) == "redirect" for child in element)

            if namespace == "0" and not is_redirect and raw_text:
                plain_text = wikitext_to_plain_text(raw_text)
                page_id = int(page_id_text) if page_id_text and page_id_text.isdigit() else None
                yield WikipediaArticle(title=title, text=plain_text, page_id=page_id)

            element.clear()
