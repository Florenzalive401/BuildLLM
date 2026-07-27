import bz2
from pathlib import Path

from src.wikipedia_source import stream_articles, wikitext_to_plain_text


def test_wikitext_to_plain_text_removes_basic_markup():
    result = wikitext_to_plain_text("== History ==\n'''Alpha''' links to [[Beta|the second article]].")
    assert "Alpha" in result
    assert "the second article" in result
    assert "[[" not in result


def test_stream_articles_keeps_main_articles_and_skips_redirects(tmp_path: Path):
    xml = """<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.11/\">
    <page><title>Kept</title><ns>0</ns><id>1</id><revision><text>Useful article text with enough words.</text></revision></page>
    <page><title>Redirect</title><ns>0</ns><id>2</id><redirect title=\"Kept\"/><revision><text>#REDIRECT [[Kept]]</text></revision></page>
    <page><title>Talk:Skipped</title><ns>1</ns><id>3</id><revision><text>Talk text</text></revision></page>
    </mediawiki>"""
    path = tmp_path / "sample.xml.bz2"
    path.write_bytes(bz2.compress(xml.encode("utf-8")))

    articles = list(stream_articles(path))
    assert len(articles) == 1
    assert articles[0].title == "Kept"
    assert articles[0].page_id == 1
