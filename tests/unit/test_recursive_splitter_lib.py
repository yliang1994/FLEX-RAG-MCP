from __future__ import annotations

from libs.splitter.recursive_splitter import RecursiveSplitter


def test_recursive_splitter_preserves_markdown_headings() -> None:
    splitter = RecursiveSplitter(chunk_size=48, chunk_overlap=8)

    chunks = splitter.split_text(
        "# Intro\nThis is a short introduction.\n\n## Details\nMore details live here and should stay under the correct heading."
    )

    assert chunks[0].startswith("# Intro")
    assert any(chunk.startswith("## Details") for chunk in chunks)


def test_recursive_splitter_keeps_fenced_code_block_intact() -> None:
    splitter = RecursiveSplitter(chunk_size=40, chunk_overlap=8)

    chunks = splitter.split_text(
        "# Example\n\n```python\nprint('alpha')\nprint('beta')\n```\n\nClosing explanation paragraph."
    )

    code_chunks = [chunk for chunk in chunks if "```python" in chunk]
    assert len(code_chunks) == 1
    assert "print('alpha')" in code_chunks[0]
    assert "print('beta')" in code_chunks[0]


def test_recursive_splitter_splits_large_paragraph_without_empty_chunks() -> None:
    splitter = RecursiveSplitter(chunk_size=24, chunk_overlap=4)

    chunks = splitter.split_text("alpha beta gamma delta epsilon zeta eta theta iota")

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 24 for chunk in chunks)
