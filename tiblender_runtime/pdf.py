"""Minimal, dependency-free PDF export of a report's Markdown text.

The writer packages the Markdown as readable pages for compact runtime
demonstrations; it is not a general-purpose PDF renderer.
"""

from __future__ import annotations

import re
import textwrap


def _pdf_text_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"^\|[-:| ]+\|$", "", line)
        line = line.replace("**", "")
        # Headings are rendered as plain text lines; the recommendation
        # groups introduced with the released report format use `### `, so
        # strip the deepest marker first.
        if line.startswith("### "):
            line = line[4:]
        elif line.startswith("## "):
            line = line[3:]
        elif line.startswith("# "):
            line = line[2:]
        if not line:
            lines.append("")
            continue
        for wrapped in textwrap.wrap(line, width=92, break_long_words=True, replace_whitespace=False):
            lines.append(wrapped)
    return lines or [""]


def _pdf_escape(text: str) -> str:
    safe = text.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_bytes(markdown: str) -> bytes:
    # The PDF writer is deliberately minimal and dependency-free; it packages
    # the Markdown text as readable pages for compact runtime demonstrations.
    lines = _pdf_text_lines(markdown)
    pages = [lines[index : index + 48] for index in range(0, len(lines), 48)] or [[""]]
    objects: list[bytes] = []

    def add_object(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_object(b"")
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    for page_lines in pages:
        escaped_lines = [f"({_pdf_escape(line)}) Tj" for line in page_lines]
        content = "BT\n/F1 10 Tf\n50 760 Td\n14 TL\n" + "\nT*\n".join(escaped_lines) + "\nET\n"
        content_bytes = content.encode("latin-1", "replace")
        page_id = len(objects) + 1
        content_id = page_id + 1
        page_ids.append(page_id)
        add_object(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        add_object(b"<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"endstream")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)
