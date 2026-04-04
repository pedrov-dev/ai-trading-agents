from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap


PAGE_WIDTH = 1280
PAGE_HEIGHT = 720
MARGIN_X = 72
TOP = 654
BOTTOM = 56
CONTENT_WIDTH = PAGE_WIDTH - (MARGIN_X * 2)


def pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", " ")
    )


def approx_wrap(text: str, font_size: int, max_width: int) -> list[str]:
    chars_per_line = max(18, int(max_width / (font_size * 0.56)))
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(wrap(paragraph, width=chars_per_line, break_long_words=False))
    return lines or [""]


@dataclass(frozen=True)
class Slide:
    title: str
    body_lines: list[str]
    kicker: str | None = None


def text_block(x: int, y: int, lines: list[str], font: str, size: int, leading: int) -> str:
    parts = [f"BT /{font} {size} Tf {x} {y} Td"]
    for index, line in enumerate(lines):
        if index > 0:
            parts.append(f"0 -{leading} Td")
        parts.append(f"({pdf_escape(line)}) Tj")
    parts.append("ET")
    return "\n".join(parts)


def draw_slide(slide: Slide, number: int, total: int) -> str:
    parts: list[str] = []
    parts.append("q")
    parts.append("0.070 0.098 0.153 rg")
    parts.append(f"0 0 {PAGE_WIDTH} {PAGE_HEIGHT} re f")
    parts.append("0.118 0.192 0.286 rg")
    parts.append(f"0 {PAGE_HEIGHT - 118} {PAGE_WIDTH} 118 re f")
    parts.append("0.925 0.569 0.176 rg")
    parts.append(f"0 {PAGE_HEIGHT - 14} {PAGE_WIDTH} 14 re f")
    parts.append("0.137 0.827 0.686 rg")
    parts.append(f"{PAGE_WIDTH - 250} 0 250 {PAGE_HEIGHT} re f")
    parts.append("Q")

    parts.append("0.094 0.137 0.200 rg")
    parts.append("0.18 0.18 0.18 RG")
    parts.append("2 w")
    parts.append(f"{MARGIN_X - 16} {BOTTOM - 10} {CONTENT_WIDTH + 32} {PAGE_HEIGHT - 160} re S")

    if slide.kicker:
        kicker_lines = approx_wrap(slide.kicker, 14, CONTENT_WIDTH)
        parts.append("0.929 0.957 1.000 rg")
        parts.append(text_block(MARGIN_X, TOP + 44, kicker_lines, "F1", 14, 18))

    parts.append("0.988 0.992 0.996 rg")
    parts.append(text_block(MARGIN_X, TOP, approx_wrap(slide.title, 30, CONTENT_WIDTH), "F2", 30, 34))

    body_y = TOP - 74
    body_lines: list[str] = []
    for line in slide.body_lines:
        if line.startswith("[BOX]"):
            body_lines.append(line)
        else:
            wrapped = approx_wrap(line, 18, CONTENT_WIDTH - 34)
            body_lines.extend([f"- {wrapped[0]}"] + [f"  {chunk}" for chunk in wrapped[1:]])

    current_y = body_y
    for line in body_lines:
        if line.startswith("[BOX]"):
            box_title, box_text = line[5:].split("|", 1)
            parts.append("0.102 0.153 0.224 rg")
            parts.append(f"{MARGIN_X} {current_y - 12} {CONTENT_WIDTH} 74 re f")
            parts.append("0.325 0.427 0.549 RG")
            parts.append("1.5 w")
            parts.append(f"{MARGIN_X} {current_y - 12} {CONTENT_WIDTH} 74 re S")
            parts.append("0.933 0.957 1.000 rg")
            parts.append(text_block(MARGIN_X + 18, current_y + 30, approx_wrap(box_title, 18, CONTENT_WIDTH - 36), "F2", 18, 22))
            parts.append("0.796 0.843 0.902 rg")
            parts.append(text_block(MARGIN_X + 18, current_y + 4, approx_wrap(box_text, 14, CONTENT_WIDTH - 36), "F1", 14, 17))
            current_y -= 100
            continue

        if line.startswith("- "):
            wrapped = approx_wrap(line, 18, CONTENT_WIDTH - 28)
            parts.append("0.933 0.957 1.000 rg")
            parts.append(text_block(MARGIN_X + 6, current_y, wrapped, "F1", 18, 23))
            current_y -= 36 + (len(wrapped) - 1) * 22
        else:
            wrapped = approx_wrap(line, 18, CONTENT_WIDTH)
            parts.append("0.796 0.843 0.902 rg")
            parts.append(text_block(MARGIN_X, current_y, wrapped, "F1", 18, 22))
            current_y -= 30 + (len(wrapped) - 1) * 20

    parts.append("0.522 0.584 0.675 rg")
    footer = f"ai-trading-agents  |  slide {number}/{total}"
    parts.append(text_block(MARGIN_X, 28, [footer], "F1", 11, 12))
    return "\n".join(parts)


def pdf_stream(content: str) -> bytes:
    data = content.encode("latin-1")
    return f"<< /Length {len(data)} >>\nstream\n".encode("ascii") + data + b"\nendstream"


def build_pdf(slides: list[Slide], output_path: Path) -> None:
    objects: list[bytes] = []

    def add_object(payload: bytes | str) -> int:
        if isinstance(payload, str):
            payload = payload.encode("latin-1")
        objects.append(payload)
        return len(objects)

    font_regular = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_numbers: list[int] = []
    for index, slide in enumerate(slides, start=1):
        stream = pdf_stream(draw_slide(slide, index, len(slides)))
        content_obj = add_object(stream)
        page_obj = add_object(
            (
                f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> >> "
                f"/Contents {content_obj} 0 R >>"
            ).encode("ascii")
        )
        page_numbers.append(page_obj)

    kids = " ".join(f"{obj} 0 R" for obj in page_numbers)
    pages_obj = add_object(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode("ascii")
    )

    catalog_obj = add_object(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode("ascii"))

    for page_obj in page_numbers:
        page_index = page_obj - 1
        objects[page_index] = objects[page_index].replace(
            b"/Parent 0 0 R", f"/Parent {pages_obj} 0 R".encode("ascii")
        )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, payload in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(payload)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf)


def main() -> int:
    slides = [
        Slide(
            title="AI Trading Agents",
            kicker="A local-first crypto trading MVP for the Kraken and ERC-8004 hackathon tracks.",
            body_lines=[
                "Event-driven agent that turns market signals into explainable trade intents.",
                "Safe by default: local dry-run is the baseline, with explicit live-connected paper mode.",
                "Built to show both execution quality and trust signals in one submission.",
            ],
        ),
        Slide(
            title="The problem",
            body_lines=[
                "Hackathon agents need more than a model output - they need a real financial function.",
                "Traders and judges also need clear evidence: what was seen, why a decision was made, and what happened next.",
                "Pure signal demos are weak; production-minded control, validation, and auditability are stronger.",
            ],
        ),
        Slide(
            title="The solution",
            body_lines=[
                "[BOX]One agent, two challenge paths|A Kraken-style trading loop for market execution and an ERC-8004 path for identity, reputation, and validation artifacts.",
                "Ingests RSS news and Kraken price quotes.",
                "Classifies crypto-relevant events and generates trade intents.",
                "Applies conservative risk checks before paper execution.",
            ],
        ),
        Slide(
            title="How it works",
            body_lines=[
                "Ingest -> Detect -> Strategy -> Risk -> Execution -> Audit",
                "Each run writes audit logs, validation artifacts, checkpoints, and a summary file.",
                "The runtime supports a safe paper mode, a scheduler service, and shared Sepolia actions when explicitly enabled.",
            ],
        ),
        Slide(
            title="Why it matters",
            body_lines=[
                "Simple to explain, easy to inspect, and designed to avoid accidental live trading.",
                "Shows both measurable trading behavior and verifiable on-chain progress.",
                "A clean hackathon story: one repo, one agent, two prize tracks.",
            ],
        ),
    ]

    output_path = Path("artifacts") / "ai_trading_agents_pitch_deck.pdf"
    build_pdf(slides, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())