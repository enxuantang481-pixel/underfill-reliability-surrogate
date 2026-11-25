from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


SOURCE = Path(r".\Competition")
OUTPUT = Path(__file__).with_name("official_audit")

PDFS = {
    "technical_overview": "ESTC 2026 Student Competition – Technical Subject Overview (1).pdf",
    "call_for_participation": "IEEE ESTC 2026 Student Competition – Call for Participation.pdf",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for output_name, filename in PDFS.items():
        pdf_path = SOURCE / filename
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(f"\n\n===== PAGE {page_number} =====\n\n")
            pages.append(page.extract_text(extraction_mode="layout") or "")
        target = OUTPUT / f"{output_name}.txt"
        target.write_text("".join(pages), encoding="utf-8")
        print(f"{pdf_path.name}: {len(reader.pages)} pages -> {target}")


if __name__ == "__main__":
    main()
