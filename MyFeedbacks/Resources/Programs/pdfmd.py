# pdf_md_converter.py
import os
import sys

from pdfminer.high_level import extract_text
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate


# PDF → MD
def pdf_to_md(pdf_path, md_path=None):
    if md_path is None:
        base = os.path.splitext(pdf_path)[0]
        md_path = base + ".md"

    text = extract_text(pdf_path)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[OK] PDF→MD 変換完了: {md_path}")

# MD → PDF
def md_to_pdf(md_path, pdf_path=None):
    if pdf_path is None:
        base = os.path.splitext(md_path)[0]
        pdf_path = base + ".pdf"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Markdownの見出しや改行を単純に処理（本格的にやるならmarkdown2とか使う）
    story = []
    styles = getSampleStyleSheet()
    for line in md_text.splitlines():
        if line.startswith("#"):
            story.append(Paragraph(f"<b>{line.strip('# ')}</b>", styles["Heading1"]))
        else:
            story.append(Paragraph(line, styles["Normal"]))

    doc = SimpleDocTemplate(pdf_path)
    doc.build(story)

    print(f"[OK] MD→PDF 変換完了: {pdf_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使い方: python pdf_md_converter.py [mode] [file]")
        print("  mode: pdf2md or md2pdf")
        print("  例: python pdf_md_converter.py pdf2md ./docs/sample.pdf")
        print("      python pdf_md_converter.py md2pdf ./docs/sample.md")
        sys.exit(1)

    mode = sys.argv[1].lower()
    file_path = sys.argv[2]

    if mode == "pdf2md":
        pdf_to_md(file_path)
    elif mode == "md2pdf":
        md_to_pdf(file_path)
    else:
        print("modeは 'pdf2md' か 'md2pdf' を指定してください")