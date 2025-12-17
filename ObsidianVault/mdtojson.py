import json
import re
from pathlib import Path


def collect_md_files(root: str, exclude_dirs=("Resources",)):
    root = Path(root)
    md_files = []

    for p in root.rglob("*.md"):
        if any(excl in p.parts for excl in exclude_dirs):
            continue
        md_files.append(p)

    return md_files


def parse_md_to_records(md_path: Path):
    text = md_path.read_text(encoding="utf-8")

    blocks = re.split(r"\n##\s+", text)
    records = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        title = lines[0].strip()
        body = " ".join(l.strip() for l in lines[1:] if l.strip())

        records.append({
            "timestamp": title,
            "text": body,
            "source": "md",
            "file": str(md_path)
        })

    return records


def md_folder_to_jsonl(root_dir: str, jsonl_path: str):
    md_files = collect_md_files(root_dir)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for md_path in md_files:
            records = parse_md_to_records(md_path)
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ===== usage =====
md_folder_to_jsonl(
    root_dir=".",          # プロジェクトルート
    jsonl_path="feedback.jsonl"
)
