#!/usr/bin/env python3
"""
HEP-VLMQA paper processing with MinerU.

Converts a directory of PDF papers into the per-paper structure used by
HEP-VLMQA:

<paper_id>/
    <paper_id>.md
    images/
    text_corpus.jsonl
    image_caption.jsonl

The script runs MinerU, copies extracted Markdown and non-formula images,
extracts text/formulas/tables, builds figure-caption/context records, and
writes a machine-readable processing summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger("hep_vlmqa.paper_processing")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FORMULA_NAME_PATTERNS = ("formula", "eq_", "math_", "latex_", "equation")

MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\((images/[^)]+\.(?:png|jpg|jpeg))\)",
    flags=re.IGNORECASE,
)
CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table)\s+"
    r"([A-Za-z]?\d+(?:\.\d+)*(?:[A-Za-z])?)"
    r"\s*[:.]?\s*(.*)$",
    flags=re.IGNORECASE,
)
DISPLAY_MATH_RE = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", flags=re.DOTALL)
LATEX_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", flags=re.DOTALL)
LATEX_INLINE_RE = re.compile(r"\\\((.*?)\\\)", flags=re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HEP-VLMQA paper content with MinerU."
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing source PDFs.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for per-paper outputs.")
    parser.add_argument("--mineru-command", default="mineru",
                        help="MinerU executable name/path (default: mineru).")
    parser.add_argument("--backend", default="pipeline",
                        help="MinerU backend (default: pipeline).")
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu",
                        help="Execution device policy (default: cpu).")
    parser.add_argument("--timeout", type=int, default=300,
                        help="MinerU timeout per attempt in seconds.")
    parser.add_argument("--retries", type=int, default=2,
                        help="MinerU attempts per paper (default: 2).")
    parser.add_argument("--retry-delay", type=int, default=10,
                        help="Seconds between failed attempts.")
    parser.add_argument("--context-window", type=int, default=4,
                        help="Neighboring Markdown lines used as figure context.")
    parser.add_argument("--formula-image-max-bytes", type=int, default=5000,
                        help="Original formula-image size heuristic threshold.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess existing paper output directories.")
    parser.add_argument("--keep-temporary", action="store_true",
                        help="Keep MinerU temporary outputs for debugging.")
    parser.add_argument("--log-level",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_formula_image(image_name: str, image_path: Path, max_bytes: int) -> bool:
    """Apply the original project formula-image filtering heuristics."""
    lowered = image_name.lower()
    if any(pattern in lowered for pattern in FORMULA_NAME_PATTERNS):
        return True
    if re.fullmatch(r"[a-f0-9]{32}\.png", lowered):
        return True
    try:
        return image_path.is_file() and image_path.stat().st_size < max_bytes
    except OSError:
        LOGGER.warning("Could not inspect image size: %s", image_path)
        return False


def get_context(lines: list[str], index: int, window: int) -> str:
    context: list[str] = []
    for line_index in range(max(0, index - window),
                            min(len(lines), index + 1 + window)):
        if line_index == index:
            continue
        line = lines[line_index].strip()
        if not line or MARKDOWN_IMAGE_RE.search(line):
            continue
        context.append(line)
    return clean_text(" ".join(context))


def extract_tables(content: str) -> list[str]:
    tables: list[str] = []
    lines = content.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("|") and "|" in line:
            table_lines: list[str] = []
            while index < len(lines):
                current = lines[index].strip()
                if not (current.startswith("|") and "|" in current):
                    break
                table_lines.append(current)
                index += 1
            if table_lines:
                tables.append("\n".join(table_lines))
            continue
        index += 1
    return tables


def extract_formulas(content: str) -> list[str]:
    formulas: list[str] = []
    for regex in (DISPLAY_MATH_RE, INLINE_MATH_RE, LATEX_DISPLAY_RE, LATEX_INLINE_RE):
        formulas.extend(regex.findall(content))
    seen: set[str] = set()
    unique: list[str] = []
    for formula in formulas:
        formula = formula.strip()
        if len(formula) <= 2 or formula in seen:
            continue
        seen.add(formula)
        unique.append(formula)
    return unique


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def copy_non_formula_images(source_dir: Path, destination_dir: Path,
                            formula_image_max_bytes: int) -> int:
    if not source_dir.is_dir():
        return 0
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if is_formula_image(source.name, source, formula_image_max_bytes):
            continue
        shutil.copy2(source, destination_dir / source.name)
        copied += 1
    return copied


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_markdown_and_build(markdown_path: Path, paper_id: str,
                             paper_output_dir: Path,
                             context_window: int) -> dict[str, int]:
    content = markdown_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    text_chunks: list[str] = []
    tables_from_scan: list[str] = []
    image_entries: list[dict[str, str]] = []
    current_caption = ""
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        caption_match = CAPTION_RE.search(line)
        if caption_match:
            current_caption = clean_text(caption_match.group(0))
            index += 1
            continue

        image_match = MARKDOWN_IMAGE_RE.search(line)
        if image_match:
            image_relative_path = image_match.group(1)
            context = get_context(lines, index, context_window)
            description = clean_text(f"{current_caption} {context}")
            if description:
                image_entries.append({
                    "paper_id": paper_id,
                    "image": image_relative_path,
                    "caption": current_caption,
                    "description": description,
                })
            current_caption = ""
            index += 1
            continue

        if line.startswith("|") and "|" in line:
            table_lines: list[str] = []
            while index < len(lines):
                current = lines[index].strip()
                if not (current.startswith("|") and "|" in current):
                    break
                table_lines.append(current)
                index += 1
            if table_lines:
                tables_from_scan.append("\n".join(table_lines))
            continue

        if line and not line.startswith("|"):
            clean_line = clean_text(MARKDOWN_IMAGE_RE.sub("", line))
            if len(clean_line) > 30:
                text_chunks.append(clean_line)
        index += 1

    formulas = extract_formulas(content)
    tables = unique_preserve_order([*tables_from_scan, *extract_tables(content)])

    text_rows: list[dict[str, str]] = []
    text_rows.extend({"paper_id": paper_id, "type": "text", "text": x}
                     for x in text_chunks)
    text_rows.extend({"paper_id": paper_id, "type": "formula", "text": x}
                     for x in formulas)
    text_rows.extend({"paper_id": paper_id, "type": "table", "text": x}
                     for x in tables if x.strip())

    write_jsonl(paper_output_dir / "text_corpus.jsonl", text_rows)
    write_jsonl(paper_output_dir / "image_caption.jsonl", image_entries)

    return {
        "text_chunks": len(text_chunks),
        "formulas": len(formulas),
        "tables": len(tables),
        "image_caption_entries": len(image_entries),
    }


def build_environment(device: str) -> dict[str, str]:
    environment = os.environ.copy()
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def run_mineru(pdf_path: Path, temporary_dir: Path, mineru_command: str,
               backend: str, environment: dict[str, str], timeout: int,
               retries: int, retry_delay: int) -> tuple[bool, str]:
    command = [mineru_command, "-p", str(pdf_path), "-o", str(temporary_dir),
               "--backend", backend]
    error_message = ""
    for attempt in range(1, retries + 1):
        LOGGER.info("MinerU attempt %d/%d: %s", attempt, retries, pdf_path.name)
        try:
            result = subprocess.run(
                command, env=environment, capture_output=True, text=True,
                timeout=timeout, check=False,
            )
            if result.returncode == 0:
                return True, ""
            error_message = (result.stderr.strip() or result.stdout.strip()
                             or f"MinerU exited with status {result.returncode}")
        except subprocess.TimeoutExpired:
            error_message = f"MinerU timed out after {timeout} seconds."
        except OSError as exc:
            error_message = f"Could not execute MinerU: {exc}"
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

        LOGGER.warning("MinerU attempt %d failed: %s",
                       attempt, error_message[:500])
        if attempt < retries:
            time.sleep(retry_delay)
    return False, error_message


def locate_mineru_output(temporary_dir: Path,
                         paper_id: str) -> tuple[Path | None, Path | None]:
    for output_dir in (temporary_dir / paper_id / "auto",
                       temporary_dir / paper_id):
        if not output_dir.is_dir():
            continue
        markdown_files = sorted(
            path for path in output_dir.glob("*.md")
            if not path.name.endswith("_layout.md")
        )
        if markdown_files:
            return markdown_files[0], output_dir / "images"
    return None, None


def process_paper(pdf_path: Path, output_base: Path,
                  environment: dict[str, str], args: argparse.Namespace) -> dict:
    paper_id = pdf_path.stem
    paper_output_dir = output_base / paper_id
    temporary_dir = output_base / f".mineru_tmp_{paper_id}"
    record: dict = {"paper_id": paper_id, "source_pdf": pdf_path.name,
                    "status": "pending"}

    if paper_output_dir.exists():
        if not args.overwrite:
            LOGGER.info("Skipping existing paper: %s", paper_id)
            record["status"] = "skipped_existing"
            return record
        shutil.rmtree(paper_output_dir)

    shutil.rmtree(temporary_dir, ignore_errors=True)
    LOGGER.info("Processing paper: %s", paper_id)

    success, error = run_mineru(
        pdf_path, temporary_dir, args.mineru_command, args.backend,
        environment, args.timeout, args.retries, args.retry_delay,
    )
    if not success:
        LOGGER.error("MinerU failed for %s: %s", paper_id, error[:500])
        record.update({"status": "mineru_failed", "error": error})
        if not args.keep_temporary:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        return record

    markdown_path, source_images_dir = locate_mineru_output(temporary_dir, paper_id)
    if markdown_path is None:
        LOGGER.error("No MinerU Markdown output found for %s", paper_id)
        record["status"] = "markdown_missing"
        if not args.keep_temporary:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        return record

    paper_output_dir.mkdir(parents=True, exist_ok=True)
    final_markdown_path = paper_output_dir / f"{paper_id}.md"
    shutil.copy2(markdown_path, final_markdown_path)

    copied_images = 0
    if source_images_dir is not None:
        copied_images = copy_non_formula_images(
            source_images_dir, paper_output_dir / "images",
            args.formula_image_max_bytes,
        )

    counts = parse_markdown_and_build(
        final_markdown_path, paper_id, paper_output_dir, args.context_window
    )
    record.update({"status": "processed", "images_copied": copied_images,
                   **counts})

    LOGGER.info(
        "Completed %s | images=%d | text=%d | formulas=%d | tables=%d | captions=%d",
        paper_id, copied_images, counts["text_chunks"], counts["formulas"],
        counts["tables"], counts["image_caption_entries"],
    )

    if not args.keep_temporary:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return record


def write_run_summary(output_dir: Path, records: list[dict]) -> None:
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "papers_seen": len(records),
        "status_counts": status_counts,
        "papers": records,
    }
    with (output_dir / "paper_processing_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        LOGGER.error("Input directory does not exist: %s", input_dir)
        return 2
    if args.retries < 1 or args.timeout < 1:
        LOGGER.error("--retries and --timeout must be positive.")
        return 2
    if shutil.which(args.mineru_command) is None and not Path(
        args.mineru_command
    ).is_file():
        LOGGER.error("MinerU executable not found: %s", args.mineru_command)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        LOGGER.warning("No PDF files found in %s", input_dir)
        write_run_summary(output_dir, [])
        return 0

    LOGGER.info("Found %d PDF papers.", len(pdf_files))
    environment = build_environment(args.device)
    records: list[dict] = []

    for position, pdf_path in enumerate(pdf_files, start=1):
        LOGGER.info("Paper %d/%d: %s", position, len(pdf_files), pdf_path.name)
        records.append(process_paper(pdf_path, output_dir, environment, args))

    write_run_summary(output_dir, records)
    failed = sum(
        1 for record in records
        if record.get("status") in {"mineru_failed", "markdown_missing"}
    )
    LOGGER.info("Run complete: %d papers, %d failures", len(records), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
