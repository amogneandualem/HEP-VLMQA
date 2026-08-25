#!/usr/bin/env python3
"""
HEP-VLMQA semantic-concept generation.

This stage consumes the cleaned per-paper metadata produced by
`preprocess_metadata.py` and generates one grounded semantic concept for
each scientific figure.

Input per paper
---------------
clean_image_caption.jsonl

Required/expected fields:
    paper_id
    image
    caption
    clean_context
    domain
    image_type
    physics_keywords

Output per paper
----------------
semantic_concepts.jsonl

Each output record preserves the source metadata and adds:
    semantic_concept
    semantic_model

Example
-------
export HEPAI_API_KEY="..."

python generate_semantic_concepts.py \
    --root /path/to/per_paper_dataset \
    --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from hepai import HepAI


LOGGER = logging.getLogger("hep_vlmqa.semantic_generation")

DEFAULT_BASE_URL = "https://aiapi.ihep.ac.cn/apiv2"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_INPUT_NAME = "clean_image_caption.jsonl"
DEFAULT_OUTPUT_NAME = "semantic_concepts.jsonl"
DEFAULT_ERROR_NAME = "semantic_generation_errors.jsonl"

SEMANTIC_PROMPT = """
You are a scientific figure understanding agent.

Use IMAGE as primary evidence.
Use CAPTION as secondary evidence.
Use CONTEXT only as supporting evidence.

Write one strong semantic_concept paragraph in 2-4 scientific sentences.
It must explain what the figure represents and why it matters in the paper.

Rules:
- Do not hallucinate values, labels, particles, curves, colors, or conclusions.
- Do not claim proof, discovery, or confirmation unless explicitly stated.
- If the figure is a table, describe what quantities the table compares.
- If the figure is a plot, describe axes, plotted quantities, trends, and
  implications only when supported.
- If the figure is an apparatus or schematic, describe the experimental role
  of the components.
- If the figure is a Feynman diagram, describe the production or decay
  topology without unsupported claims.
- If image and caption appear mismatched, mention the mismatch carefully.
- Return strict JSON only.

Domain:
{domain}

Image type:
{image_type}

Physics keywords:
{physics_keywords}

Caption:
{caption}

Context:
{context}

Return:
{{
  "semantic_concept": "..."
}}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grounded semantic concepts for HEP-VLMQA figures."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing per-paper folders.",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help=f"Per-paper input JSONL filename (default: {DEFAULT_INPUT_NAME}).",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Per-paper output JSONL filename (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--error-name",
        default=DEFAULT_ERROR_NAME,
        help=f"Per-paper error JSONL filename (default: {DEFAULT_ERROR_NAME}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HEPAI_BASE_URL", DEFAULT_BASE_URL),
        help="HepAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SEM_MODEL", DEFAULT_MODEL),
        help=f"Semantic generation model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--paper",
        default=None,
        help="Process only one paper folder by name.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index into sorted paper folders.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end index into sorted paper folders.",
    )
    parser.add_argument(
        "--max-images-per-paper",
        type=int,
        default=None,
        help="Optional maximum number of figure records per paper.",
    )
    parser.add_argument(
        "--max-context-words",
        type=int,
        default=180,
        help="Maximum context words sent to the model (default: 180).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API retries per figure (default: 3).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Base retry delay in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between figures in seconds (default: 0.2).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature (default: 0.2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=550,
        help="Maximum generated tokens (default: 550).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing semantic output for selected papers.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def short_text(text: Any, max_words: int) -> str:
    return " ".join(str(text or "").split()[:max_words])


def make_uid(paper_dir: Path, record: dict[str, Any]) -> str:
    raw = (
        f"{paper_dir.name}|"
        f"{record.get('image', '')}|"
        f"{record.get('caption', '')}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{line_number}."
                )

            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    temporary.replace(path)


def strip_json_fences(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^```json\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^```\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_json_fences(text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(
            f"No JSON object found in model response: {cleaned[:300]}"
        )

    candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    parsed = json.loads(candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object.")

    return parsed


class SemanticGenerator:
    def __init__(
        self,
        args: argparse.Namespace,
        client: HepAI,
    ) -> None:
        self.args = args
        self.client = client

    def call_model(
        self,
        image_path: Path,
        prompt: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "Return strict valid JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path)
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        request = {
            "model": self.args.model,
            "messages": messages,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "stream": False,
        }

        try:
            response = self.client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.client.chat.completions.create(**request)

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Semantic model returned an empty response.")

        return str(content)

    def generate_semantic(
        self,
        image_path: Path,
        record: dict[str, Any],
    ) -> str:
        keywords = record.get("physics_keywords", [])
        if isinstance(keywords, list):
            keywords_text = ", ".join(map(str, keywords))
        else:
            keywords_text = str(keywords)

        prompt = SEMANTIC_PROMPT.format(
            domain=record.get("domain", ""),
            image_type=record.get("image_type", ""),
            physics_keywords=keywords_text,
            caption=record.get("caption", ""),
            context=short_text(
                record.get(
                    "clean_context",
                    record.get("description", ""),
                ),
                self.args.max_context_words,
            ),
        )

        last_error: Exception | None = None

        for attempt in range(1, self.args.max_retries + 1):
            try:
                response = self.call_model(
                    image_path=image_path,
                    prompt=prompt,
                )
                parsed = extract_json_object(response)
                semantic = str(
                    parsed.get("semantic_concept", "")
                ).strip()

                if not semantic:
                    raise ValueError("Empty semantic_concept.")

                return semantic

            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Semantic generation attempt %d/%d failed: %s",
                    attempt,
                    self.args.max_retries,
                    exc,
                )

                if attempt < self.args.max_retries:
                    time.sleep(
                        self.args.retry_backoff * attempt
                    )

        raise RuntimeError(
            f"Semantic generation failed: {last_error}"
        )

    def process_paper(self, paper_dir: Path) -> None:
        input_path = paper_dir / self.args.input_name

        if not input_path.is_file():
            LOGGER.warning(
                "Skipping %s: %s not found.",
                paper_dir.name,
                self.args.input_name,
            )
            return

        output_path = paper_dir / self.args.output_name
        error_path = paper_dir / self.args.error_name

        if output_path.exists() and not self.args.overwrite:
            LOGGER.info(
                "Skipping %s: %s already exists.",
                paper_dir.name,
                self.args.output_name,
            )
            return

        records = read_jsonl(input_path)

        if self.args.max_images_per_paper is not None:
            records = records[: self.args.max_images_per_paper]

        outputs: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for index, record in enumerate(records, start=1):
            image_relative = str(
                record.get("image", "")
            ).strip()

            LOGGER.info(
                "%s [%d/%d] %s",
                paper_dir.name,
                index,
                len(records),
                image_relative,
            )

            try:
                if not image_relative:
                    raise ValueError("Record has no image field.")

                image_path = paper_dir / image_relative
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found: {image_path}"
                    )

                semantic = self.generate_semantic(
                    image_path=image_path,
                    record=record,
                )

                output_record = dict(record)
                output_record["uid"] = make_uid(
                    paper_dir,
                    record,
                )
                output_record["semantic_concept"] = semantic
                output_record["semantic_model"] = self.args.model

                outputs.append(output_record)

            except Exception as exc:
                LOGGER.error(
                    "Failed %s/%s: %s",
                    paper_dir.name,
                    image_relative,
                    exc,
                )
                errors.append(
                    {
                        "uid": make_uid(paper_dir, record),
                        "paper_id": record.get(
                            "paper_id",
                            paper_dir.name,
                        ),
                        "image": image_relative,
                        "error": str(exc),
                        "semantic_model": self.args.model,
                    }
                )

            time.sleep(self.args.sleep)

        write_jsonl(output_path, outputs)

        if errors:
            write_jsonl(error_path, errors)
        elif error_path.exists() and self.args.overwrite:
            error_path.unlink()

        LOGGER.info(
            "%s complete | semantic=%d | errors=%d",
            paper_dir.name,
            len(outputs),
            len(errors),
        )


def select_papers(
    root: Path,
    paper: str | None,
    start: int,
    end: int | None,
) -> list[Path]:
    if paper:
        return [root / paper]

    folders = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    return folders[start:end]


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    api_key = os.environ.get("HEPAI_API_KEY")
    if not api_key:
        LOGGER.error(
            "HEPAI_API_KEY is not set. Export the API key before running."
        )
        return 2

    root = args.root.expanduser().resolve()

    if not root.is_dir():
        LOGGER.error("Dataset root does not exist: %s", root)
        return 2

    if args.max_context_words < 1:
        LOGGER.error("--max-context-words must be positive.")
        return 2

    if args.max_retries < 1:
        LOGGER.error("--max-retries must be at least 1.")
        return 2

    if args.max_tokens < 1:
        LOGGER.error("--max-tokens must be positive.")
        return 2

    client = HepAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    generator = SemanticGenerator(
        args=args,
        client=client,
    )

    paper_dirs = select_papers(
        root=root,
        paper=args.paper,
        start=args.start,
        end=args.end,
    )

    LOGGER.info(
        "Starting semantic generation | papers=%d | model=%s",
        len(paper_dirs),
        args.model,
    )

    for paper_dir in paper_dirs:
        if not paper_dir.is_dir():
            LOGGER.warning(
                "Paper directory not found: %s",
                paper_dir,
            )
            continue

        generator.process_paper(paper_dir)

    LOGGER.info("Semantic generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
