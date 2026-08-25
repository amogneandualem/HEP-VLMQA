#!/usr/bin/env python3
"""
HEP-VLMQA figure QA generation and validation.

For each figure, this stage takes:
    IMAGE + CAPTION + CONTEXT + SEMANTIC_CONCEPT

and performs:
    GPT-4o-mini QA generation
        -> rule-based screening
        -> GPT-4o validation
        -> one GPT-4o-mini repair attempt for rejected QA
        -> GPT-4o re-validation
        -> save accepted and rejected QA pairs

Exactly four scientific figure-understanding QA types are targeted:
    recognition
    visual_detail
    interpretation
    reasoning

Expected per-paper input
------------------------
By default, each paper folder should contain `semantic_concepts.jsonl`.
Each row should include at least:

{
  "paper_id": "1101.0398v1",
  "image": "images/abc.jpg",
  "caption": "...",
  "context": "...",
  "semantic_concept": "...",
  "domain": "...",
  "image_type": "...",
  "physics_keywords": [...]
}

The script also accepts a JSON list containing the same records.

Output
------
By default:
    validated_qa.jsonl
    qa_generation_errors.jsonl

Example
-------
export HEPAI_API_KEY="..."

python generate_and_validate_qa.py \
    --root /path/to/per_paper_dataset \
    --generator-model openai/gpt-4o-mini \
    --validator-model openai/gpt-4o

Security
--------
The API key is read only from the HEPAI_API_KEY environment variable.
No credentials are stored in this file.
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
from typing import Any, Iterable

from openai import OpenAI


LOGGER = logging.getLogger("hep_vlmqa.qa_generation")

DEFAULT_BASE_URL = "https://aiapi.ihep.ac.cn/apiv2"
DEFAULT_GENERATOR_MODEL = "openai/gpt-4o-mini"
DEFAULT_VALIDATOR_MODEL = "openai/gpt-4o"

DEFAULT_INPUT_NAME = "semantic_concepts.jsonl"
DEFAULT_OUTPUT_NAME = "validated_qa.jsonl"
DEFAULT_ERROR_NAME = "qa_generation_errors.jsonl"

QA_TYPES = (
    "recognition",
    "visual_detail",
    "interpretation",
    "reasoning",
)


QA_GENERATION_PROMPT = """
You are a scientific visual question-answer generation agent for HEP-VLMQA.

Use the following evidence hierarchy:
1. IMAGE is the primary evidence.
2. CAPTION is secondary evidence.
3. CONTEXT and SEMANTIC_CONCEPT are supporting evidence.

Generate exactly four QA pairs for this scientific figure, one for each type:
- recognition
- visual_detail
- interpretation
- reasoning

Definitions:
- recognition: identify the main scientific object, plot, process, detector,
  quantity, or figure purpose.
- visual_detail: ask about a specific visible scientific detail such as axes,
  markers, curves, regions, components, comparisons, or trends.
- interpretation: ask what a visible pattern, comparison, or relationship
  means scientifically.
- reasoning: require a grounded inference that combines visible evidence with
  caption/context, without relying on unsupported external knowledge.

Rules:
- Every question must be specific to this figure.
- Do not use yes/no questions.
- Do not ask generic questions such as "What does the figure show?"
- Do not duplicate questions.
- Do not invent values, labels, particles, colors, curves, regions,
  conclusions, or relationships.
- Do not rely on broad textbook knowledge.
- Do not claim proof, discovery, confirmation, or causality unless directly
  supported by the supplied evidence.
- Avoid color-only questions unless color has scientific meaning.
- Keep each answer to one concise grounded sentence.
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

Semantic concept:
{semantic_concept}

Return exactly:
{{
  "qa_pairs": [
    {{
      "qa_type": "recognition",
      "question": "...",
      "answer": "..."
    }},
    {{
      "qa_type": "visual_detail",
      "question": "...",
      "answer": "..."
    }},
    {{
      "qa_type": "interpretation",
      "question": "...",
      "answer": "..."
    }},
    {{
      "qa_type": "reasoning",
      "question": "...",
      "answer": "..."
    }}
  ]
}}
""".strip()


VALIDATION_PROMPT = """
You are the independent scientific QA validation agent for HEP-VLMQA.

Evaluate whether the proposed QA pair is scientifically grounded in the
supplied IMAGE, CAPTION, CONTEXT, and SEMANTIC_CONCEPT.

The IMAGE is the primary evidence.

Reject the QA pair if any of the following applies:
- the question or answer hallucinates content;
- the answer is contradicted by the image or caption;
- it refers to the wrong figure;
- the question is generic or not figure-specific;
- it duplicates another QA for the same figure;
- it relies primarily on broad textbook knowledge;
- it makes unsupported proof, discovery, confirmation, or causal claims;
- reasoning is speculative rather than grounded;
- it is a weak color-only question;
- the question can be answered without this figure and is not figure-specific;
- the answer is unsupported by the supplied visual/textual evidence.

Score:
4 = strongly grounded and publication-quality
3 = grounded and acceptable
2 = weak, ambiguous, or partly unsupported
1 = invalid or substantially hallucinated

A QA pair is accepted only when valid=true and score >= 3.

Return strict JSON only.

Caption:
{caption}

Context:
{context}

Semantic concept:
{semantic_concept}

QA type:
{qa_type}

Question:
{question}

Answer:
{answer}

Return:
{{
  "valid": true,
  "score": 4,
  "issues": [],
  "reason": "brief evidence-based reason",
  "repair_hint": ""
}}
""".strip()


REPAIR_PROMPT = """
You are a scientific visual QA repair agent for HEP-VLMQA.

A QA pair was rejected by an independent validator.

Generate one replacement QA pair for the SAME qa_type.

Use the following evidence hierarchy:
1. IMAGE is primary evidence.
2. CAPTION is secondary evidence.
3. CONTEXT and SEMANTIC_CONCEPT are supporting evidence.

Rules:
- Keep exactly the same qa_type.
- Make the question specific to this figure.
- Do not use yes/no or generic questions.
- Do not hallucinate values, labels, particles, colors, curves, regions,
  trends, or conclusions.
- Avoid broad textbook knowledge.
- Avoid unsupported proof, discovery, confirmation, or causal claims.
- Keep the answer to one concise grounded sentence.
- Address the validator's identified issue.
- Return strict JSON only.

Caption:
{caption}

Context:
{context}

Semantic concept:
{semantic_concept}

Target QA type:
{qa_type}

Rejected question:
{old_question}

Rejected answer:
{old_answer}

Validation issues:
{issues}

Validator reason:
{reason}

Repair hint:
{repair_hint}

Return:
{{
  "qa_type": "{qa_type}",
  "question": "...",
  "answer": "..."
}}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate HEP-VLMQA figure QA with GPT-4o-mini and validate "
            "with GPT-4o."
        )
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
        help=(
            "Per-paper semantic input filename "
            f"(default: {DEFAULT_INPUT_NAME})."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Per-paper validated output filename (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--error-name",
        default=DEFAULT_ERROR_NAME,
        help=f"Per-paper error filename (default: {DEFAULT_ERROR_NAME}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HEPAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--generator-model",
        default=os.environ.get("GEN_MODEL", DEFAULT_GENERATOR_MODEL),
        help=(
            "QA generation and repair model "
            f"(default: {DEFAULT_GENERATOR_MODEL})."
        ),
    )
    parser.add_argument(
        "--validator-model",
        default=os.environ.get("VAL_MODEL", DEFAULT_VALIDATOR_MODEL),
        help=f"Independent validation model (default: {DEFAULT_VALIDATOR_MODEL}).",
    )
    parser.add_argument(
        "--paper",
        default=None,
        help="Process one paper folder by name.",
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
        help="Maximum context words sent to the models (default: 180).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API retries per operation (default: 3).",
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
        "--overwrite",
        action="store_true",
        help="Overwrite an existing per-paper output file.",
    )
    parser.add_argument(
        "--disable-repair",
        action="store_true",
        help="Do not attempt one repair for rejected QA pairs.",
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


def normalize_text(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def make_uid(paper_dir: Path, record: dict[str, Any]) -> str:
    raw = (
        f"{paper_dir.name}|"
        f"{record.get('image', '')}|"
        f"{record.get('semantic_concept', '')}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    """
    Read either JSONL records or a JSON list.

    The file extension does not determine the parser, which makes this useful
    for converting older internal artifacts into the public modular pipeline.
    """
    if not path.is_file():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{line_number}."
                )
            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


class QAStage:
    def __init__(
        self,
        args: argparse.Namespace,
        client: OpenAI,
    ) -> None:
        self.args = args
        self.client = client

    def call_model(
        self,
        model: str,
        image_path: Path,
        prompt: str,
        max_tokens: int,
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
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
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
            raise ValueError(f"Empty response from {model}.")

        return str(content)

    def call_with_retry(
        self,
        model: str,
        image_path: Path,
        prompt: str,
        max_tokens: int,
        operation: str,
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self.args.max_retries + 1):
            try:
                return self.call_model(
                    model=model,
                    image_path=image_path,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "%s attempt %d/%d failed: %s",
                    operation,
                    attempt,
                    self.args.max_retries,
                    exc,
                )
                if attempt < self.args.max_retries:
                    time.sleep(self.args.retry_backoff * attempt)

        raise RuntimeError(
            f"{operation} failed after {self.args.max_retries} attempts: "
            f"{last_error}"
        )

    def generate_qa(
        self,
        image_path: Path,
        record: dict[str, Any],
    ) -> list[dict[str, str]]:
        keywords = record.get("physics_keywords", [])
        if isinstance(keywords, list):
            keywords_text = ", ".join(map(str, keywords))
        else:
            keywords_text = str(keywords)

        prompt = QA_GENERATION_PROMPT.format(
            domain=record.get("domain", ""),
            image_type=record.get("image_type", ""),
            physics_keywords=keywords_text,
            caption=record.get("caption", ""),
            context=short_text(
                record.get(
                    "context",
                    record.get(
                        "clean_context",
                        record.get("description", ""),
                    ),
                ),
                self.args.max_context_words,
            ),
            semantic_concept=record.get("semantic_concept", ""),
        )

        raw = self.call_with_retry(
            model=self.args.generator_model,
            image_path=image_path,
            prompt=prompt,
            max_tokens=900,
            operation="QA generation",
        )

        parsed = extract_json_object(raw)
        pairs = parsed.get("qa_pairs", [])

        if not isinstance(pairs, list):
            raise ValueError("Generator returned qa_pairs that is not a list.")

        indexed: dict[str, dict[str, str]] = {}

        for pair in pairs:
            if not isinstance(pair, dict):
                continue

            qa_type = str(pair.get("qa_type", "")).strip()
            question = str(pair.get("question", "")).strip()
            answer = str(pair.get("answer", "")).strip()

            if qa_type not in QA_TYPES:
                continue
            if not question or not answer:
                continue

            if qa_type not in indexed:
                indexed[qa_type] = {
                    "qa_type": qa_type,
                    "question": question,
                    "answer": answer,
                }

        missing = [qa_type for qa_type in QA_TYPES if qa_type not in indexed]
        if missing:
            raise ValueError(
                "Generator did not return exactly one usable QA for each type. "
                f"Missing: {', '.join(missing)}"
            )

        return [indexed[qa_type] for qa_type in QA_TYPES]

    @staticmethod
    def too_similar(
        question: str,
        previous_questions: list[str],
        threshold: float = 0.90,
    ) -> bool:
        words = set(normalize_text(question).split())
        if not words:
            return True

        for previous in previous_questions:
            old_words = set(normalize_text(previous).split())
            if not old_words:
                continue

            similarity = (
                len(words & old_words)
                / max(1, min(len(words), len(old_words)))
            )
            if similarity >= threshold:
                return True

        return False

    @classmethod
    def rule_filter(
        cls,
        qa: dict[str, Any],
        previous_questions: list[str],
    ) -> tuple[bool, str]:
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        normalized_question = normalize_text(question)

        if len(question.split()) < 5:
            return False, "question_too_short"

        if len(answer.split()) < 3:
            return False, "answer_too_short"

        if cls.too_similar(question, previous_questions):
            return False, "duplicate_question"

        generic_questions = (
            "what does the figure show",
            "what does the figure represent",
            "what is shown in the figure",
            "what is represented in the figure",
        )
        if any(text in normalized_question for text in generic_questions):
            return False, "generic_question"

        broad_reasoning = (
            "why is this important",
            "why is it important",
            "why are these results important",
        )
        if any(text in normalized_question for text in broad_reasoning):
            return False, "broad_reasoning_question"

        overclaims = (
            "definitively proves",
            "undeniably proves",
            "confirmed discovery",
            "establishes the existence",
        )
        if any(text in answer.lower() for text in overclaims):
            return False, "unsupported_overclaim"

        return True, "passed"

    def validate_qa(
        self,
        image_path: Path,
        record: dict[str, Any],
        qa: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = VALIDATION_PROMPT.format(
            caption=record.get("caption", ""),
            context=short_text(
                record.get(
                    "context",
                    record.get(
                        "clean_context",
                        record.get("description", ""),
                    ),
                ),
                self.args.max_context_words,
            ),
            semantic_concept=record.get("semantic_concept", ""),
            qa_type=qa.get("qa_type", ""),
            question=qa.get("question", ""),
            answer=qa.get("answer", ""),
        )

        raw = self.call_with_retry(
            model=self.args.validator_model,
            image_path=image_path,
            prompt=prompt,
            max_tokens=350,
            operation="QA validation",
        )

        parsed = extract_json_object(raw)

        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]

        try:
            score = int(parsed.get("score", 0))
        except (TypeError, ValueError):
            score = 0

        return {
            "valid": bool(parsed.get("valid", False)),
            "score": score,
            "issues": [str(issue) for issue in issues],
            "reason": str(parsed.get("reason", "")).strip(),
            "repair_hint": str(parsed.get("repair_hint", "")).strip(),
            "validator_model": self.args.validator_model,
        }

    @staticmethod
    def accepted(validation: dict[str, Any]) -> bool:
        return (
            bool(validation.get("valid", False))
            and int(validation.get("score", 0)) >= 3
        )

    def repair_qa(
        self,
        image_path: Path,
        record: dict[str, Any],
        failed_qa: dict[str, Any],
        validation: dict[str, Any],
    ) -> dict[str, str]:
        qa_type = str(failed_qa.get("qa_type", "")).strip()

        if qa_type not in QA_TYPES:
            raise ValueError(f"Cannot repair unknown QA type: {qa_type}")

        prompt = REPAIR_PROMPT.format(
            caption=record.get("caption", ""),
            context=short_text(
                record.get(
                    "context",
                    record.get(
                        "clean_context",
                        record.get("description", ""),
                    ),
                ),
                self.args.max_context_words,
            ),
            semantic_concept=record.get("semantic_concept", ""),
            qa_type=qa_type,
            old_question=failed_qa.get("question", ""),
            old_answer=failed_qa.get("answer", ""),
            issues="; ".join(
                map(str, validation.get("issues", []))
            ),
            reason=validation.get("reason", ""),
            repair_hint=validation.get("repair_hint", ""),
        )

        raw = self.call_with_retry(
            model=self.args.generator_model,
            image_path=image_path,
            prompt=prompt,
            max_tokens=450,
            operation="QA repair",
        )

        parsed = extract_json_object(raw)

        returned_type = str(parsed.get("qa_type", qa_type)).strip()
        if returned_type != qa_type:
            returned_type = qa_type

        question = str(parsed.get("question", "")).strip()
        answer = str(parsed.get("answer", "")).strip()

        if not question or not answer:
            raise ValueError("Repair returned an empty question or answer.")

        return {
            "qa_type": returned_type,
            "question": question,
            "answer": answer,
        }

    def process_figure(
        self,
        paper_dir: Path,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        image_relative = str(record.get("image", "")).strip()
        if not image_relative:
            raise ValueError("Record has no image field.")

        image_path = paper_dir / image_relative
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        semantic_concept = str(
            record.get("semantic_concept", "")
        ).strip()
        if not semantic_concept:
            raise ValueError("Record has no semantic_concept.")

        generated_pairs = self.generate_qa(
            image_path=image_path,
            record=record,
        )

        accepted_pairs: list[dict[str, Any]] = []
        rejected_pairs: list[dict[str, Any]] = []
        previous_questions: list[str] = []

        for original_qa in generated_pairs:
            qa = dict(original_qa)

            rule_ok, rule_issue = self.rule_filter(
                qa,
                previous_questions,
            )

            if rule_ok:
                validation = self.validate_qa(
                    image_path=image_path,
                    record=record,
                    qa=qa,
                )
            else:
                validation = {
                    "valid": False,
                    "score": 2,
                    "issues": [rule_issue],
                    "reason": rule_issue,
                    "repair_hint": (
                        "Generate a more specific and grounded QA pair."
                    ),
                    "validator_model": "rule_filter",
                }

            qa["validation"] = validation
            qa["generation_model"] = self.args.generator_model

            if self.accepted(validation):
                accepted_pairs.append(qa)
                previous_questions.append(qa["question"])
                continue

            if self.args.disable_repair:
                rejected_pairs.append(qa)
                continue

            try:
                repaired = self.repair_qa(
                    image_path=image_path,
                    record=record,
                    failed_qa=qa,
                    validation=validation,
                )

                repaired_rule_ok, repaired_rule_issue = self.rule_filter(
                    repaired,
                    previous_questions,
                )

                if repaired_rule_ok:
                    repaired_validation = self.validate_qa(
                        image_path=image_path,
                        record=record,
                        qa=repaired,
                    )
                else:
                    repaired_validation = {
                        "valid": False,
                        "score": 2,
                        "issues": [repaired_rule_issue],
                        "reason": repaired_rule_issue,
                        "repair_hint": "",
                        "validator_model": "rule_filter_after_repair",
                    }

                repaired["validation"] = repaired_validation
                repaired["generation_model"] = self.args.generator_model
                repaired["repaired"] = True
                repaired["repair_source"] = {
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "validation": validation,
                }

                if self.accepted(repaired_validation):
                    accepted_pairs.append(repaired)
                    previous_questions.append(repaired["question"])
                else:
                    rejected_pairs.append(qa)
                    rejected_pairs.append(repaired)

            except Exception as exc:
                qa["repair_error"] = str(exc)
                rejected_pairs.append(qa)

        return {
            "uid": make_uid(paper_dir, record),
            "paper_id": record.get("paper_id", paper_dir.name),
            "paper_folder": paper_dir.name,
            "image": image_relative,
            "domain": record.get("domain", ""),
            "image_type": record.get("image_type", ""),
            "similarity_group": record.get("similarity_group", ""),
            "physics_keywords": record.get("physics_keywords", []),
            "caption": record.get("caption", ""),
            "context": record.get(
                "context",
                record.get(
                    "clean_context",
                    record.get("description", ""),
                ),
            ),
            "semantic_concept": semantic_concept,
            "qa_pairs": accepted_pairs,
            "rejected_qa_pairs": rejected_pairs,
            "generator_model": self.args.generator_model,
            "validator_model": self.args.validator_model,
            "strategy": (
                "image_semantic_caption_context_generate_"
                "validate_repair_once"
            ),
        }

    def process_paper(self, paper_dir: Path) -> None:
        input_path = paper_dir / self.args.input_name
        if not input_path.is_file():
            LOGGER.warning(
                "Skipping %s: input not found: %s",
                paper_dir.name,
                input_path.name,
            )
            return

        output_path = paper_dir / self.args.output_name
        error_path = paper_dir / self.args.error_name

        if output_path.exists() and not self.args.overwrite:
            LOGGER.info(
                "Skipping %s: %s already exists.",
                paper_dir.name,
                output_path.name,
            )
            return

        source_records = read_records(input_path)

        if self.args.max_images_per_paper is not None:
            source_records = source_records[
                :self.args.max_images_per_paper
            ]

        outputs: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        LOGGER.info(
            "Processing %s: %d figure records.",
            paper_dir.name,
            len(source_records),
        )

        for index, record in enumerate(source_records, start=1):
            image = str(record.get("image", ""))

            LOGGER.info(
                "%s [%d/%d] %s",
                paper_dir.name,
                index,
                len(source_records),
                image,
            )

            try:
                outputs.append(
                    self.process_figure(
                        paper_dir=paper_dir,
                        record=record,
                    )
                )
            except Exception as exc:
                LOGGER.error(
                    "Failed %s/%s: %s",
                    paper_dir.name,
                    image,
                    exc,
                )
                errors.append(
                    {
                        "uid": make_uid(paper_dir, record),
                        "paper_id": record.get(
                            "paper_id",
                            paper_dir.name,
                        ),
                        "image": image,
                        "error": str(exc),
                    }
                )

            time.sleep(self.args.sleep)

        write_jsonl(output_path, outputs)

        if errors:
            write_jsonl(error_path, errors)
        elif error_path.exists() and self.args.overwrite:
            error_path.unlink()

        accepted_count = sum(
            len(row.get("qa_pairs", []))
            for row in outputs
        )
        rejected_count = sum(
            len(row.get("rejected_qa_pairs", []))
            for row in outputs
        )

        LOGGER.info(
            (
                "Completed %s | figures=%d | accepted_qa=%d | "
                "rejected_records=%d | errors=%d"
            ),
            paper_dir.name,
            len(outputs),
            accepted_count,
            rejected_count,
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

    if args.generator_model == args.validator_model:
        LOGGER.error(
            "Generator and validator must be different models."
        )
        return 2

    if args.max_retries < 1:
        LOGGER.error("--max-retries must be at least 1.")
        return 2

    if args.max_context_words < 1:
        LOGGER.error("--max-context-words must be positive.")
        return 2

    if (
        args.max_images_per_paper is not None
        and args.max_images_per_paper < 1
    ):
        LOGGER.error(
            "--max-images-per-paper must be positive when specified."
        )
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    stage = QAStage(args=args, client=client)

    paper_folders = select_papers(
        root=root,
        paper=args.paper,
        start=args.start,
        end=args.end,
    )

    LOGGER.info(
        (
            "Starting QA generation | papers=%d | generator=%s | "
            "validator=%s"
        ),
        len(paper_folders),
        args.generator_model,
        args.validator_model,
    )

    for paper_dir in paper_folders:
        if not paper_dir.is_dir():
            LOGGER.warning(
                "Paper directory not found: %s",
                paper_dir,
            )
            continue

        stage.process_paper(paper_dir)

    LOGGER.info("QA generation and validation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
