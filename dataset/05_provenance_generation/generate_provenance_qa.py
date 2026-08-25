#!/usr/bin/env python3
"""Duplicate-safe multimodal Auto Research provenance-QA generation.

For each unfinished eligible plot, send the actual image together with the
caption, semantic description, and nearby paper context, then generate exactly
three QA pairs:
1. provenance_inputs
2. provenance_method
3. provenance_figure_generation

Design:
- one GPT-4o-mini multimodal call per unfinished image;
- actual image is mandatory;
- existing complete final papers are preserved and skipped regardless of
  whether they were produced by V7 or V8;
- V7 partial papers are not reused, preventing mixed text-only/multimodal
  records within newly completed papers;
- deterministic local quality checks;
- V8 partial save after every successful image;
- paper-level locks with dead-process recovery;
- global stop marker on balance/authentication failures;
- no overwrite option.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


PIPELINE_VERSION = "auto_research_v8_1_multimodal"

WORK = Path(os.environ.get("AR_WORK", "./autoresearch_work"))
DATA_ROOT = Path(os.environ.get("AR_DATA_ROOT", "."))

BASE_URL = os.environ.get("HEPAI_BASE_URL", "https://aiapi.ihep.ac.cn/apiv2")
API_KEY = os.environ.get("HEPAI_API_KEY")
GEN_MODEL = os.environ.get("GEN_MODEL", "openai/gpt-4o-mini")
RUN_ID = os.environ.get("RUN_ID", "production_v8_multimodal")

INPUT_NAME = "semantic_qa_agent_final_api.json"
OUTPUT_NAME = "auto_research_qa.json"

MAX_CONTEXT_WORDS = int(os.environ.get("MAX_CONTEXT_WORDS", "350"))
MAX_CAPTION_WORDS = int(os.environ.get("MAX_CAPTION_WORDS", "180"))
MAX_SEMANTIC_WORDS = int(os.environ.get("MAX_SEMANTIC_WORDS", "40"))
MAX_QUESTION_WORDS = int(os.environ.get("MAX_QUESTION_WORDS", "35"))
MAX_ANSWER_WORDS = int(os.environ.get("MAX_ANSWER_WORDS", "60"))
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "420"))
MAX_API_RETRIES = int(os.environ.get("MAX_API_RETRIES", "3"))
MAX_CONTENT_RETRIES = int(os.environ.get("MAX_CONTENT_RETRIES", "4"))
API_RETRY_WAIT = float(os.environ.get("API_RETRY_WAIT", "2.0"))
REQUEST_SLEEP = float(os.environ.get("REQUEST_SLEEP", "0.15"))
STALE_LOCK_HOURS = float(os.environ.get("STALE_LOCK_HOURS", "6"))

SEND_IMAGE = os.environ.get("SEND_IMAGE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if not SEND_IMAGE:
    raise RuntimeError(
        "V8 multimodal requires SEND_IMAGE=1; text-only execution is disabled"
    )

PROGRESS_DIR = WORK / "progress"
PARTIAL_DIR = PROGRESS_DIR / "partials_v8"
LOCK_DIR = PROGRESS_DIR / "locks_v8"
LOG_DIR = WORK / "logs"
INVALID_FINAL_DIR = WORK / "backups" / "invalid_final_v8"
STOP_FILE = PROGRESS_DIR / "STOP_AUTO_RESEARCH_V8"

AUTO_RESEARCH_IMAGE_TYPES = {
    "line_plot",
    "scatter_or_correlation_plot",
    "histogram_or_distribution",
    "fit_or_spectrum_plot",
    "heatmap_or_matrix",
    "limit_or_contour_plot",
    "performance_plot",
}

QA_TYPES = [
    "provenance_inputs",
    "provenance_method",
    "provenance_figure_generation",
]

CODE_TO_QA_TYPE = {
    "i": "provenance_inputs",
    "inputs": "provenance_inputs",
    "provenance_inputs": "provenance_inputs",
    "m": "provenance_method",
    "method": "provenance_method",
    "provenance_method": "provenance_method",
    "f": "provenance_figure_generation",
    "figure_generation": "provenance_figure_generation",
    "provenance_figure_generation": "provenance_figure_generation",
}

TYPE_GUIDANCE = {
    "line_plot": (
        "scientific inputs or parameters; documented calculation or fit; "
        "how computed values were organized into the plotted comparison"
    ),
    "scatter_or_correlation_plot": (
        "sample and paired variables; documented matching, regression, or "
        "correlation method; how paired results were assembled"
    ),
    "histogram_or_distribution": (
        "event or sample inputs; documented selection, reconstruction, "
        "binning, or normalization; how counts or densities were assembled"
    ),
    "fit_or_spectrum_plot": (
        "measurements and model components; documented calibration, correction, "
        "likelihood, or fit; how data, predictions, and residuals were combined"
    ),
    "heatmap_or_matrix": (
        "samples, variables, or cells; documented aggregation, correlation, "
        "or normalization; how computed cell values were arranged"
    ),
    "limit_or_contour_plot": (
        "datasets, hypotheses, and parameter ranges; documented scan, "
        "likelihood, test statistic, or confidence procedure; how limits "
        "or contours were derived"
    ),
    "performance_plot": (
        "evaluation sample and metrics; documented threshold scan, "
        "cross-validation, efficiency, or error calculation; how performance "
        "values were compared"
    ),
}

VISUAL_ONLY_PATTERN = re.compile(
    r"\b(?:x|y)[ -]?axis\b|\baxes\b|\blegend\b|\bcolou?r(?:ed|s)?\b|"
    r"\bsolid line\b|\bdashed line\b|\bdotted line\b|\bline style\b|"
    r"\bmarkers?\b|\bsymbols?\b|\bupper panel\b|\blower panel\b|"
    r"\bleft panel\b|\bright panel\b|\bhow many\s+"
    r"(?:curves|lines|panels|points|bars)\b",
    re.I,
)

SPECULATIVE_PATTERN = re.compile(
    r"\b(?:likely|probably|presumably|apparently|may have|might have|"
    r"could have|seems to|appears to)\b",
    re.I,
)

GENERIC_METHOD_PATTERN = re.compile(
    r"\b(?:the data were analyzed|the data was analyzed|"
    r"the data were processed|the values were calculated|"
    r"the results were compared|the analysis focused on)\b",
    re.I,
)

GENERIC_FIGURE_PATTERN = re.compile(
    r"\b(?:shown|displayed|presented)\s+(?:in|as)\s+(?:the\s+)?"
    r"(?:figure|plot|graph)\b|"
    r"\bplotted as (?:a|an) (?:line|bar|scatter) (?:plot|graph)\b",
    re.I,
)

MALFORMED_MATH_PATTERN = re.compile(
    r"(?:\bO|\bexp|\blog|\bln|\bsin|\bcos|\btan)\([^)]*$"
    r"|[=+×*/^]\s*[.!?]?$",
    re.I,
)

for directory in [
    WORK,
    PROGRESS_DIR,
    PARTIAL_DIR,
    LOCK_DIR,
    LOG_DIR,
    INVALID_FINAL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


class FatalAPIError(RuntimeError):
    """Non-retryable API failure such as insufficient balance."""


class StopRequested(RuntimeError):
    """Shared stop marker was detected."""


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def truncate_words(value: Any, limit: int) -> str:
    words = clean_text(value).split()
    return " ".join(words[:limit])


def word_count(value: str) -> int:
    return len(clean_text(value).split())


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def append_log(path: Path, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def stable_shard(paper_id: str, num_shards: int) -> int:
    digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % num_shards


def strip_json_fence(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^```json\s*", "", value, flags=re.I)
    value = re.sub(r"^```\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_json(text: Any) -> dict:
    cleaned = strip_json_fence(text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("model output did not contain a JSON object")

    candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    data = json.loads(candidate)

    if not isinstance(data, dict):
        raise ValueError("extracted output was not a JSON object")

    return data


def resolve_image(record: dict, paper_dir: Path) -> Path | None:
    candidates: list[Path] = []

    stored = record.get("image_path")
    relative = record.get("image")

    if stored:
        candidates.append(Path(str(stored)))

    if relative:
        candidates.append(paper_dir / str(relative))
        candidates.append(
            paper_dir.parent
            / paper_dir.name
            / str(relative)
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def write_stop_marker(error: Exception) -> None:
    text = (
        f"time={now()}\n"
        f"host={socket.gethostname()}\n"
        f"pid={os.getpid()}\n"
        f"run_id={RUN_ID}\n"
        f"reason={clean_text(error)}\n"
    )
    atomic_text(STOP_FILE, text)


def check_stop_marker() -> None:
    if STOP_FILE.exists():
        raise StopRequested(
            f"stop marker exists: {STOP_FILE}. "
            "Remove it only after the API account is usable."
        )


def classify_api_error(error: Exception) -> str:
    text = str(error).lower()

    balance_terms = (
        "余额不足",
        "insufficient balance",
        "insufficient credit",
        "payment required",
        "payg",
        "billing",
        "recharge",
        "model_forbidden",
    )
    auth_terms = (
        "invalid api key",
        "unauthorized",
        "authentication",
        "error code: 401",
        "error code: 403",
    )

    if any(term in text for term in balance_terms):
        return "fatal_balance"
    if any(term in text for term in auth_terms):
        return "fatal_auth"
    if "error code: 400" in text or "bad request" in text:
        return "fatal_request"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "retryable"
    if "timeout" in text or "timed out" in text:
        return "retryable"
    if "connection" in text or "temporarily unavailable" in text:
        return "retryable"

    return "retryable"


_CLIENT: OpenAI | None = None


def get_client() -> OpenAI:
    global _CLIENT

    if _CLIENT is not None:
        return _CLIENT

    if not API_KEY:
        raise RuntimeError("HEPAI_API_KEY is not set")

    _CLIENT = OpenAI-compatible(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    return _CLIENT


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    result: dict[str, int] = {}

    for source_name, target_name in [
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ]:
        value = getattr(usage, source_name, None)
        if isinstance(value, int):
            result[target_name] = value

    return result


def call_generation_model(
    prompt: str,
    image_path: Path | None,
) -> tuple[dict, dict[str, int]]:
    check_stop_marker()

    if image_path is None:
        user_content: Any = prompt
    else:
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(image_path)},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a scientific multimodal provenance-QA generator. "
                "Return one strict JSON object only. "
                "No markdown, explanation, evidence, or analysis."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    last_error: Exception | None = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            parameters = {
                "model": GEN_MODEL,
                "messages": messages,
                "temperature": 0.05,
                "max_tokens": GEN_MAX_TOKENS,
                "stream": False,
            }

            try:
                response = get_client().chat.completions.create(
                    **parameters,
                    response_format={"type": "json_object"},
                )
            except Exception as format_error:
                lower = str(format_error).lower()
                if (
                    "response_format" not in lower
                    and "json_object" not in lower
                ):
                    raise
                response = get_client().chat.completions.create(
                    **parameters
                )

            content = response.choices[0].message.content
            return extract_json(content), response_usage(response)

        except Exception as error:
            last_error = error
            category = classify_api_error(error)

            if category.startswith("fatal_"):
                write_stop_marker(error)
                raise FatalAPIError(str(error)) from error

            if attempt < MAX_API_RETRIES:
                delay = API_RETRY_WAIT * (2 ** (attempt - 1))
                delay += random.uniform(0.0, 0.75)
                time.sleep(delay)

    raise RuntimeError(
        f"generation failed after {MAX_API_RETRIES} attempts: {last_error}"
    )


def generation_prompt(
    record: dict,
    correction_feedback: str = "",
) -> str:
    image_type = clean_text(record.get("image_type_v2"))
    caption = truncate_words(
        record.get("caption", ""),
        MAX_CAPTION_WORDS,
    )
    context = truncate_words(
        record.get("context", ""),
        MAX_CONTEXT_WORDS,
    )
    semantic_source = (
        record.get("semantic_concept")
        or record.get("semantic_description")
        or record.get("semantic_summary")
        or record.get("semantic")
        or ""
    )
    semantic = truncate_words(
        semantic_source,
        MAX_SEMANTIC_WORDS,
    )
    guidance = TYPE_GUIDANCE.get(
        image_type,
        "documented inputs, method, and scientific figure construction",
    )

    return f"""Generate exactly three grounded scientific provenance QA pairs for one HEP figure.

Required types, exactly once and in this order:
i = inputs, data, parameters, or setup used
m = documented analysis, calculation, reconstruction, fit, or evaluation method
f = documented transformation from processed scientific results into the figure

Evidence policy:
- Use the image to verify plot-specific variables, mathematical symbols, axis quantities, legends, panel assignments, and curve or category identity.
- Use the caption, semantic description, and nearby context for scientific provenance: datasets, parameters, equations, calculations, reconstruction, fitting, normalization, evaluation, and figure construction.
- Resolve conflicts conservatively: do not contradict clearly readable image labels, but do not infer an undocumented scientific method from visual appearance alone.

Rules:
- Ground every answer in the supplied image or text evidence.
- Do not invent standard practice, missing parameters, or undocumented operations.
- Ask about scientific provenance, not superficial appearance.
- Do not answer only with axes, legends, colors, markers, line styles, or panel positions.
- Do not use the terms x-axis, y-axis, axes, legend, color, marker, line style, upper panel, lower panel, left panel, or right panel in any question or answer.
- The figure-generation answer must explain how scientific results were computed, grouped, normalized, scanned, fitted, binned, or otherwise transformed before display; it must not merely say that values were plotted.
- Use visible labels only to disambiguate variables or categories.
- Preserve mathematical notation only when it is complete and unambiguous.
- If source notation is damaged, explain it in plain words instead of copying a broken expression.
- Never return unmatched parentheses, brackets, braces, truncated equations, or incomplete scientific notation.
- Each question must be specific and no longer than {MAX_QUESTION_WORDS} words.
- Each answer must be one concise sentence and no longer than {MAX_ANSWER_WORDS} words.
- Avoid speculative wording such as likely, probably, may have, or might have.
- Return no workflow, evidence, scores, reasons, validation, or commentary.

Return exactly:
{{"q":[["i","question","answer"],["m","question","answer"],["f","question","answer"]]}}

Figure type: {image_type}
Type focus: {guidance}
Domain: {clean_text(record.get("domain", ""))}
Caption: {caption}
Context: {context}
Semantic hint: {semantic}
{("Correction required from the previous rejected attempt: " + correction_feedback) if correction_feedback else ""}""".strip()


def normalize_question(value: Any) -> str:
    question = clean_text(value)
    if question and not question.endswith("?"):
        question += "?"
    return question


def normalize_answer(value: Any) -> str:
    answer = clean_text(value)
    if answer and answer[-1] not in ".!?":
        answer += "."
    return answer


def delimiters_balanced(value: str) -> bool:
    pairs = {
        ")": "(",
        "]": "[",
        "}": "{",
    }
    stack: list[str] = []

    for character in value:
        if character in "([{":
            stack.append(character)
        elif character in ")]}":
            if (
                not stack
                or stack[-1] != pairs[character]
            ):
                return False
            stack.pop()

    return not stack


def parse_qa_item(
    item: Any,
) -> tuple[str, str, str] | None:
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        code, question, answer = item[0], item[1], item[2]
    elif isinstance(item, dict):
        code = item.get(
            "qa_type",
            item.get("type", item.get("code", "")),
        )
        question = item.get("question", "")
        answer = item.get("answer", "")
    else:
        return None

    qa_type = CODE_TO_QA_TYPE.get(
        clean_text(code).lower()
    )

    if not qa_type:
        return None

    return (
        qa_type,
        normalize_question(question),
        normalize_answer(answer),
    )


def validate_qa_pairs(qas: list[dict]) -> list[str]:
    errors: list[str] = []

    if len(qas) != 3:
        errors.append(
            f"expected 3 QA pairs, found {len(qas)}"
        )
        return errors

    actual_types = [
        qa.get("qa_type")
        for qa in qas
    ]

    if actual_types != QA_TYPES:
        errors.append(
            f"QA types/order mismatch: {actual_types}"
        )

    normalized_questions: list[str] = []

    for qa in qas:
        qa_type = clean_text(qa.get("qa_type"))
        question = clean_text(qa.get("question"))
        answer = clean_text(qa.get("answer"))
        combined = f"{question} {answer}"

        if not question:
            errors.append(f"{qa_type}: empty question")
        if not answer:
            errors.append(f"{qa_type}: empty answer")

        if word_count(question) > MAX_QUESTION_WORDS:
            errors.append(
                f"{qa_type}: question has "
                f"{word_count(question)} words; "
                f"limit is {MAX_QUESTION_WORDS}"
            )

        if word_count(answer) > MAX_ANSWER_WORDS:
            errors.append(
                f"{qa_type}: answer has "
                f"{word_count(answer)} words; "
                f"limit is {MAX_ANSWER_WORDS}"
            )

        if VISUAL_ONLY_PATTERN.search(question):
            errors.append(
                f"{qa_type}: visual-only question wording"
            )

        if VISUAL_ONLY_PATTERN.search(answer):
            errors.append(
                f"{qa_type}: visual-only answer wording"
            )

        if SPECULATIVE_PATTERN.search(combined):
            errors.append(
                f"{qa_type}: speculative wording"
            )

        if not delimiters_balanced(combined):
            errors.append(
                f"{qa_type}: unmatched delimiter"
            )

        if MALFORMED_MATH_PATTERN.search(answer):
            errors.append(
                f"{qa_type}: malformed mathematical notation"
            )

        if (
            qa_type == "provenance_method"
            and GENERIC_METHOD_PATTERN.search(answer)
        ):
            errors.append(
                f"{qa_type}: generic or circular method answer"
            )

        if (
            qa_type == "provenance_figure_generation"
            and GENERIC_FIGURE_PATTERN.search(answer)
        ):
            errors.append(
                f"{qa_type}: generic figure-generation answer"
            )

        normalized_questions.append(
            re.sub(
                r"\W+",
                " ",
                question.lower(),
            ).strip()
        )

    if len(set(normalized_questions)) != len(
        normalized_questions
    ):
        errors.append("duplicate questions")

    return errors


def normalize_generation(output: dict) -> list[dict]:
    raw_items = output.get("q")

    if raw_items is None:
        raw_items = output.get("provenance_qa_pairs")

    if not isinstance(raw_items, list):
        raise ValueError(
            "model output is missing list field 'q'"
        )

    qa_map: dict[str, dict] = {}

    for item in raw_items:
        parsed = parse_qa_item(item)

        if parsed is None:
            continue

        qa_type, question, answer = parsed

        if qa_type in qa_map:
            continue

        qa_map[qa_type] = {
            "qa_type": qa_type,
            "question": question,
            "answer": answer,
        }

    qas = [
        qa_map[qa_type]
        for qa_type in QA_TYPES
        if qa_type in qa_map
    ]

    errors = validate_qa_pairs(qas)

    if errors:
        raise ValueError("; ".join(errors))

    return qas


def v8_record_complete(record: dict) -> bool:
    """Validate a V8 partial or newly generated V8 record."""

    if record.get("pipeline_version") != PIPELINE_VERSION:
        return False

    qas = record.get("provenance_qa_pairs")

    if not isinstance(qas, list) or len(qas) != 3:
        return False

    actual_types = [
        qa.get("qa_type")
        if isinstance(qa, dict)
        else None
        for qa in qas
    ]

    if actual_types != QA_TYPES:
        return False

    normalized_questions: list[str] = []

    for qa in qas:
        if not isinstance(qa, dict):
            return False

        question = clean_text(qa.get("question"))
        answer = clean_text(qa.get("answer"))

        if not question or not answer:
            return False

        normalized_questions.append(
            re.sub(
                r"\W+",
                " ",
                question.lower(),
            ).strip()
        )

    return len(set(normalized_questions)) == 3


def preserved_final_record_complete(record: dict) -> bool:
    """Recognize a complete previously paid final record from any pipeline.

    This intentionally performs structural validation only. Existing complete
    papers are preserved and never regenerated during the V8 switch.
    """

    qas = record.get("provenance_qa_pairs")

    if not isinstance(qas, list) or len(qas) != 3:
        return False

    actual_types = [
        qa.get("qa_type") if isinstance(qa, dict) else None
        for qa in qas
    ]

    if actual_types != QA_TYPES:
        return False

    normalized_questions: list[str] = []

    for qa in qas:
        if not isinstance(qa, dict):
            return False

        question = clean_text(qa.get("question"))
        answer = clean_text(qa.get("answer"))

        if not question or not answer:
            return False

        normalized_questions.append(
            re.sub(r"\W+", " ", question.lower()).strip()
        )

    return len(set(normalized_questions)) == 3


def minimal_record(
    paper_dir: Path,
    source: dict,
    qas: list[dict],
) -> dict:
    return {
        "uid": source.get("uid"),
        "paper_id": source.get(
            "paper_id",
            paper_dir.name,
        ),
        "image": source.get("image", ""),
        "image_type_v2": source.get(
            "image_type_v2",
            "",
        ),
        "domain": source.get("domain", ""),
        "provenance_qa_pairs": qas,
        "pipeline_version": PIPELINE_VERSION,
        "generator_model": GEN_MODEL,
        "input_mode": "image+caption+semantic+context",
        "image_sent": True,
        "generated_at": now(),
    }


def process_image(
    paper_dir: Path,
    record: dict,
) -> tuple[dict, dict[str, int]]:
    image_path: Path | None = None

    if SEND_IMAGE:
        image_path = resolve_image(record, paper_dir)

        if image_path is None:
            raise FileNotFoundError(
                str(record.get("image", ""))
            )

    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    correction_feedback = ""
    last_error: Exception | None = None

    for content_attempt in range(1, MAX_CONTENT_RETRIES + 1):
        raw, usage = call_generation_model(
            generation_prompt(
                record,
                correction_feedback=correction_feedback,
            ),
            image_path,
        )

        for key in total_usage:
            total_usage[key] += int(usage.get(key, 0) or 0)

        try:
            qas = normalize_generation(raw)
            return minimal_record(
                paper_dir,
                record,
                qas,
            ), total_usage

        except ValueError as error:
            last_error = error
            correction_feedback = (
                f"Attempt {content_attempt} was rejected: {clean_text(error)}. "
                "Regenerate all three QA pairs from scratch. Preserve the required "
                "order and scientific grounding, and remove every rejected wording "
                "pattern rather than paraphrasing it visually."
            )

    raise ValueError(
        "content validation failed after "
        f"{MAX_CONTENT_RETRIES} attempts: {last_error}"
    )


def read_lock_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}

    try:
        for line in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()

    except Exception:
        pass

    return metadata


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def acquire_lock(path: Path) -> None:
    if path.exists():
        metadata = read_lock_metadata(path)

        try:
            age_seconds = (
                time.time()
                - path.stat().st_mtime
            )
        except FileNotFoundError:
            age_seconds = 0.0

        lock_host = metadata.get("host", "")

        try:
            lock_pid = int(
                metadata.get("pid", "-1")
            )
        except ValueError:
            lock_pid = -1

        same_host_dead = (
            lock_host == socket.gethostname()
            and lock_pid > 0
            and not process_is_alive(lock_pid)
        )

        expired = (
            age_seconds
            > STALE_LOCK_HOURS * 3600
        )

        if same_host_dead or expired:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY,
    )

    content = (
        f"time={now()}\n"
        f"host={socket.gethostname()}\n"
        f"pid={os.getpid()}\n"
        f"run_id={RUN_ID}\n"
        f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}\n"
        f"slurm_array_task_id={os.environ.get('SLURM_ARRAY_TASK_ID', '')}\n"
    )

    os.write(
        descriptor,
        content.encode("utf-8"),
    )
    os.close(descriptor)


def release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def expected_eligible_records(
    paper_dir: Path,
) -> list[dict]:
    records = load_json_list(
        paper_dir / INPUT_NAME
    )

    return [
        record
        for record in records
        if (
            record.get("uid")
            and record.get("image_type_v2")
            in AUTO_RESEARCH_IMAGE_TYPES
        )
    ]


def valid_partial_records(
    partial_path: Path,
    expected_uids: set[str],
) -> dict[str, dict]:
    return {
        str(record.get("uid")): record
        for record in load_json_list(partial_path)
        if (
            str(record.get("uid"))
            in expected_uids
            and v8_record_complete(record)
        )
    }


def final_output_valid(
    path: Path,
    expected_uids: set[str],
) -> bool:
    records = load_json_list(path)

    if not expected_uids:
        return records == []

    if len(records) != len(expected_uids):
        return False

    output_uids = {
        str(record.get("uid"))
        for record in records
        if record.get("uid")
    }

    return (
        output_uids == expected_uids
        and all(
            preserved_final_record_complete(record)
            for record in records
        )
    )


def quarantine_invalid_final(
    final_path: Path,
    paper_id: str,
) -> None:
    if not final_path.exists():
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = (
        INVALID_FINAL_DIR
        / paper_id
        / f"{timestamp}_{OUTPUT_NAME}"
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.replace(final_path, target)


def paper_progress(
    paper_dir: Path,
) -> dict[str, int | bool]:
    eligible = expected_eligible_records(paper_dir)

    expected_uids = {
        str(record.get("uid"))
        for record in eligible
    }

    final_path = paper_dir / OUTPUT_NAME
    partial_path = (
        PARTIAL_DIR
        / f"{PIPELINE_VERSION}_{paper_dir.name}.json"
    )

    final_complete = final_output_valid(
        final_path,
        expected_uids,
    )

    partial_records: dict[str, dict] = {}

    if not final_complete:
        partial_records = valid_partial_records(
            partial_path,
            expected_uids,
        )

    completed_images = (
        len(eligible)
        if final_complete
        else len(partial_records)
    )

    return {
        "eligible_images": len(eligible),
        "final_complete": final_complete,
        "partial_images": len(partial_records),
        "completed_images": completed_images,
        "remaining_images": (
            len(eligible)
            - completed_images
        ),
    }


def process_paper(
    paper_dir: Path,
    log_file: Path,
) -> dict:
    eligible = expected_eligible_records(paper_dir)

    expected_uids = {
        str(record.get("uid"))
        for record in eligible
    }

    final_path = paper_dir / OUTPUT_NAME
    partial_path = (
        PARTIAL_DIR
        / f"{PIPELINE_VERSION}_{paper_dir.name}.json"
    )
    lock_path = (
        LOCK_DIR
        / f"{PIPELINE_VERSION}_{paper_dir.name}.lock"
    )

    if final_output_valid(
        final_path,
        expected_uids,
    ):
        return {
            "paper_id": paper_dir.name,
            "status": "already_complete",
            "images": len(eligible),
        }

    try:
        acquire_lock(lock_path)
    except FileExistsError:
        return {
            "paper_id": paper_dir.name,
            "status": "locked",
            "images": len(eligible),
        }

    try:
        check_stop_marker()

        if final_path.exists():
            quarantine_invalid_final(
                final_path,
                paper_dir.name,
            )

        if not eligible:
            atomic_json(final_path, [])
            return {
                "paper_id": paper_dir.name,
                "status": "complete_empty",
                "images": 0,
            }

        outputs_by_uid = valid_partial_records(
            partial_path,
            expected_uids,
        )

        failures = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        for index, record in enumerate(
            eligible,
            start=1,
        ):
            check_stop_marker()

            uid = str(record.get("uid"))

            if uid in outputs_by_uid:
                continue

            append_log(
                log_file,
                f"{paper_dir.name} "
                f"[{index}/{len(eligible)}] "
                f"{record.get('image_type_v2')}",
            )

            started = time.time()

            try:
                output, usage = process_image(
                    paper_dir,
                    record,
                )

                outputs_by_uid[uid] = output

                ordered_partial = [
                    outputs_by_uid[
                        str(source.get("uid"))
                    ]
                    for source in eligible
                    if (
                        str(source.get("uid"))
                        in outputs_by_uid
                    )
                ]

                atomic_json(
                    partial_path,
                    ordered_partial,
                )

                prompt_tokens += usage.get(
                    "prompt_tokens",
                    0,
                )
                completion_tokens += usage.get(
                    "completion_tokens",
                    0,
                )
                total_tokens += usage.get(
                    "total_tokens",
                    0,
                )

                usage_text = ""

                if usage:
                    usage_text = (
                        f" tokens="
                        f"{usage.get('prompt_tokens', 0)}/"
                        f"{usage.get('completion_tokens', 0)}/"
                        f"{usage.get('total_tokens', 0)}"
                    )

                append_log(
                    log_file,
                    f"SAVED uid={uid} qa=3 "
                    f"elapsed="
                    f"{time.time() - started:.1f}s"
                    f"{usage_text}",
                )

            except (
                FatalAPIError,
                StopRequested,
            ):
                raise

            except Exception as error:
                failures += 1

                append_log(
                    log_file,
                    f"FAILED paper={paper_dir.name} "
                    f"uid={uid} "
                    f"image={record.get('image', '')} "
                    f"error={clean_text(error)}",
                )

            time.sleep(REQUEST_SLEEP)

        completed = {
            uid: record
            for uid, record in outputs_by_uid.items()
            if v8_record_complete(record)
        }

        if set(completed) != expected_uids:
            return {
                "paper_id": paper_dir.name,
                "status": "incomplete",
                "images": len(eligible),
                "completed": len(completed),
                "failures": failures,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        final_records = [
            completed[str(source.get("uid"))]
            for source in eligible
        ]

        atomic_json(
            final_path,
            final_records,
        )

        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass

        return {
            "paper_id": paper_dir.name,
            "status": "complete",
            "images": len(final_records),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    finally:
        release_lock(lock_path)


def select_papers(
    requested: list[str] | None,
    num_shards: int,
    shard_id: int,
    max_papers: int,
) -> list[Path]:
    if requested:
        candidates = [
            DATA_ROOT / paper_id
            for paper_id in requested
        ]
    else:
        candidates = sorted(
            path
            for path in DATA_ROOT.iterdir()
            if (
                path.is_dir()
                and (path / INPUT_NAME).exists()
            )
        )

    selected = [
        path
        for path in candidates
        if (
            path.exists()
            and stable_shard(
                path.name,
                num_shards,
            )
            == shard_id
        )
    ]

    if max_papers > 0:
        return selected[:max_papers]

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--paper",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit(
            "--num-shards must be >= 1"
        )

    if not (
        0 <= args.shard_id < args.num_shards
    ):
        raise SystemExit(
            "--shard-id must satisfy "
            "0 <= shard-id < num-shards"
        )

    if not args.dry_run and not API_KEY:
        raise SystemExit(
            "HEPAI_API_KEY is not set"
        )

    if not args.dry_run:
        check_stop_marker()

    paper_dirs = select_papers(
        args.paper,
        args.num_shards,
        args.shard_id,
        args.max_papers,
    )

    selected_images = 0
    complete_papers = 0
    complete_images = 0
    partial_papers = 0
    partial_images = 0
    remaining_images = 0
    remaining_paper_ids: list[str] = []

    for paper_dir in paper_dirs:
        progress = paper_progress(paper_dir)

        selected_images += int(
            progress["eligible_images"]
        )
        complete_images += int(
            progress["completed_images"]
        )
        partial_images += int(
            progress["partial_images"]
        )
        remaining_images += int(
            progress["remaining_images"]
        )

        if bool(progress["final_complete"]):
            complete_papers += 1
        elif int(progress["partial_images"]) > 0:
            partial_papers += 1

        if int(progress["remaining_images"]) > 0:
            remaining_paper_ids.append(
                paper_dir.name
            )

    log_file = (
        LOG_DIR
        / f"{RUN_ID}_shard{args.shard_id}.log"
    )

    print("=" * 80)
    print("AUTO RESEARCH V8 MULTIMODAL GENERATION")
    print("=" * 80)
    print("Pipeline:", PIPELINE_VERSION)
    print("Run ID:", RUN_ID)
    print(
        "Shard:",
        args.shard_id,
        "/",
        args.num_shards,
    )
    print("Papers selected:", len(paper_dirs))
    print("Complete papers:", complete_papers)
    print("Partial papers:", partial_papers)
    print("Eligible images:", selected_images)
    print("Complete/saved images:", complete_images)
    print("Reusable partial images:", partial_images)
    print("Remaining images:", remaining_images)
    print("Estimated API calls:", remaining_images)
    print("Generator:", GEN_MODEL)
    print("Send image:", SEND_IMAGE)
    print(
        "Context word limit:",
        MAX_CONTEXT_WORDS,
    )
    print(
        "Generation token ceiling:",
        GEN_MAX_TOKENS,
    )
    print("=" * 80)

    if args.dry_run:
        for paper_id in remaining_paper_ids[:20]:
            print(paper_id)
        return

    complete = 0
    incomplete = 0
    skipped = 0
    locked = 0
    new_final_images = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:
        for paper_dir in paper_dirs:
            result = process_paper(
                paper_dir,
                log_file,
            )

            status = result.get("status")

            if status in {
                "complete",
                "complete_empty",
            }:
                complete += 1
                new_final_images += int(
                    result.get("images", 0)
                )
            elif status == "already_complete":
                skipped += 1
            elif status == "locked":
                locked += 1
            else:
                incomplete += 1

            prompt_tokens += int(
                result.get("prompt_tokens", 0)
            )
            completion_tokens += int(
                result.get(
                    "completion_tokens",
                    0,
                )
            )
            total_tokens += int(
                result.get("total_tokens", 0)
            )

    except (
        FatalAPIError,
        StopRequested,
    ) as error:
        append_log(
            log_file,
            f"SHARD STOPPED: {clean_text(error)}",
        )

        print(
            f"STOP FILE: {STOP_FILE}",
            file=sys.stderr,
        )

        raise SystemExit(75) from error

    append_log(
        log_file,
        f"SHARD COMPLETE "
        f"complete={complete} "
        f"skipped={skipped} "
        f"incomplete={incomplete} "
        f"locked={locked} "
        f"new_final_images={new_final_images} "
        f"prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} "
        f"total_tokens={total_tokens}",
    )


if __name__ == "__main__":
    main()
