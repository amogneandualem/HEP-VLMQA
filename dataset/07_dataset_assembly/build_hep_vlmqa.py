#!/usr/bin/env python3
"""
Build the canonical seven-task HEP-VLMQA source dataset.

V1 — Scientific Figure Understanding
    recognition
    visual_detail
    interpretation
    reasoning

V2 — Scientific Provenance Reconstruction
    provenance_inputs
    provenance_method
    provenance_figure_generation

Supported V1 source files, in priority order:
    validated_qa.jsonl
    semantic_qa_agent_final_api.json

V2 source:
    auto_research_qa.json

This stage intentionally DOES NOT:
    - create train/validation/test splits;
    - construct grouped GM-LoRA records;
    - create model-specific training examples.

Those operations belong to Step 08.

Outputs
-------
hep_vlmqa.jsonl
dataset_manifest.json
dataset_validation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# 1. Canonical task definitions
# ============================================================

V1_SOURCE_FILES = (
    "validated_qa.jsonl",
    "semantic_qa_agent_final_api.json",
)

V2_SOURCE_FILE = "auto_research_qa.json"

V1_TYPES = (
    "recognition",
    "visual_detail",
    "interpretation",
    "reasoning",
)

V2_TYPES = (
    "provenance_inputs",
    "provenance_method",
    "provenance_figure_generation",
)

ALL_TYPES = set(V1_TYPES) | set(V2_TYPES)

TYPE_ALIASES = {
    "recognition": "recognition",

    "visual_detail": "visual_detail",
    "visual_details": "visual_detail",

    "interpretation": "interpretation",
    "reasoning": "reasoning",

    "inputs": "provenance_inputs",
    "input": "provenance_inputs",
    "provenance_inputs": "provenance_inputs",

    "method": "provenance_method",
    "provenance_method": "provenance_method",

    "figure_generation": "provenance_figure_generation",
    "provenance_figure_generation":
        "provenance_figure_generation",
}

IMAGE_KEYS = (
    "image",
    "image_path",
    "image_file",
    "image_name",
    "filename",
    "file_name",
    "figure",
    "figure_path",
)

UID_KEYS = (
    "uid",
    "image_id",
    "id",
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

VERSION_SUFFIX = re.compile(
    r"v\d+$",
    flags=re.IGNORECASE,
)


# ============================================================
# 2. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the canonical seven-task HEP-VLMQA dataset."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing per-paper HEP-VLMQA "
            "source folders."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for canonical dataset files.",
    )

    parser.add_argument(
        "--skip-image-hash",
        action="store_true",
        help=(
            "Skip file-content SHA256 hashing. "
            "Useful only for a quick dry audit."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when integrity problems or expected-count "
            "mismatches are detected."
        ),
    )

    parser.add_argument(
        "--expect-v1-qa",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--expect-v2-qa",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--expect-total-qa",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--expect-images",
        type=int,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# 3. Text helpers
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalized_text(value: Any) -> str:
    return clean_text(value).casefold()


def safe_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    return clean_text(value).casefold() in {
        "true",
        "1",
        "yes",
        "valid",
        "accepted",
    }


def normalize_qa_type(value: Any) -> str:
    value = clean_text(value).casefold()

    value = re.sub(
        r"[\s\-]+",
        "_",
        value,
    )

    return TYPE_ALIASES.get(
        value,
        "",
    )


def task_family(qa_type: str) -> str:
    if qa_type in V1_TYPES:
        return "v1"

    if qa_type in V2_TYPES:
        return "v2"

    return ""


# ============================================================
# 4. Historical V1 acceptance logic
# ============================================================

def is_valid_v1_qa(
    qa: Dict[str, Any],
) -> bool:
    """
    Preserve historical canonicalization behavior.

    If validation.valid exists and is false, remove the QA.
    If validation metadata is absent, retain the QA.
    """

    validation = qa.get(
        "validation"
    )

    if validation is None:
        return True

    if not isinstance(
        validation,
        dict,
    ):
        return True

    if "valid" not in validation:
        return True

    return safe_boolean(
        validation["valid"]
    )


# ============================================================
# 5. JSON loading
# ============================================================

def load_json_records(
    path: Path,
) -> List[Dict[str, Any]]:

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    # First try normal JSON.
    try:
        parsed = json.loads(text)

        if isinstance(
            parsed,
            dict,
        ):
            return [parsed]

        if isinstance(
            parsed,
            list,
        ):
            return [
                row
                for row in parsed
                if isinstance(row, dict)
            ]

    except json.JSONDecodeError:
        pass

    # Fall back to JSONL.
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in "
                    f"{path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    f"Expected JSON object in "
                    f"{path}:{line_number}"
                )

            records.append(row)

    return records


# ============================================================
# 6. Record helpers
# ============================================================

def first_value(
    record: Dict[str, Any],
    keys: Iterable[str],
) -> str:

    for key in keys:
        value = clean_text(
            record.get(
                key,
                "",
            )
        )

        if value:
            return value

    return ""


def record_image_keys(
    record: Dict[str, Any],
) -> Set[str]:

    keys = set()

    for field in IMAGE_KEYS:
        value = clean_text(
            record.get(
                field,
                "",
            )
        )

        if value:
            keys.add(value)
            keys.add(
                Path(value).name
            )

    for field in UID_KEYS:
        value = clean_text(
            record.get(
                field,
                "",
            )
        )

        if value:
            keys.add(value)

    return keys


def build_semantic_index(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:

    index = defaultdict(list)

    for record in records:
        for key in record_image_keys(
            record
        ):
            index[key].append(
                record
            )

    return index


def match_semantic_record(
    record: Dict[str, Any],
    semantic_index: Dict[
        str,
        List[Dict[str, Any]],
    ],
) -> Optional[Dict[str, Any]]:

    candidates = {}

    for key in record_image_keys(
        record
    ):
        for candidate in semantic_index.get(
            key,
            [],
        ):
            candidates[
                id(candidate)
            ] = candidate

    if len(candidates) == 1:
        return next(
            iter(
                candidates.values()
            )
        )

    return None


# ============================================================
# 7. Image resolution
# ============================================================

def list_images(
    paper_dir: Path,
) -> List[Path]:

    image_dir = (
        paper_dir
        / "images"
    )

    if not image_dir.is_dir():
        return []

    return sorted(
        path
        for path in image_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def resolve_image(
    data_root: Path,
    paper_dir: Path,
    record: Dict[str, Any],
    fallback_record:
        Optional[Dict[str, Any]] = None,
) -> Optional[Path]:

    values = []

    sources = [
        record,
    ]

    if fallback_record:
        sources.append(
            fallback_record
        )

    for source in sources:

        for field in IMAGE_KEYS:

            value = clean_text(
                source.get(
                    field,
                    "",
                )
            )

            if (
                value
                and value not in values
            ):
                values.append(
                    value
                )

    for value in values:

        supplied = Path(
            value
        )

        candidates = []

        if supplied.is_absolute():

            candidates.append(
                supplied
            )

        else:

            candidates.extend(
                [
                    paper_dir
                    / supplied,

                    paper_dir
                    / "images"
                    / supplied,

                    paper_dir
                    / "images"
                    / supplied.name,

                    data_root
                    / supplied,
                ]
            )

        for candidate in candidates:

            if candidate.is_file():
                return candidate.resolve()

    # Filename fallback.
    names = {
        Path(value).name
        for value in values
        if value
    }

    if names:

        matches = [
            path
            for path in list_images(
                paper_dir
            )
            if path.name in names
        ]

        if len(matches) == 1:
            return matches[0].resolve()

    # Historical fallback:
    # only one image exists in paper folder.
    images = list_images(
        paper_dir
    )

    if len(images) == 1:
        return images[0].resolve()

    return None


def relative_public_path(
    path: Path,
    data_root: Path,
) -> str:

    resolved = path.resolve()
    root = data_root.resolve()

    try:
        return resolved.relative_to(
            root
        ).as_posix()

    except ValueError as exc:

        raise ValueError(
            "Image lies outside --data-root: "
            f"{path}"
        ) from exc


# ============================================================
# 8. Hashing / stable IDs
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def stable_id(
    values: Iterable[str],
    length: int = 24,
) -> str:

    payload = "\n".join(
        values
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[:length]


# ============================================================
# 9. Metadata normalization
# ============================================================

def metadata_from_records(
    primary: Dict[str, Any],
    fallback:
        Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    fallback = fallback or {}

    def pick(
        *keys: str,
    ) -> Any:

        for source in (
            primary,
            fallback,
        ):
            for key in keys:

                value = source.get(
                    key
                )

                if value not in (
                    None,
                    "",
                    [],
                    {},
                ):
                    return value

        return ""

    physics_keywords = pick(
        "physics_keywords"
    )

    if not isinstance(
        physics_keywords,
        list,
    ):

        value = clean_text(
            physics_keywords
        )

        physics_keywords = (
            [value]
            if value
            else []
        )

    return {
        "domain":
            clean_text(
                pick(
                    "domain"
                )
            ),

        "image_type":
            clean_text(
                pick(
                    "image_type_v2",
                    "image_type",
                )
            ),

        "similarity_group":
            clean_text(
                pick(
                    "similarity_group"
                )
            ),

        "physics_keywords": [
            clean_text(value)
            for value
            in physics_keywords
            if clean_text(value)
        ],

        "caption":
            clean_text(
                pick(
                    "caption"
                )
            ),

        "clean_context":
            clean_text(
                pick(
                    "clean_context",
                    "context",
                    "description",
                )
            ),

        "semantic_concept":
            clean_text(
                pick(
                    "semantic_concept"
                )
            ),
    }


# ============================================================
# 10. Source discovery
# ============================================================

def discover_papers(
    data_root: Path,
) -> Dict[str, Path]:

    source_names = (
        *V1_SOURCE_FILES,
        V2_SOURCE_FILE,
    )

    candidate_dirs = set()

    for source_name in source_names:

        for path in data_root.rglob(
            source_name
        ):
            candidate_dirs.add(
                path.parent.resolve()
            )

    papers = {}
    canonical_versions = {}

    for paper_dir in sorted(
        candidate_dirs
    ):

        paper_id = paper_dir.name

        if (
            paper_id in papers
            and papers[paper_id]
            != paper_dir
        ):

            raise RuntimeError(
                "Duplicate paper directory "
                f"for {paper_id}:\n"
                f"{papers[paper_id]}\n"
                f"{paper_dir}"
            )

        papers[
            paper_id
        ] = paper_dir

        canonical_id = VERSION_SUFFIX.sub(
            "",
            paper_id,
        )

        previous = canonical_versions.get(
            canonical_id
        )

        if (
            previous
            and previous != paper_id
        ):

            raise RuntimeError(
                "Multiple versions of the same "
                "canonical paper detected: "
                f"{previous}, {paper_id}"
            )

        canonical_versions[
            canonical_id
        ] = paper_id

    return papers


def find_v1_source(
    paper_dir: Path,
) -> Optional[Path]:

    for filename in V1_SOURCE_FILES:

        path = (
            paper_dir
            / filename
        )

        if path.is_file():
            return path

    return None


# ============================================================
# 11. Dataset builder
# ============================================================

class DatasetBuilder:

    def __init__(
        self,
        data_root: Path,
        skip_image_hash: bool,
    ) -> None:

        self.data_root = (
            data_root.resolve()
        )

        self.skip_image_hash = (
            skip_image_hash
        )

        self.rows = []

        self.image_hash_cache = {}

        self.exact_seen = set()

        self.conflict_answers = {}

        self.qa_ids = set()

        self.qa_types = Counter()

        self.family_counts = Counter()

        self.stats = Counter()

        self.v2_types_per_image = defaultdict(
            set
        )

        self.problem_examples = []


    def problem(
        self,
        kind: str,
        paper_id: str,
        detail: str,
    ) -> None:

        self.stats[
            f"problem_{kind}"
        ] += 1

        if len(
            self.problem_examples
        ) < 200:

            self.problem_examples.append(
                {
                    "kind": kind,
                    "paper_id": paper_id,
                    "detail": detail,
                }
            )


    def image_hash(
        self,
        path: Path,
    ) -> str:

        resolved = path.resolve()

        if self.skip_image_hash:

            return stable_id(
                [
                    str(
                        resolved
                    )
                ],
                length=64,
            )

        if (
            resolved
            not in self.image_hash_cache
        ):

            self.image_hash_cache[
                resolved
            ] = sha256_file(
                resolved
            )

        return self.image_hash_cache[
            resolved
        ]


    def add_qa(
        self,
        paper_id: str,
        image_path: Path,
        source_uid: str,
        qa_type: str,
        question: Any,
        answer: Any,
        metadata: Dict[str, Any],
        source_file: str,
    ) -> None:

        question = clean_text(
            question
        )

        answer = clean_text(
            answer
        )

        if qa_type not in ALL_TYPES:

            self.problem(
                "invalid_qa_type",
                paper_id,
                repr(
                    qa_type
                ),
            )

            return

        if not question:

            self.problem(
                "empty_question",
                paper_id,
                source_file,
            )

            return

        if not answer:

            self.problem(
                "empty_answer",
                paper_id,
                source_file,
            )

            return

        image_sha256 = self.image_hash(
            image_path
        )

        image_relative = relative_public_path(
            image_path,
            self.data_root,
        )

        # Canonical record identity uses the source image location,
        # not file-content SHA256. Distinct source figures can contain
        # byte-identical image files and must remain distinct records.
        exact_key = (
            paper_id,
            image_relative,
            qa_type,
            normalized_text(
                question
            ),
            normalized_text(
                answer
            ),
        )

        if exact_key in self.exact_seen:

            self.stats[
                "exact_duplicates_removed"
            ] += 1

            return

        self.exact_seen.add(
            exact_key
        )

        # Same question/type for the same image but
        # conflicting answers is an integrity problem.
        conflict_key = (
            paper_id,
            image_relative,
            qa_type,
            normalized_text(
                question
            ),
        )

        answer_norm = normalized_text(
            answer
        )

        previous_answer = (
            self.conflict_answers.get(
                conflict_key
            )
        )

        if (
            previous_answer is not None
            and previous_answer
            != answer_norm
        ):

            self.problem(
                "conflicting_answer",
                paper_id,
                (
                    f"{qa_type}: "
                    f"{question[:160]}"
                ),
            )

            return

        self.conflict_answers[
            conflict_key
        ] = answer_norm

        qa_id = stable_id(
            [
                paper_id,
                image_relative,
                qa_type,
                normalized_text(
                    question
                ),
                answer_norm,
            ]
        )

        if qa_id in self.qa_ids:

            self.problem(
                "duplicate_qa_id",
                paper_id,
                qa_id,
            )

            return

        self.qa_ids.add(
            qa_id
        )

        family = task_family(
            qa_type
        )

        image_id = stable_id(
            [
                paper_id,
                image_relative,
            ]
        )

        row = {
            "qa_id":
                qa_id,

            "image_id":
                image_id,

            "paper_id":
                paper_id,

            "image":
                image_relative,

            "image_sha256":
                image_sha256,

            "task_family":
                family,

            "qa_type":
                qa_type,

            "question":
                question,

            "answer":
                answer,

            "source_uid":
                source_uid,

            "source_file":
                source_file,

            **metadata,
        }

        self.rows.append(
            row
        )

        self.qa_types[
            qa_type
        ] += 1

        self.family_counts[
            family
        ] += 1

        if family == "v2":

            self.v2_types_per_image[
                image_id
            ].add(
                qa_type
            )


    # --------------------------------------------------------
    # V1
    # --------------------------------------------------------

    def process_v1(
        self,
        paper_id: str,
        paper_dir: Path,
        source_path: Path,
        records: List[Dict[str, Any]],
    ) -> None:

        for record_index, record in enumerate(
            records
        ):

            if not isinstance(
                record,
                dict,
            ):

                self.problem(
                    "malformed_v1_record",
                    paper_id,
                    str(
                        record_index
                    ),
                )

                continue

            image_path = resolve_image(
                self.data_root,
                paper_dir,
                record,
            )

            if image_path is None:

                self.problem(
                    "missing_v1_image",
                    paper_id,
                    first_value(
                        record,
                        IMAGE_KEYS,
                    ),
                )

                continue

            qa_pairs = record.get(
                "qa_pairs",
                [],
            )

            if not isinstance(
                qa_pairs,
                list,
            ):

                self.problem(
                    "invalid_v1_qa_list",
                    paper_id,
                    str(
                        record_index
                    ),
                )

                continue

            metadata = metadata_from_records(
                record,
                None,
            )

            source_uid = first_value(
                record,
                UID_KEYS,
            )

            for qa_index, qa in enumerate(
                qa_pairs
            ):

                if not isinstance(
                    qa,
                    dict,
                ):

                    self.problem(
                        "malformed_v1_qa",
                        paper_id,
                        (
                            f"{record_index}:"
                            f"{qa_index}"
                        ),
                    )

                    continue

                if not is_valid_v1_qa(
                    qa
                ):

                    self.stats[
                        "invalid_v1_qa_filtered"
                    ] += 1

                    continue

                qa_type = normalize_qa_type(
                    qa.get(
                        "qa_type",
                        qa.get(
                            "question_type",
                            qa.get(
                                "type",
                                "",
                            ),
                        ),
                    )
                )

                if qa_type not in V1_TYPES:

                    self.problem(
                        "invalid_v1_type",
                        paper_id,
                        repr(
                            qa_type
                        ),
                    )

                    continue

                self.add_qa(
                    paper_id=paper_id,
                    image_path=image_path,
                    source_uid=source_uid,
                    qa_type=qa_type,
                    question=qa.get(
                        "question",
                        "",
                    ),
                    answer=qa.get(
                        "answer",
                        "",
                    ),
                    metadata=metadata,
                    source_file=source_path.name,
                )


    # --------------------------------------------------------
    # V2
    # --------------------------------------------------------

    def process_v2(
        self,
        paper_id: str,
        paper_dir: Path,
        records: List[Dict[str, Any]],
        semantic_records: List[Dict[str, Any]],
    ) -> None:

        semantic_index = build_semantic_index(
            semantic_records
        )

        for record_index, record in enumerate(
            records
        ):

            if not isinstance(
                record,
                dict,
            ):

                self.problem(
                    "malformed_v2_record",
                    paper_id,
                    str(
                        record_index
                    ),
                )

                continue

            semantic_match = match_semantic_record(
                record,
                semantic_index,
            )

            image_path = resolve_image(
                self.data_root,
                paper_dir,
                record,
                semantic_match,
            )

            if image_path is None:

                self.problem(
                    "missing_v2_image",
                    paper_id,
                    first_value(
                        record,
                        IMAGE_KEYS,
                    ),
                )

                continue

            qa_pairs = record.get(
                "provenance_qa_pairs",
                [],
            )

            if not isinstance(
                qa_pairs,
                list,
            ):

                self.problem(
                    "invalid_v2_qa_list",
                    paper_id,
                    str(
                        record_index
                    ),
                )

                continue

            metadata = metadata_from_records(
                record,
                semantic_match,
            )

            source_uid = (
                first_value(
                    record,
                    UID_KEYS,
                )
                or
                first_value(
                    semantic_match or {},
                    UID_KEYS,
                )
            )

            local_types = set()

            for qa_index, qa in enumerate(
                qa_pairs
            ):

                if not isinstance(
                    qa,
                    dict,
                ):

                    self.problem(
                        "malformed_v2_qa",
                        paper_id,
                        (
                            f"{record_index}:"
                            f"{qa_index}"
                        ),
                    )

                    continue

                qa_type = normalize_qa_type(
                    qa.get(
                        "qa_type",
                        qa.get(
                            "question_type",
                            qa.get(
                                "type",
                                "",
                            ),
                        ),
                    )
                )

                if qa_type not in V2_TYPES:

                    self.problem(
                        "invalid_v2_type",
                        paper_id,
                        repr(
                            qa_type
                        ),
                    )

                    continue

                if qa_type in local_types:

                    self.problem(
                        "duplicate_v2_type_within_image",
                        paper_id,
                        qa_type,
                    )

                    continue

                local_types.add(
                    qa_type
                )

                self.add_qa(
                    paper_id=paper_id,
                    image_path=image_path,
                    source_uid=source_uid,
                    qa_type=qa_type,
                    question=qa.get(
                        "question",
                        "",
                    ),
                    answer=qa.get(
                        "answer",
                        "",
                    ),
                    metadata=metadata,
                    source_file=V2_SOURCE_FILE,
                )

            if (
                local_types
                and local_types
                != set(V2_TYPES)
            ):

                self.problem(
                    "incomplete_v2_triplet",
                    paper_id,
                    ",".join(
                        sorted(
                            local_types
                        )
                    ),
                )


    # --------------------------------------------------------
    # Paper
    # --------------------------------------------------------

    def process_paper(
        self,
        paper_id: str,
        paper_dir: Path,
    ) -> None:

        v1_source = find_v1_source(
            paper_dir
        )

        v2_source = (
            paper_dir
            / V2_SOURCE_FILE
        )

        semantic_records = []

        if v1_source is not None:

            semantic_records = load_json_records(
                v1_source
            )

            self.stats[
                "papers_with_v1"
            ] += 1

            self.process_v1(
                paper_id,
                paper_dir,
                v1_source,
                semantic_records,
            )

        if v2_source.is_file():

            v2_records = load_json_records(
                v2_source
            )

            self.stats[
                "papers_with_v2"
            ] += 1

            self.process_v2(
                paper_id,
                paper_dir,
                v2_records,
                semantic_records,
            )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validation(
        self,
    ) -> Dict[str, Any]:

        papers = {
            row["paper_id"]
            for row in self.rows
        }

        images = {
            row["image_id"]
            for row in self.rows
        }

        incomplete_v2_images = sum(
            qa_types
            != set(V2_TYPES)
            for qa_types
            in self.v2_types_per_image.values()
        )

        problem_counts = {}

        for key, value in self.stats.items():

            if key.startswith(
                "problem_"
            ):

                problem_counts[
                    key[len("problem_"):]
                ] = value

        status = "PASS"

        if problem_counts:
            status = "FAIL"

        if incomplete_v2_images:
            status = "FAIL"

        missing_task_types = [
            qa_type
            for qa_type in (
                *V1_TYPES,
                *V2_TYPES,
            )
            if self.qa_types.get(
                qa_type,
                0,
            ) == 0
        ]

        if missing_task_types:
            status = "FAIL"

        return {
            "status":
                status,

            "papers":
                len(
                    papers
                ),

            "unique_images":
                len(
                    images
                ),

            "qa_pairs":
                len(
                    self.rows
                ),

            "task_families":
                dict(
                    sorted(
                        self.family_counts.items()
                    )
                ),

            "qa_types": {
                qa_type:
                    self.qa_types.get(
                        qa_type,
                        0,
                    )
                for qa_type in (
                    *V1_TYPES,
                    *V2_TYPES,
                )
            },

            "papers_with_v1":
                self.stats[
                    "papers_with_v1"
                ],

            "papers_with_v2":
                self.stats[
                    "papers_with_v2"
                ],

            "exact_duplicates_removed":
                self.stats[
                    "exact_duplicates_removed"
                ],

            "invalid_v1_qa_filtered":
                self.stats[
                    "invalid_v1_qa_filtered"
                ],

            "v2_images":
                len(
                    self.v2_types_per_image
                ),

            "incomplete_v2_images":
                incomplete_v2_images,

            "missing_task_types":
                missing_task_types,

            "problems":
                problem_counts,

            "problem_examples":
                self.problem_examples,
        }


# ============================================================
# 12. Atomic output
# ============================================================

def atomic_jsonl(
    path: Path,
    rows: Iterable[Dict[str, Any]],
) -> None:

    temporary = path.with_name(
        "." + path.name + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for row in rows:

            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )

    temporary.replace(
        path
    )


def atomic_json(
    path: Path,
    value: Any,
) -> None:

    temporary = path.with_name(
        "." + path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


# ============================================================
# 13. Expected-count validation
# ============================================================

def expectation_checks(
    validation: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:

    failures = []

    actual_v1 = int(
        validation[
            "task_families"
        ].get(
            "v1",
            0,
        )
    )

    actual_v2 = int(
        validation[
            "task_families"
        ].get(
            "v2",
            0,
        )
    )

    checks = (
        (
            "V1 QA",
            args.expect_v1_qa,
            actual_v1,
        ),
        (
            "V2 QA",
            args.expect_v2_qa,
            actual_v2,
        ),
        (
            "total QA",
            args.expect_total_qa,
            validation[
                "qa_pairs"
            ],
        ),
        (
            "unique images",
            args.expect_images,
            validation[
                "unique_images"
            ],
        ),
    )

    for label, expected, actual in checks:

        if (
            expected is not None
            and expected != actual
        ):

            failures.append(
                f"{label}: "
                f"expected {expected}, "
                f"found {actual}"
            )

    return failures


# ============================================================
# 14. Main
# ============================================================

def main() -> int:

    args = parse_args()

    data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not data_root.is_dir():

        print(
            "ERROR: data root does not exist:",
            data_root,
            file=sys.stderr,
        )

        return 2

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    papers = discover_papers(
        data_root
    )

    print(
        "Discovered paper folders:",
        len(
            papers
        ),
    )

    builder = DatasetBuilder(
        data_root=data_root,
        skip_image_hash=args.skip_image_hash,
    )

    for index, (
        paper_id,
        paper_dir,
    ) in enumerate(
        sorted(
            papers.items()
        ),
        start=1,
    ):

        builder.process_paper(
            paper_id,
            paper_dir,
        )

        if index % 500 == 0:

            print(
                f"Processed "
                f"{index}/{len(papers)} "
                f"papers | "
                f"QA={len(builder.rows)}"
            )

    # Stable deterministic order.
    builder.rows.sort(
        key=lambda row: (
            row["paper_id"],
            row["image_id"],
            row["task_family"],
            row["qa_type"],
            row["qa_id"],
        )
    )

    validation = builder.validation()

    expectation_failures = (
        expectation_checks(
            validation,
            args,
        )
    )

    validation[
        "expectation_failures"
    ] = expectation_failures

    if expectation_failures:

        validation[
            "status"
        ] = "FAIL"

    manifest = {
        "dataset":
            "HEP-VLMQA",

        "schema_version":
            "1.0",

        "source_files": {
            "v1_priority":
                list(
                    V1_SOURCE_FILES
                ),

            "v2":
                V2_SOURCE_FILE,
        },

        "task_families": {
            "scientific_figure_understanding":
                list(
                    V1_TYPES
                ),

            "scientific_provenance_reconstruction":
                list(
                    V2_TYPES
                ),
        },

        "records":
            len(
                builder.rows
            ),

        "papers":
            validation[
                "papers"
            ],

        "unique_images":
            validation[
                "unique_images"
            ],

        "qa_types":
            validation[
                "qa_types"
            ],

        "task_family_counts":
            validation[
                "task_families"
            ],

        "image_paths":
            "relative_to_data_root",

        "image_hash":
            (
                "sha256_file_content"
                if not args.skip_image_hash
                else
                "path_hash_for_quick_audit_only"
            ),
    }

    atomic_jsonl(
        output_dir
        / "hep_vlmqa.jsonl",
        builder.rows,
    )

    atomic_json(
        output_dir
        / "dataset_manifest.json",
        manifest,
    )

    atomic_json(
        output_dir
        / "dataset_validation.json",
        validation,
    )

    print()
    print(
        "===== HEP-VLMQA DATASET ASSEMBLY ====="
    )

    print(
        "Status:",
        validation[
            "status"
        ],
    )

    print(
        "Papers:",
        validation[
            "papers"
        ],
    )

    print(
        "Unique images:",
        validation[
            "unique_images"
        ],
    )

    print(
        "QA pairs:",
        validation[
            "qa_pairs"
        ],
    )

    print(
        "V1 QA:",
        validation[
            "task_families"
        ].get(
            "v1",
            0,
        ),
    )

    print(
        "V2 QA:",
        validation[
            "task_families"
        ].get(
            "v2",
            0,
        ),
    )

    print(
        "QA types:"
    )

    for qa_type in (
        *V1_TYPES,
        *V2_TYPES,
    ):

        print(
            f"  {qa_type}: "
            f"{validation['qa_types'][qa_type]}"
        )

    print(
        "Exact duplicates removed:",
        validation[
            "exact_duplicates_removed"
        ],
    )

    print(
        "Invalid V1 QA filtered:",
        validation[
            "invalid_v1_qa_filtered"
        ],
    )

    print(
        "Incomplete V2 images:",
        validation[
            "incomplete_v2_images"
        ],
    )

    if validation[
        "problems"
    ]:

        print(
            "Problems:"
        )

        for key, value in sorted(
            validation[
                "problems"
            ].items()
        ):

            print(
                f"  {key}: {value}"
            )

    if expectation_failures:

        print(
            "Expectation failures:"
        )

        for failure in expectation_failures:

            print(
                " ",
                failure,
            )

    print(
        "Output:",
        output_dir,
    )

    if (
        args.strict
        and validation[
            "status"
        ] != "PASS"
    ):

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
