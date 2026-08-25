#!/usr/bin/env python3
"""
Prepare the locked HEP-VLMQA benchmark splits.

This script does NOT generate a new random split.

It uses the authoritative paper-ID manifests:
    train.txt
    validation.txt
    test.txt

Expected locked source split:
    train:      14,240 papers
    validation:    100 papers
    test:          100 papers
    total:       14,440 papers

Some source-corpus papers may legitimately contain zero accepted QA.
Such papers remain part of the locked paper manifest but naturally do
not appear in the QA-level JSONL output.

Input
-----
hep_vlmqa.jsonl

Outputs
-------
train.jsonl
validation.jsonl
test.jsonl

manifests/
    train.txt
    validation.txt
    test.txt

split_manifest.json
split_validation.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


QA_TYPES = (
    "recognition",
    "visual_detail",
    "interpretation",
    "reasoning",
    "provenance_inputs",
    "provenance_method",
    "provenance_figure_generation",
)

SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)

EXPECTED_SOURCE_PAPERS = {
    "train": 14240,
    "validation": 100,
    "test": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply and validate the locked HEP-VLMQA "
            "paper-level benchmark splits."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical Step-07 hep_vlmqa.jsonl.",
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing train.txt, "
            "validation.txt and test.txt."
        ),
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Optional full source corpus root. When supplied, "
            "the 14,440 locked manifest IDs are checked against "
            "the actual source paper folders."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when validation fails.",
    )

    return parser.parse_args()


def load_manifest(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing split manifest: {path}"
        )

    values = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        value = line.strip()

        if not value or value.startswith("#"):
            continue

        values.append(value)

    if len(values) != len(set(values)):
        raise RuntimeError(
            f"Duplicate paper IDs in {path}"
        )

    return set(values)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Expected JSON object at "
                    f"{path}:{line_number}"
                )

            rows.append(row)

    return rows


def atomic_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
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
                    separators=(",", ":"),
                )
                + "\n"
            )

    temporary.replace(path)


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

    temporary.replace(path)


def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    return {
        "qa_bearing_papers": len(
            {
                str(row["paper_id"])
                for row in rows
            }
        ),

        "unique_images": len(
            {
                str(row["image_id"])
                for row in rows
            }
        ),

        "unique_image_sha256": len(
            {
                str(row["image_sha256"])
                for row in rows
            }
        ),

        "qa_pairs": len(rows),

        "task_families": dict(
            Counter(
                str(row["task_family"])
                for row in rows
            )
        ),

        "qa_types": {
            qa_type: sum(
                1
                for row in rows
                if row["qa_type"] == qa_type
            )
            for qa_type in QA_TYPES
        },
    }


def validate_source_rows(
    rows: list[dict[str, Any]],
) -> list[str]:

    errors = []

    seen_qa_ids = set()

    required = (
        "qa_id",
        "image_id",
        "paper_id",
        "image",
        "image_sha256",
        "task_family",
        "qa_type",
        "question",
        "answer",
    )

    for index, row in enumerate(rows):

        missing = [
            field
            for field in required
            if row.get(field) in (
                None,
                "",
            )
        ]

        if missing:
            errors.append(
                f"row {index}: missing "
                f"{','.join(missing)}"
            )

        qa_id = str(
            row.get("qa_id", "")
        )

        if qa_id in seen_qa_ids:
            errors.append(
                f"duplicate qa_id: {qa_id}"
            )

        seen_qa_ids.add(qa_id)

        qa_type = str(
            row.get("qa_type", "")
        )

        if qa_type not in QA_TYPES:
            errors.append(
                f"unexpected qa_type: {qa_type}"
            )

    return errors


def main() -> int:
    args = parse_args()

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    split_dir = (
        args.split_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not input_path.is_file():
        print(
            f"ERROR: input not found: {input_path}",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_output = (
        output_dir
        / "manifests"
    )

    manifest_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load authoritative manifests
    # --------------------------------------------------------

    manifests = {}

    for split_name in SPLIT_NAMES:
        source = (
            split_dir
            / f"{split_name}.txt"
        )

        manifests[split_name] = (
            load_manifest(source)
        )

        shutil.copy2(
            source,
            manifest_output
            / f"{split_name}.txt",
        )

    errors = []

    # --------------------------------------------------------
    # Verify locked source counts
    # --------------------------------------------------------

    for split_name in SPLIT_NAMES:

        actual = len(
            manifests[split_name]
        )

        expected = (
            EXPECTED_SOURCE_PAPERS[
                split_name
            ]
        )

        if actual != expected:
            errors.append(
                f"{split_name}: expected "
                f"{expected} manifest papers, "
                f"found {actual}"
            )

    # --------------------------------------------------------
    # Verify paper-level disjointness
    # --------------------------------------------------------

    comparisons = {}

    for i, left in enumerate(
        SPLIT_NAMES
    ):
        for right in SPLIT_NAMES[
            i + 1:
        ]:

            overlap = (
                manifests[left]
                & manifests[right]
            )

            key = (
                f"{left}_vs_{right}"
            )

            comparisons[key] = {
                "paper_overlap":
                    len(overlap),

                "paper_examples":
                    sorted(overlap)[:20],
            }

            if overlap:
                errors.append(
                    f"{key}: "
                    f"{len(overlap)} "
                    "overlapping papers"
                )

    all_manifest_papers = (
        set().union(
            *manifests.values()
        )
    )

    if len(all_manifest_papers) != 14440:
        errors.append(
            "Expected 14,440 unique locked "
            f"source papers, found "
            f"{len(all_manifest_papers)}"
        )

    # --------------------------------------------------------
    # Optional complete source-corpus check
    # --------------------------------------------------------

    source_corpus_validation = None

    if args.source_root is not None:

        source_root = (
            args.source_root
            .expanduser()
            .resolve()
        )

        source_papers = {
            path.name
            for path in source_root.iterdir()
            if (
                path.is_dir()
                and not path.name.startswith(".")
            )
        }

        missing_from_manifests = (
            source_papers
            - all_manifest_papers
        )

        manifest_not_in_source = (
            all_manifest_papers
            - source_papers
        )

        source_corpus_validation = {
            "source_paper_folders":
                len(source_papers),

            "manifest_papers":
                len(all_manifest_papers),

            "source_missing_from_manifests":
                len(missing_from_manifests),

            "manifest_missing_from_source":
                len(manifest_not_in_source),

            "source_missing_examples":
                sorted(
                    missing_from_manifests
                )[:20],

            "manifest_missing_examples":
                sorted(
                    manifest_not_in_source
                )[:20],
        }

        if missing_from_manifests:
            errors.append(
                "Source papers absent from "
                f"locked manifests: "
                f"{len(missing_from_manifests)}"
            )

        if manifest_not_in_source:
            errors.append(
                "Locked manifest papers absent "
                f"from source corpus: "
                f"{len(manifest_not_in_source)}"
            )

    # --------------------------------------------------------
    # Load canonical QA dataset
    # --------------------------------------------------------

    rows = load_jsonl(
        input_path
    )

    errors.extend(
        validate_source_rows(
            rows
        )
    )

    canonical_papers = {
        str(row["paper_id"])
        for row in rows
    }

    # A source paper is allowed to have no QA.
    # But every QA-bearing paper must occur in a locked manifest.

    qa_papers_not_in_manifest = (
        canonical_papers
        - all_manifest_papers
    )

    if qa_papers_not_in_manifest:
        errors.append(
            "QA-bearing papers absent from "
            f"locked manifests: "
            f"{len(qa_papers_not_in_manifest)}"
        )

    zero_qa_papers = (
        all_manifest_papers
        - canonical_papers
    )

    # --------------------------------------------------------
    # Assign every QA record using locked paper IDs
    # --------------------------------------------------------

    paper_to_split = {}

    for split_name in SPLIT_NAMES:

        for paper_id in manifests[
            split_name
        ]:

            paper_to_split[
                paper_id
            ] = split_name

    split_rows = {
        split_name: []
        for split_name
        in SPLIT_NAMES
    }

    for row in rows:

        paper_id = str(
            row["paper_id"]
        )

        split_name = (
            paper_to_split.get(
                paper_id
            )
        )

        if split_name is None:
            errors.append(
                f"No locked split for "
                f"QA-bearing paper: {paper_id}"
            )
            continue

        output_row = dict(row)
        output_row["split"] = (
            split_name
        )

        split_rows[
            split_name
        ].append(
            output_row
        )

    # --------------------------------------------------------
    # Stable ordering and output
    # --------------------------------------------------------

    for split_name in SPLIT_NAMES:

        split_rows[
            split_name
        ].sort(
            key=lambda row: (
                str(row["paper_id"]),
                str(row["image_id"]),
                str(row["qa_type"]),
                str(row["qa_id"]),
            )
        )

        atomic_jsonl(
            output_dir
            / f"{split_name}.jsonl",
            split_rows[
                split_name
            ],
        )

    # --------------------------------------------------------
    # QA/image leakage audit
    # --------------------------------------------------------

    qa_sets = {}
    image_id_sets = {}
    image_hash_sets = {}

    for split_name in SPLIT_NAMES:

        qa_sets[
            split_name
        ] = {
            str(row["qa_id"])
            for row in
            split_rows[split_name]
        }

        image_id_sets[
            split_name
        ] = {
            str(row["image_id"])
            for row in
            split_rows[split_name]
        }

        image_hash_sets[
            split_name
        ] = {
            str(row["image_sha256"])
            for row in
            split_rows[split_name]
        }

    for i, left in enumerate(
        SPLIT_NAMES
    ):
        for right in SPLIT_NAMES[
            i + 1:
        ]:

            key = (
                f"{left}_vs_{right}"
            )

            qa_overlap = (
                qa_sets[left]
                & qa_sets[right]
            )

            image_id_overlap = (
                image_id_sets[left]
                & image_id_sets[right]
            )

            image_hash_overlap = (
                image_hash_sets[left]
                & image_hash_sets[right]
            )

            comparisons[key].update(
                {
                    "qa_id_overlap":
                        len(qa_overlap),

                    "image_id_overlap":
                        len(
                            image_id_overlap
                        ),

                    "image_sha256_overlap":
                        len(
                            image_hash_overlap
                        ),

                    "image_sha256_examples":
                        sorted(
                            image_hash_overlap
                        )[:20],
                }
            )

            if qa_overlap:
                errors.append(
                    f"{key}: QA leakage="
                    f"{len(qa_overlap)}"
                )

            if image_id_overlap:
                errors.append(
                    f"{key}: image-ID leakage="
                    f"{len(image_id_overlap)}"
                )

            if image_hash_overlap:
                errors.append(
                    f"{key}: identical-image "
                    f"SHA256 leakage="
                    f"{len(image_hash_overlap)}"
                )

    # --------------------------------------------------------
    # Task coverage
    # --------------------------------------------------------

    summaries = {}

    for split_name in SPLIT_NAMES:

        summaries[
            split_name
        ] = summarize(
            split_rows[
                split_name
            ]
        )

        missing_types = [
            qa_type
            for qa_type
            in QA_TYPES
            if summaries[
                split_name
            ]["qa_types"][
                qa_type
            ] == 0
        ]

        if missing_types:
            errors.append(
                f"{split_name}: missing "
                f"QA types: "
                f"{','.join(missing_types)}"
            )

    # --------------------------------------------------------
    # Zero-QA source papers
    # --------------------------------------------------------

    zero_qa_by_split = {
        split_name:
            sorted(
                manifests[
                    split_name
                ]
                - canonical_papers
            )
        for split_name
        in SPLIT_NAMES
    }

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    manifest_report = {
        "dataset":
            "HEP-VLMQA",

        "split_policy":
            "locked_paper_manifests",

        "source_manifest_dir":
            str(split_dir),

        "source_paper_counts": {
            split_name:
                len(
                    manifests[
                        split_name
                    ]
                )
            for split_name
            in SPLIT_NAMES
        },

        "qa_bearing_paper_counts": {
            split_name:
                summaries[
                    split_name
                ]["qa_bearing_papers"]
            for split_name
            in SPLIT_NAMES
        },

        "zero_qa_source_papers": {
            split_name:
                zero_qa_by_split[
                    split_name
                ]
            for split_name
            in SPLIT_NAMES
        },

        "splits":
            summaries,
    }

    validation_report = {
        "status":
            (
                "PASS"
                if not errors
                else "FAIL"
            ),

        "source_manifest_total":
            len(
                all_manifest_papers
            ),

        "canonical_qa_bearing_papers":
            len(
                canonical_papers
            ),

        "zero_qa_source_papers":
            len(
                zero_qa_papers
            ),

        "zero_qa_by_split":
            zero_qa_by_split,

        "source_corpus_validation":
            source_corpus_validation,

        "comparisons":
            comparisons,

        "errors":
            errors,
    }

    atomic_json(
        output_dir
        / "split_manifest.json",
        manifest_report,
    )

    atomic_json(
        output_dir
        / "split_validation.json",
        validation_report,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print(
        "===== LOCKED HEP-VLMQA SPLITS ====="
    )

    print(
        "Status:",
        validation_report[
            "status"
        ],
    )

    print(
        "Locked source papers:",
        len(
            all_manifest_papers
        ),
    )

    print(
        "QA-bearing papers:",
        len(
            canonical_papers
        ),
    )

    print(
        "Zero-QA source papers:",
        len(
            zero_qa_papers
        ),
    )

    print()

    for split_name in SPLIT_NAMES:

        summary = summaries[
            split_name
        ]

        print(
            f"{split_name:10s} "
            f"source_papers="
            f"{len(manifests[split_name]):6d} "
            f"qa_papers="
            f"{summary['qa_bearing_papers']:6d} "
            f"images="
            f"{summary['unique_images']:7d} "
            f"qa="
            f"{summary['qa_pairs']:7d}"
        )

    print()

    for key, result in comparisons.items():

        print(
            f"{key}: "
            f"papers={result['paper_overlap']} "
            f"qa={result.get('qa_id_overlap', 0)} "
            f"image_id="
            f"{result.get('image_id_overlap', 0)} "
            f"image_sha256="
            f"{result.get('image_sha256_overlap', 0)}"
        )

    if errors:

        print(
            "\nValidation errors:"
        )

        for error in errors:
            print(
                " ",
                error,
            )

    print(
        "\nOutput:",
        output_dir,
    )

    if (
        args.strict
        and errors
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
