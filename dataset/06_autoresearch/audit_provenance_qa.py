#!/usr/bin/env python3
"""Audit preserved complete papers plus V8 multimodal partial/final outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
import os
from pathlib import Path
from typing import Any

PIPELINE_VERSION = "auto_research_v8_1_multimodal"
INPUT_NAME = "semantic_qa_agent_final_api.json"
OUTPUT_NAME = "auto_research_qa.json"

DEFAULT_DATA = Path(
    os.environ.get(
        "HEPVLMQA_DATA_ROOT",
        "data",
    )
)
DEFAULT_WORK = Path(os.environ.get("AR_WORK", "./autoresearch_work"))

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


def load_json_list(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def expected_uids(paper_dir: Path) -> list[str]:
    return [
        str(record.get("uid"))
        for record in load_json_list(paper_dir / INPUT_NAME)
        if (
            record.get("uid")
            and record.get("image_type_v2") in AUTO_RESEARCH_IMAGE_TYPES
        )
    ]


def record_structurally_valid(record: Any, require_v8: bool = False) -> bool:
    if not isinstance(record, dict):
        return False
    if require_v8 and record.get("pipeline_version") != PIPELINE_VERSION:
        return False

    qas = record.get("provenance_qa_pairs")
    if not isinstance(qas, list) or len(qas) != 3:
        return False

    actual_types = [
        qa.get("qa_type") if isinstance(qa, dict) else None
        for qa in qas
    ]
    if actual_types != QA_TYPES:
        return False

    questions: list[str] = []
    for qa in qas:
        if not isinstance(qa, dict):
            return False
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if not question or not answer:
            return False
        questions.append(" ".join(question.lower().split()))

    return len(set(questions)) == 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    source_papers = sorted(
        path
        for path in args.data_root.iterdir()
        if path.is_dir() and (path / INPUT_NAME).exists()
    )
    expected_by_paper = {
        paper_dir.name: expected_uids(paper_dir)
        for paper_dir in source_papers
    }
    expected_total = sum(len(uids) for uids in expected_by_paper.values())

    final_files = sorted(args.data_root.glob(f"*/{OUTPUT_NAME}"))
    valid_final_papers = 0
    valid_final_images = 0
    valid_final_qas = 0
    empty_completed_papers = 0
    preserved_pre_v8_papers = 0
    new_v8_final_papers = 0
    mixed_pipeline_final_files = 0
    invalid_final_files = 0
    final_uids: set[tuple[str, str]] = set()
    valid_final_paper_ids: set[str] = set()
    pipeline_counts: Counter[str] = Counter()
    problems: list[tuple[str, str]] = []

    for final_path in final_files:
        paper_id = final_path.parent.name
        expected = expected_by_paper.get(paper_id, [])
        records = load_json_list(final_path)

        if not expected:
            if records == []:
                valid_final_papers += 1
                empty_completed_papers += 1
                valid_final_paper_ids.add(paper_id)
            else:
                invalid_final_files += 1
                problems.append((paper_id, "expected empty final output"))
            continue

        actual_uids = [
            str(record.get("uid"))
            for record in records
            if isinstance(record, dict) and record.get("uid")
        ]
        valid = (
            len(records) == len(expected)
            and len(actual_uids) == len(set(actual_uids))
            and set(actual_uids) == set(expected)
            and all(record_structurally_valid(record) for record in records)
        )
        if not valid:
            invalid_final_files += 1
            problems.append((paper_id, "invalid final structure or UID set"))
            continue

        versions = {
            str(record.get("pipeline_version", "unknown"))
            for record in records
        }
        if len(versions) > 1:
            mixed_pipeline_final_files += 1
            problems.append((paper_id, f"mixed final pipelines: {sorted(versions)}"))
            continue

        version = next(iter(versions)) if versions else "empty"
        pipeline_counts[version] += 1
        if version == PIPELINE_VERSION:
            new_v8_final_papers += 1
        else:
            preserved_pre_v8_papers += 1

        valid_final_papers += 1
        valid_final_images += len(records)
        valid_final_qas += len(records) * 3
        valid_final_paper_ids.add(paper_id)
        for uid in actual_uids:
            final_uids.add((paper_id, uid))

    partial_dir = args.work / "progress" / "partials_v8"
    partial_files = sorted(partial_dir.glob(f"{PIPELINE_VERSION}_*.json"))
    valid_partial_papers = 0
    valid_partial_images = 0
    invalid_partial_files = 0
    duplicate_partial_uids = 0
    partial_seen: set[tuple[str, str]] = set()

    for partial_path in partial_files:
        prefix = f"{PIPELINE_VERSION}_"
        paper_id = partial_path.stem[len(prefix):]
        expected = set(expected_by_paper.get(paper_id, []))
        records = load_json_list(partial_path)

        if paper_id in valid_final_paper_ids:
            invalid_partial_files += 1
            problems.append((paper_id, "V8 partial exists for completed final paper"))
            continue

        actual_uids = [
            str(record.get("uid"))
            for record in records
            if isinstance(record, dict) and record.get("uid")
        ]
        valid = (
            bool(records)
            and len(actual_uids) == len(records)
            and len(actual_uids) == len(set(actual_uids))
            and set(actual_uids).issubset(expected)
            and all(
                record_structurally_valid(record, require_v8=True)
                for record in records
            )
        )
        if not valid:
            invalid_partial_files += 1
            problems.append((paper_id, "invalid V8 partial structure or UID set"))
            continue

        duplicate_here = False
        for uid in actual_uids:
            key = (paper_id, uid)
            if key in final_uids or key in partial_seen:
                duplicate_partial_uids += 1
                duplicate_here = True
            partial_seen.add(key)

        if duplicate_here:
            continue

        valid_partial_papers += 1
        valid_partial_images += len(records)

    safely_saved = valid_final_images + valid_partial_images
    remaining = max(0, expected_total - safely_saved)
    progress = 100.0 * safely_saved / expected_total if expected_total else 100.0

    print("=" * 80)
    print("AUTO RESEARCH V8 MULTIMODAL AUDIT")
    print("=" * 80)
    print(f"Source papers:                  {len(source_papers):,}")
    print(f"Expected eligible images:       {expected_total:,}")
    print(f"Final files discovered:         {len(final_files):,}")
    print(f"Valid completed papers:         {valid_final_papers:,}")
    print(f"Preserved pre-V8 papers:        {preserved_pre_v8_papers:,}")
    print(f"New V8 multimodal papers:       {new_v8_final_papers:,}")
    print(f"Valid final images:             {valid_final_images:,}")
    print(f"Valid final QA pairs:           {valid_final_qas:,}")
    print(f"Empty completed papers:         {empty_completed_papers:,}")
    print(f"Mixed-pipeline final files:     {mixed_pipeline_final_files:,}")
    print(f"Invalid final files:            {invalid_final_files:,}")
    print()
    print(f"Valid V8 partial papers:        {valid_partial_papers:,}")
    print(f"Valid V8 partial images:        {valid_partial_images:,}")
    print(f"Invalid V8 partial files:       {invalid_partial_files:,}")
    print(f"Duplicate V8 partial UIDs:      {duplicate_partial_uids:,}")
    print(f"Safely saved images total:      {safely_saved:,}")
    print(f"Estimated remaining images:     {remaining:,}")
    print(f"Saved-image progress:           {progress:.2f}%")

    if pipeline_counts:
        print()
        print("Completed-paper pipeline counts:")
        for version, count in sorted(pipeline_counts.items()):
            print(f"  {version}: {count:,}")

    if problems:
        print()
        print("Problem examples:")
        for paper_id, reason in problems[:20]:
            print(f"  {paper_id}: {reason}")

    dirty = any(
        [
            mixed_pipeline_final_files,
            invalid_final_files,
            invalid_partial_files,
            duplicate_partial_uids,
        ]
    )
    if args.require_clean and dirty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
