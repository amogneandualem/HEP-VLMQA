#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = [
    (
        "V1 LoRA",
        "v1_lora",
        "V1-only",
    ),
    (
        "Multi-task LoRA",
        "multitask_lora",
        "V1+V2",
    ),
    (
        "GM-LoRA",
        "grouped_multitask_lora",
        "V1+V2",
    ),
]


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def nested_metric(data, group, key):
    block = data.get(group)

    if not isinstance(block, dict):
        return ""

    value = block.get(key)

    if value is None:
        return ""

    return value


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact HEP-VLMQA comparison table "
            "from released test metric JSON files."
        )
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/main_results.csv"),
    )

    args = parser.parse_args()

    rows = []

    for experiment, folder, scope in EXPERIMENTS:
        path = (
            args.results_root
            / folder
            / "test_metrics.json"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        data = load_json(path)

        row = {
            "experiment": experiment,
            "evaluation_scope": scope,
            "papers": data.get("papers", ""),
            "images": data.get("images", ""),
            "total_qa": data.get("total_qa", ""),
            "token_f1_micro": data.get(
                "Token_F1",
                "",
            ),
            "rouge_l_f1_micro": data.get(
                "ROUGE_L",
                "",
            ),
            "bertscore_f1_micro": data.get(
                "BERTScore_F1",
                "",
            ),
            "macro_7_token_f1":
                nested_metric(
                    data,
                    "macro_7_task",
                    "Token_F1",
                ),
            "macro_7_rouge_l_f1":
                nested_metric(
                    data,
                    "macro_7_task",
                    "ROUGE_L",
                ),
            "macro_7_bertscore_f1":
                nested_metric(
                    data,
                    "macro_7_task",
                    "BERTScore_F1",
                ),
            "v1_macro_token_f1":
                nested_metric(
                    data,
                    "v1_semantic_4_task_macro",
                    "Token_F1",
                ),
            "v1_macro_rouge_l_f1":
                nested_metric(
                    data,
                    "v1_semantic_4_task_macro",
                    "ROUGE_L",
                ),
            "v1_macro_bertscore_f1":
                nested_metric(
                    data,
                    "v1_semantic_4_task_macro",
                    "BERTScore_F1",
                ),
            "v2_macro_token_f1":
                nested_metric(
                    data,
                    "v2_provenance_3_task_macro",
                    "Token_F1",
                ),
            "v2_macro_rouge_l_f1":
                nested_metric(
                    data,
                    "v2_provenance_3_task_macro",
                    "ROUGE_L",
                ),
            "v2_macro_bertscore_f1":
                nested_metric(
                    data,
                    "v2_provenance_3_task_macro",
                    "BERTScore_F1",
                ),
        }

        rows.append(row)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} experiments to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
