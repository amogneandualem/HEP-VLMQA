#!/usr/bin/env python3

import json
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PRED_ROOT = Path(
    os.environ.get(
        "HEPVLMQA_PREDICTIONS",
        "predictions",
    )
)

SOURCE_JSONL = Path(
    os.environ.get(
        "HEPVLMQA_TEST_DATA",
        "data/grouped/test.jsonl",
    )
)

OUT_DIR = Path(
    os.environ.get(
        "HEPVLMQA_METRICS_DIR",
        "results/evaluation",
    )
)

ROBERTA_PATH = Path(
    os.environ.get(
        "HEPVLMQA_BERTSCORE_MODEL",
        "models/roberta-large",
    )
)

MODEL_NAME = os.environ.get(
    "HEPVLMQA_MODEL_NAME",
    "GM-LoRA",
)

EXPECTED_PAPERS = 100
EXPECTED_IMAGES = 741
EXPECTED_QA = 5133

EXPECTED_V1 = 3240
EXPECTED_V2 = 1893

EXPECTED_TYPES = {
    "recognition": 776,
    "visual_detail": 773,
    "interpretation": 805,
    "reasoning": 886,
    "provenance_inputs": 631,
    "provenance_method": 631,
    "provenance_figure_generation": 631,
}

V1_TYPES = {
    "recognition",
    "visual_detail",
    "interpretation",
    "reasoning",
}

V2_TYPES = {
    "provenance_inputs",
    "provenance_method",
    "provenance_figure_generation",
}

BERTSCORE_BATCH_SIZE = int(
    os.environ.get(
        "BERTSCORE_BATCH_SIZE",
        "32",
    )
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# METRIC DEFINITIONS
#
# Keep identical to previous evaluator.
# ============================================================

def normalize_text(value):
    text = (
        ""
        if value is None
        else str(value)
    )

    text = text.lower()

    text = re.sub(
        r"\\\(|\\\)|\\\[|\\\]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    return text.strip()




def token_f1(
    prediction,
    ground_truth,
):
    pred_tokens = (
        normalize_text(prediction)
        .split()
    )

    gt_tokens = (
        normalize_text(ground_truth)
        .split()
    )

    if (
        not pred_tokens
        and not gt_tokens
    ):
        return 1.0

    if (
        not pred_tokens
        or not gt_tokens
    ):
        return 0.0

    common = (
        Counter(pred_tokens)
        & Counter(gt_tokens)
    )

    overlap = sum(
        common.values()
    )

    if overlap == 0:
        return 0.0

    precision = (
        overlap
        / len(pred_tokens)
    )

    recall = (
        overlap
        / len(gt_tokens)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


# ============================================================
# GENERIC HELPERS
# ============================================================

def first_value(
    record,
    keys,
    default="",
):
    if not isinstance(
        record,
        dict,
    ):
        return default

    for key in keys:
        value = record.get(key)

        if value is None:
            continue

        if isinstance(value, str):
            if value.strip():
                return value
        else:
            return value

    return default


def nested_metadata_value(
    record,
    keys,
    default="",
):
    """
    Search top-level fields first, then common
    metadata containers.

    This intentionally does not recursively search
    QA content, avoiding accidental metadata labels.
    """

    value = first_value(
        record,
        keys,
        default="",
    )

    if value not in ("", None):
        return value

    containers = [
        "metadata",
        "image_metadata",
        "figure_metadata",
        "source_metadata",
        "paper_metadata",
        "classification",
        "labels",
    ]

    for container_name in containers:
        container = record.get(
            container_name
        )

        if isinstance(
            container,
            dict,
        ):
            value = first_value(
                container,
                keys,
                default="",
            )

            if value not in (
                "",
                None,
            ):
                return value

    return default


def clean_category(value):
    if value is None:
        return "unknown"

    value = str(value).strip()

    if not value:
        return "unknown"

    return value


def infer_family(
    qa_type,
    task_family="",
):
    task_family = str(
        task_family
    ).strip().lower()

    if task_family in {
        "v1",
        "v2",
    }:
        return task_family

    if qa_type in V1_TYPES:
        return "v1"

    if qa_type in V2_TYPES:
        return "v2"

    return "unknown"


# ============================================================
# SOURCE TEST METADATA
# ============================================================

def load_source_metadata():
    if not SOURCE_JSONL.is_file():
        raise FileNotFoundError(
            f"Missing source test file: "
            f"{SOURCE_JSONL}"
        )

    by_group = {}
    by_sha = {}
    by_path = {}

    source_rows = 0

    domain_present = 0
    image_type_present = 0
    similarity_present = 0

    with SOURCE_JSONL.open(
        encoding="utf-8"
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            row = json.loads(line)
            source_rows += 1

            group_id = str(
                row.get(
                    "group_id",
                    "",
                )
            ).strip()

            image_sha = str(
                row.get(
                    "image_sha256",
                    "",
                )
            ).strip()

            image_path = str(
                row.get(
                    "image_path",
                    "",
                )
            ).strip()

            domain = clean_category(
                nested_metadata_value(
                    row,
                    [
                        "domain",
                        "physics_domain",
                        "hep_domain",
                        "subfield",
                        "physics_subfield",
                    ],
                )
            )

            image_type = clean_category(
                nested_metadata_value(
                    row,
                    [
                        "image_type",
                        "figure_type",
                        "plot_type",
                        "visual_type",
                        "chart_type",
                    ],
                )
            )

            similarity_group = (
                clean_category(
                    nested_metadata_value(
                        row,
                        [
                            "similarity_group",
                            "similarity_cluster",
                            "group_label",
                        ],
                    )
                )
            )

            if domain != "unknown":
                domain_present += 1

            if image_type != "unknown":
                image_type_present += 1

            if (
                similarity_group
                != "unknown"
            ):
                similarity_present += 1

            metadata = {
                "domain":
                    domain,

                "image_type":
                    image_type,

                "similarity_group":
                    similarity_group,

                "source_group_id":
                    group_id,

                "source_image_sha256":
                    image_sha,

                "source_image_path":
                    image_path,
            }

            if group_id:
                by_group[
                    group_id
                ] = metadata

            if image_sha:
                by_sha[
                    image_sha
                ] = metadata

            if image_path:
                by_path[
                    image_path
                ] = metadata

    if source_rows != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} "
            f"source image groups but found "
            f"{source_rows}."
        )

    coverage = {
        "source_image_groups":
            source_rows,

        "domain_labeled_images":
            domain_present,

        "domain_coverage":
            (
                domain_present
                / source_rows
            ),

        "image_type_labeled_images":
            image_type_present,

        "image_type_coverage":
            (
                image_type_present
                / source_rows
            ),

        "similarity_group_labeled_images":
            similarity_present,

        "similarity_group_coverage":
            (
                similarity_present
                / source_rows
            ),
    }

    return (
        by_group,
        by_sha,
        by_path,
        coverage,
    )


def lookup_source_metadata(
    image_record,
    by_group,
    by_sha,
    by_path,
):
    group_id = str(
        image_record.get(
            "group_id",
            "",
        )
    ).strip()

    sha = str(
        image_record.get(
            "image_sha256",
            "",
        )
    ).strip()

    image_path = str(
        image_record.get(
            "image_path",
            "",
        )
    ).strip()

    if (
        group_id
        and group_id in by_group
    ):
        return by_group[group_id]

    if (
        sha
        and sha in by_sha
    ):
        return by_sha[sha]

    if (
        image_path
        and image_path in by_path
    ):
        return by_path[image_path]

    return {
        "domain":
            "unknown",

        "image_type":
            "unknown",

        "similarity_group":
            "unknown",
    }


# ============================================================
# READ ALL 100 GENERATED PAPER FILES
# ============================================================

def collect_predictions():
    if not PRED_ROOT.is_dir():
        raise FileNotFoundError(
            f"Prediction root missing: "
            f"{PRED_ROOT}"
        )

    (
        source_by_group,
        source_by_sha,
        source_by_path,
        metadata_coverage,
    ) = load_source_metadata()

    files = sorted(
        PRED_ROOT.glob(
            "*.json"
        )
    )

    if len(files) != EXPECTED_PAPERS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAPERS} "
            f"paper JSON files; found "
            f"{len(files)}."
        )

    rows = []

    paper_ids = set()
    image_keys = set()
    qa_ids = []

    incomplete_papers = []
    generation_errors = 0
    missing_images = 0

    for json_path in tqdm(
        files,
        desc="Reading GM-LoRA predictions",
    ):
        with json_path.open(
            encoding="utf-8"
        ) as handle:
            paper = json.load(
                handle
            )

        status = str(
            paper.get(
                "status",
                "",
            )
        )

        paper_id = str(
            paper.get(
                "paper_id",
                json_path.stem,
            )
        ).strip()

        paper_ids.add(
            paper_id
        )

        if status != "complete":
            incomplete_papers.append(
                paper_id
            )

        summary = paper.get(
            "summary",
            {},
        )

        generation_errors += int(
            summary.get(
                "generation_errors",
                0,
            )
        )

        missing_images += int(
            summary.get(
                "missing_images",
                0,
            )
        )

        images = paper.get(
            "images",
            [],
        )

        for image_index, image in enumerate(
            images
        ):
            if not isinstance(
                image,
                dict,
            ):
                continue

            group_id = str(
                image.get(
                    "group_id",
                    "",
                )
            ).strip()

            image_sha = str(
                image.get(
                    "image_sha256",
                    "",
                )
            ).strip()

            image_path = str(
                image.get(
                    "image_path",
                    "",
                )
            ).strip()

            image_key = (
                group_id
                if group_id
                else (
                    f"{paper_id}::"
                    f"{image_path}"
                )
            )

            image_keys.add(
                image_key
            )

            source_meta = (
                lookup_source_metadata(
                    image,
                    source_by_group,
                    source_by_sha,
                    source_by_path,
                )
            )

            # Prefer explicit prediction metadata
            # if it exists, otherwise source labels.
            domain = clean_category(
                first_value(
                    image,
                    [
                        "domain",
                        "physics_domain",
                    ],
                    default=source_meta[
                        "domain"
                    ],
                )
            )

            image_type = clean_category(
                first_value(
                    image,
                    [
                        "image_type",
                        "figure_type",
                        "plot_type",
                    ],
                    default=source_meta[
                        "image_type"
                    ],
                )
            )

            similarity_group = (
                clean_category(
                    first_value(
                        image,
                        [
                            "similarity_group",
                            "group",
                        ],
                        default=source_meta[
                            "similarity_group"
                        ],
                    )
                )
            )

            qa_results = image.get(
                "qa_results",
                [],
            )

            for qa_index, qa in enumerate(
                qa_results
            ):
                if not isinstance(
                    qa,
                    dict,
                ):
                    continue

                qa_id = str(
                    qa.get(
                        "qa_id",
                        "",
                    )
                ).strip()

                qa_ids.append(
                    qa_id
                )

                qa_type = str(
                    qa.get(
                        "qa_type",
                        "",
                    )
                ).strip()

                family = infer_family(
                    qa_type,
                    qa.get(
                        "task_family",
                        "",
                    ),
                )

                question = str(
                    qa.get(
                        "question",
                        "",
                    )
                )

                ground_truth = str(
                    qa.get(
                        "ground_truth",
                        "",
                    )
                )

                prediction = str(
                    qa.get(
                        "prediction",
                        "",
                    )
                )

                generated_tokens = (
                    qa.get(
                        "generated_tokens",
                        None,
                    )
                )

                max_new_tokens = (
                    qa.get(
                        "max_new_tokens",
                        None,
                    )
                )

                generation_error_value = (
                    qa.get(
                        "generation_error",
                        None,
                    )
                )

                is_error = (
                    prediction
                    .strip()
                    .upper()
                    .startswith("ERROR:")
                    or generation_error_value
                    not in (None, "", False)
                )

                rows.append(
                    {
                        "model":
                            MODEL_NAME,

                        "paper_id":
                            paper_id,

                        "group_id":
                            group_id,

                        "image_sha256":
                            image_sha,

                        "image_path":
                            image_path,

                        "image_key":
                            image_key,

                        "image_index":
                            image_index,

                        "qa_index":
                            qa_index,

                        "qa_id":
                            qa_id,

                        "task_family":
                            family,

                        "qa_type":
                            qa_type,

                        "domain":
                            domain,

                        "image_type":
                            image_type,

                        "similarity_group":
                            similarity_group,

                        "question":
                            question,

                        "ground_truth":
                            ground_truth,

                        "prediction":
                            prediction,

                        "generated_tokens":
                            generated_tokens,

                        "max_new_tokens":
                            max_new_tokens,

                        "is_error_prediction":
                            is_error,

                        "source_json":
                            str(
                                json_path
                            ),
                    }
                )

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        raise RuntimeError(
            "No QA predictions parsed."
        )

    # --------------------------------------------------------
    # Mandatory integrity checks
    # --------------------------------------------------------

    if len(paper_ids) != EXPECTED_PAPERS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAPERS} papers, "
            f"found {len(paper_ids)}."
        )

    if len(image_keys) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(image_keys)}."
        )

    if len(df) != EXPECTED_QA:
        raise RuntimeError(
            f"Expected {EXPECTED_QA} QA, "
            f"found {len(df)}."
        )

    if incomplete_papers:
        raise RuntimeError(
            "Incomplete paper outputs: "
            + ", ".join(
                incomplete_papers
            )
        )

    if generation_errors != 0:
        raise RuntimeError(
            f"Generation errors = "
            f"{generation_errors}"
        )

    if missing_images != 0:
        raise RuntimeError(
            f"Missing images = "
            f"{missing_images}"
        )

    if (
        df[
            "is_error_prediction"
        ].sum()
        != 0
    ):
        raise RuntimeError(
            "ERROR predictions detected."
        )

    missing_gt = (
        df[
            "ground_truth"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_pred = (
        df[
            "prediction"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if missing_gt:
        raise RuntimeError(
            f"Missing ground truth: "
            f"{missing_gt}"
        )

    if empty_pred:
        raise RuntimeError(
            f"Empty predictions: "
            f"{empty_pred}"
        )

    # QA ID duplicate audit.
    qa_id_series = pd.Series(
        qa_ids
    )

    nonempty_ids = (
        qa_id_series[
            qa_id_series
            .astype(str)
            .str.strip()
            .ne("")
        ]
    )

    duplicate_ids = (
        nonempty_ids[
            nonempty_ids.duplicated(
                keep=False
            )
        ]
        .tolist()
    )

    if duplicate_ids:
        duplicate_file = (
            OUT_DIR
            / "duplicate_qa_ids.txt"
        )

        duplicate_file.write_text(
            "\n".join(
                sorted(
                    set(
                        duplicate_ids
                    )
                )
            ),
            encoding="utf-8",
        )

        raise RuntimeError(
            "Duplicate qa_id values detected. "
            f"See {duplicate_file}"
        )

    # --------------------------------------------------------
    # Expected QA counts
    # --------------------------------------------------------

    observed_types = (
        df["qa_type"]
        .value_counts()
        .to_dict()
    )

    for qa_type, expected in (
        EXPECTED_TYPES.items()
    ):
        observed = int(
            observed_types.get(
                qa_type,
                0,
            )
        )

        if observed != expected:
            raise RuntimeError(
                f"{qa_type}: expected "
                f"{expected}, observed "
                f"{observed}"
            )

    v1_count = int(
        (
            df["task_family"]
            == "v1"
        ).sum()
    )

    v2_count = int(
        (
            df["task_family"]
            == "v2"
        ).sum()
    )

    if v1_count != EXPECTED_V1:
        raise RuntimeError(
            f"Expected V1={EXPECTED_V1}, "
            f"observed {v1_count}"
        )

    if v2_count != EXPECTED_V2:
        raise RuntimeError(
            f"Expected V2={EXPECTED_V2}, "
            f"observed {v2_count}"
        )

    audit = {
        "paper_files":
            len(files),

        "papers":
            len(paper_ids),

        "images":
            len(image_keys),

        "qa":
            len(df),

        "v1":
            v1_count,

        "v2":
            v2_count,

        "generation_errors":
            generation_errors,

        "missing_images":
            missing_images,

        "empty_predictions":
            int(empty_pred),

        "missing_ground_truth":
            int(missing_gt),

        "qa_type_counts":
            {
                key:
                    int(value)
                for key, value
                in observed_types.items()
            },

        "metadata_coverage":
            metadata_coverage,
    }

    return (
        df,
        audit,
    )


# ============================================================
# AGGREGATION
# ============================================================

def aggregate(
    dataframe,
    group_columns,
):
    if isinstance(
        group_columns,
        str,
    ):
        group_columns = [
            group_columns
        ]

    working = (
        dataframe.copy()
    )

    for column in group_columns:
        working[column] = (
            working[column]
            .fillna("unknown")
            .astype(str)
            .replace(
                "",
                "unknown",
            )
        )

    result = (
        working
        .groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            count=(
                "qa_id",
                "size",
            ),

            papers=(
                "paper_id",
                "nunique",
            ),

            images=(
                "image_key",
                "nunique",
            ),

            Token_F1=(
                "token_f1",
                "mean",
            ),

            ROUGE_L=(
                "rouge_l",
                "mean",
            ),

            BERTScore_P=(
                "bertscore_precision",
                "mean",
            ),

            BERTScore_R=(
                "bertscore_recall",
                "mean",
            ),

            BERTScore_F1=(
                "bertscore_f1",
                "mean",
            ),
        )
        .reset_index()
    )

    return result


# ============================================================
# FIGURE
# ============================================================

def make_metric_plot(
    dataframe,
    label_column,
    title,
    output_path,
    top_n=None,
):
    plot_df = (
        dataframe.copy()
    )

    if top_n is not None:
        plot_df = (
            plot_df
            .sort_values(
                "count",
                ascending=False,
            )
            .head(top_n)
        )

    if plot_df.empty:
        return

    for metric in [
        "Token_F1",
        "ROUGE_L",
        "BERTScore_F1",
    ]:
        plot_df[metric] = (
            plot_df[metric]
            * 100
        )

    ax = plot_df.plot(
        x=label_column,
        y=[
            "Token_F1",
            "ROUGE_L",
            "BERTScore_F1",
        ],
        kind="bar",
        figsize=(14, 7),
        rot=40,
    )

    ax.set_title(
        title
    )

    ax.set_ylabel(
        "Score (%)"
    )

    ax.set_xlabel("")

    ax.set_ylim(
        0,
        100,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("GM-LoRA SHARED TEST EVALUATION")
    print("=" * 80)

    print(
        "Predictions:",
        PRED_ROOT,
    )

    print(
        "Source test:",
        SOURCE_JSONL,
    )

    print(
        "Metrics output:",
        OUT_DIR,
    )

    # --------------------------------------------------------
    # Load predictions and validate
    # --------------------------------------------------------

    df, audit = (
        collect_predictions()
    )

    print("\nINPUT AUDIT")
    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Token-F1 / ROUGE-L
    # --------------------------------------------------------

    rouge = (
        rouge_scorer
        .RougeScorer(
            ["rougeL"],
            use_stemmer=True,
        )
    )
    token_f1_scores = []
    rouge_scores = []

    for row in tqdm(
        df.itertuples(
            index=False
        ),
        total=len(df),
        desc=(
            "Token-F1 / ROUGE-L"
        ),
    ):
        prediction = (
            row.prediction
        )

        ground_truth = (
            row.ground_truth
        )


        token_f1_scores.append(
            token_f1(
                prediction,
                ground_truth,
            )
        )

        rouge_scores.append(
            rouge.score(
                str(
                    ground_truth
                ),
                str(
                    prediction
                ),
            )[
                "rougeL"
            ].fmeasure
        )

    df["token_f1"] = (
        token_f1_scores
    )

    df["rouge_l"] = (
        rouge_scores
    )

    # --------------------------------------------------------
    # BERTScore
    # --------------------------------------------------------

    if not ROBERTA_PATH.is_dir():
        raise FileNotFoundError(
            f"RoBERTa model missing: "
            f"{ROBERTA_PATH}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nBERTScore device:",
        device,
    )

    print(
        "BERTScore model:",
        ROBERTA_PATH,
    )

    print(
        "BERTScore batch size:",
        BERTSCORE_BATCH_SIZE,
    )

    predictions = (
        df["prediction"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    references = (
        df["ground_truth"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    (
        bert_p,
        bert_r,
        bert_f1,
    ) = bert_score(
        predictions,
        references,
        model_type=str(
            ROBERTA_PATH
        ),
        num_layers=17,
        lang="en",
        rescale_with_baseline=False,
        batch_size=(
            BERTSCORE_BATCH_SIZE
        ),
        device=device,
        verbose=True,
    )

    df[
        "bertscore_precision"
    ] = (
        bert_p
        .cpu()
        .tolist()
    )

    df[
        "bertscore_recall"
    ] = (
        bert_r
        .cpu()
        .tolist()
    )

    df[
        "bertscore_f1"
    ] = (
        bert_f1
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------
    # GROUPED RESULTS
    # --------------------------------------------------------

    by_family = aggregate(
        df,
        "task_family",
    )

    by_qa_type = aggregate(
        df,
        "qa_type",
    )

    by_domain = aggregate(
        df,
        "domain",
    )

    by_image_type = aggregate(
        df,
        "image_type",
    )

    by_similarity = aggregate(
        df,
        "similarity_group",
    )

    by_paper = aggregate(
        df,
        "paper_id",
    )

    by_domain_qa = aggregate(
        df,
        [
            "domain",
            "qa_type",
        ],
    )

    by_image_type_qa = aggregate(
        df,
        [
            "image_type",
            "qa_type",
        ],
    )

    # --------------------------------------------------------
    # OVERALL MICRO METRICS
    # --------------------------------------------------------

    overall = {
        "model":
            MODEL_NAME,

        "papers":
            int(
                df[
                    "paper_id"
                ].nunique()
            ),

        "images":
            int(
                df[
                    "image_key"
                ].nunique()
            ),

        "total_qa":
            int(
                len(df)
            ),

        "Token_F1":
            float(
                df[
                    "token_f1"
                ].mean()
            ),

        "ROUGE_L":
            float(
                df[
                    "rouge_l"
                ].mean()
            ),

        "BERTScore_P":
            float(
                df[
                    "bertscore_precision"
                ].mean()
            ),

        "BERTScore_R":
            float(
                df[
                    "bertscore_recall"
                ].mean()
            ),

        "BERTScore_F1":
            float(
                df[
                    "bertscore_f1"
                ].mean()
            ),
    }

    # --------------------------------------------------------
    # MACRO RESULTS
    #
    # Each QA TYPE receives equal weight.
    # --------------------------------------------------------

    macro_7 = {

        "Token_F1":
            float(
                by_qa_type[
                    "Token_F1"
                ].mean()
            ),

        "ROUGE_L":
            float(
                by_qa_type[
                    "ROUGE_L"
                ].mean()
            ),

        "BERTScore_F1":
            float(
                by_qa_type[
                    "BERTScore_F1"
                ].mean()
            ),
    }

    v1_type_table = (
        by_qa_type[
            by_qa_type[
                "qa_type"
            ].isin(
                V1_TYPES
            )
        ]
    )

    v2_type_table = (
        by_qa_type[
            by_qa_type[
                "qa_type"
            ].isin(
                V2_TYPES
            )
        ]
    )

    v1_macro = {
        "Token_F1":
            float(
                v1_type_table[
                    "Token_F1"
                ].mean()
            ),

        "ROUGE_L":
            float(
                v1_type_table[
                    "ROUGE_L"
                ].mean()
            ),

        "BERTScore_F1":
            float(
                v1_type_table[
                    "BERTScore_F1"
                ].mean()
            ),
    }

    v2_macro = {
        "Token_F1":
            float(
                v2_type_table[
                    "Token_F1"
                ].mean()
            ),

        "ROUGE_L":
            float(
                v2_type_table[
                    "ROUGE_L"
                ].mean()
            ),

        "BERTScore_F1":
            float(
                v2_type_table[
                    "BERTScore_F1"
                ].mean()
            ),
    }

    overall[
        "macro_7_task"
    ] = macro_7

    overall[
        "v1_semantic_4_task_macro"
    ] = v1_macro

    overall[
        "v2_provenance_3_task_macro"
    ] = v2_macro

    # --------------------------------------------------------
    # OUTPUT-LENGTH AUDIT
    # --------------------------------------------------------

    v1_df = df[
        df[
            "task_family"
        ]
        == "v1"
    ].copy()

    v2_df = df[
        df[
            "task_family"
        ]
        == "v2"
    ].copy()

    v1_generated = pd.to_numeric(
        v1_df[
            "generated_tokens"
        ],
        errors="coerce",
    )

    v2_generated = pd.to_numeric(
        v2_df[
            "generated_tokens"
        ],
        errors="coerce",
    )

    v1_hit_cap = int(
        (
            v1_generated == 32
        ).sum()
    )

    v2_hit_cap = int(
        (
            v2_generated == 256
        ).sum()
    )

    length_audit = {
        "v1_total":
            int(
                len(v1_df)
            ),

        "v1_max_new_tokens":
            32,

        "v1_hit_ceiling":
            v1_hit_cap,

        "v1_hit_ceiling_rate":
            float(
                v1_hit_cap
                / len(v1_df)
            ),

        "v2_total":
            int(
                len(v2_df)
            ),

        "v2_max_new_tokens":
            256,

        "v2_hit_ceiling":
            v2_hit_cap,

        "v2_hit_ceiling_rate":
            float(
                v2_hit_cap
                / len(v2_df)
            ),
    }

    # --------------------------------------------------------
    # SAVE SAMPLE-LEVEL RESULTS
    # --------------------------------------------------------

    df.to_csv(
        OUT_DIR
        / "all_qa_predictions_with_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE GROUPED TABLES
    # --------------------------------------------------------

    tables = {
        "metrics_by_task_family.csv":
            by_family,

        "metrics_by_qa_type.csv":
            by_qa_type,

        "metrics_by_domain.csv":
            by_domain,

        "metrics_by_image_type.csv":
            by_image_type,

        "metrics_by_similarity_group.csv":
            by_similarity,

        "metrics_by_paper.csv":
            by_paper,

        "metrics_by_domain_and_qa_type.csv":
            by_domain_qa,

        "metrics_by_image_type_and_qa_type.csv":
            by_image_type_qa,
    }

    for filename, table in (
        tables.items()
    ):
        table.to_csv(
            OUT_DIR
            / filename,
            index=False,
        )

    # --------------------------------------------------------
    # SAVE JSON SUMMARIES
    # --------------------------------------------------------

    with (
        OUT_DIR
        / "overall_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            overall,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    with (
        OUT_DIR
        / "evaluation_audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    with (
        OUT_DIR
        / "output_length_audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            length_audit,
            handle,
            indent=2,
        )
        handle.write("\n")

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    make_metric_plot(
        by_qa_type,
        "qa_type",
        "GM-LoRA Metrics by QA Type",
        OUT_DIR
        / "metrics_by_qa_type.png",
    )

    known_domains = (
        by_domain[
            by_domain[
                "domain"
            ]
            != "unknown"
        ]
    )

    if not known_domains.empty:
        make_metric_plot(
            known_domains,
            "domain",
            "GM-LoRA Metrics by Domain",
            OUT_DIR
            / "metrics_by_domain.png",
            top_n=15,
        )

    known_image_types = (
        by_image_type[
            by_image_type[
                "image_type"
            ]
            != "unknown"
        ]
    )

    if not known_image_types.empty:
        make_metric_plot(
            known_image_types,
            "image_type",
            (
                "GM-LoRA Metrics "
                "by Image Type"
            ),
            OUT_DIR
            / "metrics_by_image_type.png",
            top_n=15,
        )

    # --------------------------------------------------------
    # MANUSCRIPT-FRIENDLY PERCENT TABLE
    # --------------------------------------------------------

    manuscript_table = (
        by_qa_type[
            [
                "qa_type",
                "count",
                "papers",
                "images",
                "Token_F1",
                "ROUGE_L",
                "BERTScore_F1",
            ]
        ].copy()
    )

    for metric in [
        "Token_F1",
        "ROUGE_L",
        "BERTScore_F1",
    ]:
        manuscript_table[
            metric
        ] = (
            manuscript_table[
                metric
            ]
            * 100
        )

    manuscript_table.to_csv(
        OUT_DIR
        / "manuscript_table_7_tasks_percent.csv",
        index=False,
    )

    # --------------------------------------------------------
    # TERMINAL REPORT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OVERALL MICRO METRICS"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            overall,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "METRICS BY QA TYPE (%)"
    )

    print(
        "=" * 80
    )

    display = (
        manuscript_table.copy()
    )

    for metric in [
        "Token_F1",
        "ROUGE_L",
        "BERTScore_F1",
    ]:
        display[
            metric
        ] = (
            display[
                metric
            ]
            .map(
                lambda x:
                    f"{x:.3f}"
            )
        )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "METADATA COVERAGE"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            audit[
                "metadata_coverage"
            ],
            indent=2,
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUTPUT LENGTH AUDIT"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            length_audit,
            indent=2,
        )
    )

    print(
        "\nSaved evaluation outputs to:"
    )

    print(
        OUT_DIR
    )


if __name__ == "__main__":
    main()
