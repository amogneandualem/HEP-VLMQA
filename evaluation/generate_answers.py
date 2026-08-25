#!/usr/bin/env python3

import argparse
import json
import os
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
)


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


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-dir", required=True)

    # Historical V1 inference protocol.
    parser.add_argument(
        "--v1-max-new-tokens",
        type=int,
        default=32,
    )

    # Historical V2 inference protocol.
    parser.add_argument(
        "--v2-max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--v2-max-input-tokens",
        type=int,
        default=8192,
    )

    # 0 = entire test set.
    # >0 = first N papers, useful for smoke tests.
    parser.add_argument(
        "--paper-limit",
        type=int,
        default=0,
    )

    return parser.parse_args()


def timestamp():
    return datetime.now().isoformat(
        timespec="seconds"
    )


# ============================================================
# Dataset helpers
# ============================================================

def load_jsonl(path):
    rows = []

    with Path(path).open(
        encoding="utf-8"
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))

            except Exception as error:
                raise RuntimeError(
                    f"Invalid JSON at line "
                    f"{line_number}: {error}"
                )

    return rows


def group_by_paper(rows):
    grouped = defaultdict(list)

    for row in rows:
        paper_id = str(row["paper_id"])
        grouped[paper_id].append(row)

    return dict(grouped)


def count_qas(records):
    return sum(
        len(record.get("qas", []))
        for record in records
    )


def task_family(qa):
    family = str(
        qa.get("task_family", "")
    ).strip()

    qa_type = str(
        qa.get("qa_type", "")
    ).strip()

    if family in {"v1", "v2"}:
        return family

    if qa_type in V1_TYPES:
        return "v1"

    if qa_type in V2_TYPES:
        return "v2"

    raise ValueError(
        f"Unknown QA family/type: "
        f"family={family!r}, type={qa_type!r}"
    )


# ============================================================
# Balanced 8-GPU assignment
#
# Rather than simply giving every rank 12-13 papers,
# balance ranks by expected QA count.
# ============================================================

def balanced_assignments(
    by_paper,
    world_size,
):
    paper_work = []

    for paper_id, records in by_paper.items():
        paper_work.append(
            (
                paper_id,
                count_qas(records),
            )
        )

    # Largest papers first.
    paper_work.sort(
        key=lambda x: (-x[1], x[0])
    )

    bins = [
        {
            "load": 0,
            "papers": [],
        }
        for _ in range(world_size)
    ]

    for paper_id, qa_count in paper_work:

        target_rank = min(
            range(world_size),
            key=lambda rank: (
                bins[rank]["load"],
                rank,
            ),
        )

        bins[target_rank][
            "papers"
        ].append(paper_id)

        bins[target_rank][
            "load"
        ] += qa_count

    return bins


# ============================================================
# Prompt
#
# IMPORTANT:
# The model receives ONLY:
#   1. image
#   2. exact question
#
# No caption.
# No article text.
# No reference answer.
# No metadata.
# ============================================================

def build_prompt(
    processor,
    question,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ============================================================
# V1 input protocol
#
# Reproduces the earlier V1 style:
# processor(image + exact question)
# no caption/context
# no explicit text truncation
# ============================================================

def prepare_v1_inputs(
    processor,
    image,
    prompt,
):
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    if "attention_mask" in inputs:
        input_tokens = int(
            inputs["attention_mask"]
            .sum()
            .item()
        )
    else:
        input_tokens = int(
            inputs["input_ids"].shape[1]
        )

    return {
        "inputs": inputs,
        "input_tokens": input_tokens,
        "processed_image_size":
            list(image.size),
        "resize_history": [],
    }


# ============================================================
# V2 input protocol
#
# Matches historical generate_v2_answers.py:
# padding=False
# truncation=False
# max multimodal input = 8192 tokens
# resize image only if necessary.
# ============================================================

def prepare_v2_inputs(
    processor,
    image,
    prompt,
    max_input_tokens,
):
    working_image = image
    resize_history = []

    for attempt in range(8):

        inputs = processor(
            images=working_image,
            text=prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        input_length = int(
            inputs["attention_mask"]
            .sum()
            .item()
        )

        if input_length <= max_input_tokens:
            return {
                "inputs": inputs,
                "input_tokens":
                    input_length,
                "processed_image_size":
                    list(
                        working_image.size
                    ),
                "resize_history":
                    resize_history,
            }

        old_width, old_height = (
            working_image.size
        )

        scale = (
            (
                max_input_tokens
                * 0.90
                / input_length
            )
            ** 0.5
        )

        scale = max(
            0.50,
            min(0.90, scale),
        )

        new_width = max(
            64,
            int(old_width * scale),
        )

        new_height = max(
            64,
            int(old_height * scale),
        )

        if (
            new_width >= old_width
            and new_height >= old_height
        ):
            new_width = max(
                64,
                int(old_width * 0.80),
            )

            new_height = max(
                64,
                int(old_height * 0.80),
            )

        resize_history.append(
            {
                "attempt":
                    attempt + 1,

                "old_size": [
                    old_width,
                    old_height,
                ],

                "new_size": [
                    new_width,
                    new_height,
                ],

                "input_tokens_before_resize":
                    input_length,
            }
        )

        working_image = (
            working_image.resize(
                (
                    new_width,
                    new_height,
                ),
                Image.Resampling.LANCZOS,
            )
        )

    raise RuntimeError(
        "Unable to reduce multimodal "
        f"input below {max_input_tokens} tokens."
    )


# ============================================================
# Move tensors
# ============================================================

def move_inputs_to_device(
    inputs,
    device,
):
    result = {}

    for key, value in inputs.items():

        if torch.is_tensor(value):
            result[key] = value.to(
                device,
                non_blocking=True,
            )
        else:
            result[key] = value

    return result


# ============================================================
# Generate one answer
# ============================================================

def generate_answer(
    model,
    processor,
    device,
    image,
    question,
    family,
    args,
):
    # ONLY exact question enters prompt.
    prompt = build_prompt(
        processor,
        question,
    )

    if family == "v1":

        prepared = prepare_v1_inputs(
            processor,
            image,
            prompt,
        )

        max_new_tokens = (
            args.v1_max_new_tokens
        )

        protocol = (
            "v1_historical_image_question_only"
        )

    elif family == "v2":

        prepared = prepare_v2_inputs(
            processor,
            image,
            prompt,
            args.v2_max_input_tokens,
        )

        max_new_tokens = (
            args.v2_max_new_tokens
        )

        protocol = (
            "v2_historical_image_question_only"
        )

    else:
        raise ValueError(
            f"Unknown family: {family}"
        )

    inputs = move_inputs_to_device(
        prepared["inputs"],
        device,
    )

    start = time.time()

    with torch.inference_mode():

        generated = model.generate(
            **inputs,

            max_new_tokens=
                max_new_tokens,

            do_sample=False,

            num_beams=1,

            use_cache=True,

            pad_token_id=
                processor.tokenizer.pad_token_id,

            eos_token_id=
                processor.tokenizer.eos_token_id,
        )

    generation_seconds = (
        time.time() - start
    )

    prompt_length = int(
        inputs["input_ids"].shape[1]
    )

    generated_ids = generated[
        :,
        prompt_length:,
    ]

    prediction = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )

    return {
        "prediction":
            prediction,

        "input_tokens":
            prepared[
                "input_tokens"
            ],

        "generated_tokens":
            int(
                generated_ids.shape[1]
            ),

        "max_new_tokens":
            max_new_tokens,

        "processed_image_size":
            prepared[
                "processed_image_size"
            ],

        "resize_history":
            prepared[
                "resize_history"
            ],

        "generation_seconds":
            generation_seconds,

        "protocol":
            protocol,
    }


# ============================================================
# Completed paper check
# ============================================================

def paper_is_complete(
    output_path,
    expected_qas,
):
    if not output_path.is_file():
        return False

    try:
        with output_path.open(
            encoding="utf-8"
        ) as handle:
            data = json.load(handle)

        summary = data.get(
            "summary",
            {},
        )

        return (
            data.get("status")
            == "complete"

            and summary.get(
                "qa_expected"
            )
            == expected_qas

            and summary.get(
                "qa_evaluated"
            )
            == expected_qas

            and summary.get(
                "generation_errors",
                0,
            )
            == 0

            and summary.get(
                "missing_images",
                0,
            )
            == 0

            and summary.get(
                "empty_questions",
                0,
            )
            == 0
        )

    except Exception:
        return False


# ============================================================
# Evaluate one paper
# ============================================================

def evaluate_paper(
    model,
    processor,
    device,
    paper_id,
    records,
    output_path,
    rank,
    args,
):
    expected_qas = count_qas(
        records
    )

    expected_types = Counter()

    for record in records:
        for qa in record.get(
            "qas",
            [],
        ):
            expected_types[
                str(
                    qa.get(
                        "qa_type",
                        "",
                    )
                )
            ] += 1

    result = {
        "status":
            "running",

        "paper_id":
            paper_id,

        "timestamp_start":
            timestamp(),

        "model": {
            "name":
                "GM-LoRA",

            "backbone":
                "LLaVA-OneVision-Qwen2-7B",

            "base_model":
                args.base_model,

            "adapter":
                args.adapter,

            "processor":
                args.processor,
        },

        "evaluation_protocol": {
            "input":
                "image_plus_exact_question_only",

            "caption_used":
                False,

            "article_context_used":
                False,

            "retrieval_context_used":
                False,

            "ground_truth_used_as_input":
                False,

            "sampling":
                False,

            "num_beams":
                1,

            "v1_max_new_tokens":
                args.v1_max_new_tokens,

            "v2_max_new_tokens":
                args.v2_max_new_tokens,

            "v2_max_input_tokens":
                args.v2_max_input_tokens,
        },

        "images": [],

        "summary": {
            "images_expected":
                len(records),

            "images_evaluated":
                0,

            "qa_expected":
                expected_qas,

            "qa_evaluated":
                0,

            "v1_qa_evaluated":
                0,

            "v2_qa_evaluated":
                0,

            "generation_errors":
                0,

            "missing_images":
                0,

            "empty_questions":
                0,

            "expected_qa_type_counts":
                dict(expected_types),

            "generated_qa_type_counts":
                {},
        },
    }

    type_counts = Counter()

    tmp_path = Path(
        str(output_path) + ".tmp"
    )

    print(
        f"\n[rank {rank}] "
        f"PAPER={paper_id} "
        f"images={len(records)} "
        f"qas={expected_qas}",
        flush=True,
    )

    # --------------------------------------------------------
    # Each grouped image
    # --------------------------------------------------------

    for image_index, record in enumerate(
        records,
        start=1,
    ):
        image_path = Path(
            str(
                record.get(
                    "image_path",
                    "",
                )
            )
        )

        if not image_path.is_file():

            result["summary"][
                "missing_images"
            ] += 1

            print(
                f"[rank {rank}] "
                f"MISSING IMAGE: {image_path}",
                flush=True,
            )

            continue

        try:
            with Image.open(
                image_path
            ) as source:

                image = source.convert(
                    "RGB"
                )

        except Exception as error:

            result["summary"][
                "missing_images"
            ] += 1

            print(
                f"[rank {rank}] "
                f"IMAGE OPEN ERROR: "
                f"{image_path}: {error}",
                flush=True,
            )

            continue

        qas = record.get(
            "qas",
            [],
        )

        image_result = {
            "group_id":
                record.get(
                    "group_id",
                    "",
                ),

            "paper_id":
                paper_id,

            "image_sha256":
                record.get(
                    "image_sha256",
                    "",
                ),

            "image_path":
                str(image_path),

            "original_image_size":
                list(image.size),

            "qa_results": [],
        }

        # ----------------------------------------------------
        # Each QA associated with this image
        #
        # IMPORTANT:
        # ground_truth is saved only AFTER generation.
        # It is never included in model input.
        # ----------------------------------------------------

        for qa_index, qa in enumerate(
            qas,
            start=1,
        ):
            question = str(
                qa.get(
                    "question",
                    "",
                )
            ).strip()

            qa_type = str(
                qa.get(
                    "qa_type",
                    "",
                )
            ).strip()

            if not question:

                result["summary"][
                    "empty_questions"
                ] += 1

                continue

            family = task_family(
                qa
            )

            print(
                f"[rank {rank}] "
                f"{paper_id} "
                f"image={image_index}/"
                f"{len(records)} "
                f"qa={qa_index}/"
                f"{len(qas)} "
                f"{family}/{qa_type}",
                flush=True,
            )

            generation_error = None

            try:
                generated = generate_answer(
                    model=model,
                    processor=processor,
                    device=device,
                    image=image,
                    question=question,
                    family=family,
                    args=args,
                )

            except Exception as error:

                generation_error = (
                    f"{type(error).__name__}: "
                    f"{str(error)}"
                )

                generated = {
                    "prediction":
                        "ERROR: "
                        + generation_error,

                    "input_tokens":
                        None,

                    "generated_tokens":
                        0,

                    "max_new_tokens":
                        (
                            args.v1_max_new_tokens
                            if family == "v1"
                            else
                            args.v2_max_new_tokens
                        ),

                    "processed_image_size":
                        None,

                    "resize_history":
                        [],

                    "generation_seconds":
                        None,

                    "protocol":
                        (
                            "v1_historical_image_question_only"
                            if family == "v1"
                            else
                            "v2_historical_image_question_only"
                        ),
                }

                result["summary"][
                    "generation_errors"
                ] += 1

                print(
                    f"[rank {rank}] "
                    f"GENERATION ERROR "
                    f"{paper_id} "
                    f"{qa_type}: "
                    f"{generation_error}",
                    flush=True,
                )

            qa_result = {
                "qa_id":
                    qa.get(
                        "qa_id",
                        "",
                    ),

                "task_family":
                    family,

                "qa_type":
                    qa_type,

                # INPUT text
                "question":
                    question,

                # Saved reference only.
                # NEVER supplied to model.
                "ground_truth":
                    qa.get(
                        "answer",
                        "",
                    ),

                # Model output
                "prediction":
                    generated[
                        "prediction"
                    ],

                "protocol":
                    generated[
                        "protocol"
                    ],

                "input_tokens":
                    generated[
                        "input_tokens"
                    ],

                "generated_tokens":
                    generated[
                        "generated_tokens"
                    ],

                "max_new_tokens":
                    generated[
                        "max_new_tokens"
                    ],

                "processed_image_size":
                    generated[
                        "processed_image_size"
                    ],

                "resize_history":
                    generated[
                        "resize_history"
                    ],

                "generation_seconds":
                    generated[
                        "generation_seconds"
                    ],

                "generation_error":
                    generation_error,
            }

            image_result[
                "qa_results"
            ].append(
                qa_result
            )

            result["summary"][
                "qa_evaluated"
            ] += 1

            type_counts[
                qa_type
            ] += 1

            if family == "v1":

                result["summary"][
                    "v1_qa_evaluated"
                ] += 1

            elif family == "v2":

                result["summary"][
                    "v2_qa_evaluated"
                ] += 1

        result["images"].append(
            image_result
        )

        result["summary"][
            "images_evaluated"
        ] += 1

        result["summary"][
            "generated_qa_type_counts"
        ] = dict(
            type_counts
        )

        # Save progress after each image.
        with tmp_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                result,
                handle,
                indent=2,
                ensure_ascii=False,
            )

    result["summary"][
        "generated_qa_type_counts"
    ] = dict(type_counts)

    result["timestamp_end"] = (
        timestamp()
    )

    if (
        result["summary"][
            "qa_evaluated"
        ]
        == expected_qas

        and result["summary"][
            "generation_errors"
        ]
        == 0

        and result["summary"][
            "missing_images"
        ]
        == 0

        and result["summary"][
            "empty_questions"
        ]
        == 0
    ):
        result["status"] = "complete"

    else:
        result["status"] = "incomplete"

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    tmp_path.replace(
        output_path
    )

    print(
        f"[rank {rank}] "
        f"SAVED {output_path} "
        f"status={result['status']}",
        flush=True,
    )

    return result


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    rank = int(
        os.environ.get(
            "RANK",
            str(local_rank),
        )
    )

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    torch.cuda.set_device(
        local_rank
    )

    device = torch.device(
        f"cuda:{local_rank}"
    )

    output_dir = Path(
        args.output_dir
    )

    state_dir = Path(
        args.state_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load + validate test set
    # --------------------------------------------------------

    rows = load_jsonl(
        args.input_jsonl
    )

    by_paper = group_by_paper(
        rows
    )

    all_paper_ids = sorted(
        by_paper
    )

    if args.paper_limit > 0:

        selected_ids = set(
            all_paper_ids[
                :args.paper_limit
            ]
        )

        by_paper = {
            paper_id:
                by_paper[paper_id]
            for paper_id
            in all_paper_ids
            if paper_id in selected_ids
        }

    else:
        total_qas = sum(
            len(record.get("qas", []))
            for record in rows
        )

        if len(all_paper_ids) != 100:
            raise RuntimeError(
                f"Expected 100 test papers, "
                f"found {len(all_paper_ids)}"
            )

        if len(rows) != 741:
            raise RuntimeError(
                f"Expected 741 image groups, "
                f"found {len(rows)}"
            )

        if total_qas != 5133:
            raise RuntimeError(
                f"Expected 5,133 QA, "
                f"found {total_qas}"
            )

    # --------------------------------------------------------
    # Balanced work distribution
    # --------------------------------------------------------

    assignments = balanced_assignments(
        by_paper,
        world_size,
    )

    assigned_papers = assignments[
        rank
    ]["papers"]

    assigned_qas = assignments[
        rank
    ]["load"]

    print(
        f"[rank {rank}] "
        f"world_size={world_size} "
        f"assigned_papers="
        f"{len(assigned_papers)} "
        f"assigned_qas="
        f"{assigned_qas}",
        flush=True,
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = (
        AutoProcessor
        .from_pretrained(
            args.processor,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )
    )

    if (
        processor.tokenizer.pad_token_id
        is None
    ):
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    processor.tokenizer.padding_side = (
        "left"
    )

    # Avoid 8 ranks hammering filesystem
    # at exactly same millisecond.
    time.sleep(
        rank * 2
    )

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    print(
        f"[rank {rank}] "
        "Loading LLaVA-OneVision base...",
        flush=True,
    )

    base_model = (
        LlavaOnevisionForConditionalGeneration
        .from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    )

    # --------------------------------------------------------
    # Shared LoRA
    # --------------------------------------------------------

    print(
        f"[rank {rank}] "
        "Loading shared best_adapter...",
        flush=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        is_trainable=False,
    )

    model.to(device)

    model.eval()

    model.config.use_cache = True

    print(
        f"[rank {rank}] "
        f"MODEL READY on {device}",
        flush=True,
    )

    # --------------------------------------------------------
    # Generate assigned papers
    # --------------------------------------------------------

    already_complete = 0
    completed_now = 0
    incomplete_now = 0
    generation_errors = 0

    run_start = time.time()

    for paper_index, paper_id in enumerate(
        assigned_papers,
        start=1,
    ):
        records = by_paper[
            paper_id
        ]

        expected_qas = count_qas(
            records
        )

        output_path = (
            output_dir
            / f"{paper_id}.json"
        )

        if paper_is_complete(
            output_path,
            expected_qas,
        ):
            already_complete += 1

            print(
                f"[rank {rank}] "
                f"SKIP COMPLETE "
                f"{paper_id}",
                flush=True,
            )

            continue

        result = evaluate_paper(
            model=model,
            processor=processor,
            device=device,
            paper_id=paper_id,
            records=records,
            output_path=output_path,
            rank=rank,
            args=args,
        )

        if result["status"] == "complete":
            completed_now += 1
        else:
            incomplete_now += 1

        generation_errors += (
            result["summary"][
                "generation_errors"
            ]
        )

        print(
            f"[rank {rank}] "
            f"paper_progress="
            f"{paper_index}/"
            f"{len(assigned_papers)}",
            flush=True,
        )

    elapsed = (
        time.time() - run_start
    )

    rank_summary = {
        "rank":
            rank,

        "world_size":
            world_size,

        "assigned_papers":
            len(assigned_papers),

        "assigned_qas":
            assigned_qas,

        "already_complete":
            already_complete,

        "completed_now":
            completed_now,

        "incomplete_now":
            incomplete_now,

        "generation_errors":
            generation_errors,

        "elapsed_seconds":
            elapsed,

        "finished":
            timestamp(),
    }

    state_path = (
        state_dir
        / f"generation_rank_{rank:02d}.json"
    )

    state_path.write_text(
        json.dumps(
            rank_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        f"[rank {rank}] COMPLETE\n"
        + json.dumps(
            rank_summary,
            indent=2,
        ),
        flush=True,
    )

    if (
        generation_errors > 0
        or incomplete_now > 0
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
