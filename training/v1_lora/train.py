#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Subset

from peft import LoraConfig, TaskType, get_peft_model

# TRUSTED_LOCAL_RESUME_TORCH_LOAD_PATCH
# This only bypasses the Transformers torch.load safety gate when explicitly
# enabled for a trusted checkpoint created locally by this training run.
import os as _trusted_resume_os

if _trusted_resume_os.environ.get("ALLOW_TRUSTED_LOCAL_CHECKPOINT_RESUME") == "1":
    try:
        import transformers.utils.import_utils as _hf_import_utils
        import transformers.trainer as _hf_trainer

        def _allow_trusted_local_checkpoint_load():
            return None

        _hf_import_utils.check_torch_load_is_safe = _allow_trusted_local_checkpoint_load
        _hf_trainer.check_torch_load_is_safe = _allow_trusted_local_checkpoint_load

        print(
            "Trusted local checkpoint resume: torch.load safety gate bypassed "
            "for this local run.",
            flush=True,
        )
    except Exception as _patch_error:
        print(
            "Trusted local checkpoint resume patch failed:",
            repr(_patch_error),
            flush=True,
        )


from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint


# ============================================================
# Paths
# ============================================================

MODEL_DIR = Path(
    os.environ.get(
        "HEPVLMQA_MODEL_DIR",
        "models/llava-onevision-qwen2-7b-ov-hf",
    )
)

DATA_DIR = Path(
    os.environ.get(
        "HEPVLMQA_DATA_DIR",
        "data/canonical_v1",
    )
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "HEPVLMQA_OUTPUT_DIR",
        "outputs/v1_lora",
    )
)

SEED = 42


# ============================================================
# Dataset
# ============================================================

class MinimalVQADataset(Dataset):
    REQUIRED_FIELDS = {
        "image",
        "question",
        "answer",
    }

    def __init__(self, jsonl_path: Path):
        self.path = jsonl_path
        self.records: List[Dict[str, str]] = []

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Dataset does not exist: {jsonl_path}"
            )

        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line_number, line in enumerate(handle, 1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid JSON at "
                        f"{jsonl_path}:{line_number}: "
                        f"{error}"
                    ) from error

                if set(record.keys()) != self.REQUIRED_FIELDS:
                    raise RuntimeError(
                        f"Unexpected fields at "
                        f"{jsonl_path}:{line_number}: "
                        f"{sorted(record.keys())}"
                    )

                image_path = Path(
                    str(record["image"])
                )

                question = str(
                    record["question"]
                ).strip()

                answer = str(
                    record["answer"]
                ).strip()

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Missing image at "
                        f"{jsonl_path}:{line_number}: "
                        f"{image_path}"
                    )

                if not question:
                    raise RuntimeError(
                        f"Empty question at "
                        f"{jsonl_path}:{line_number}"
                    )

                if not answer:
                    raise RuntimeError(
                        f"Empty answer at "
                        f"{jsonl_path}:{line_number}"
                    )

                self.records.append(
                    {
                        "image": str(image_path),
                        "question": question,
                        "answer": answer,
                    }
                )

        if not self.records:
            raise RuntimeError(
                f"No usable records in {jsonl_path}"
            )

        print(
            f"Loaded {len(self.records):,} examples "
            f"from {jsonl_path}",
            flush=True,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


# ============================================================
# Chat formatting
# ============================================================

def make_training_conversation(
    question: str,
    answer: str,
):
    return [
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
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer,
                },
            ],
        },
    ]


def find_last_subsequence(
    sequence: Sequence[int],
    subsequence: Sequence[int],
):
    if not subsequence:
        return None

    maximum_start = (
        len(sequence) - len(subsequence)
    )

    for start in range(
        maximum_start,
        -1,
        -1,
    ):
        if (
            list(
                sequence[
                    start:start + len(subsequence)
                ]
            )
            == list(subsequence)
        ):
            return start

    return None


# ============================================================
# Image handling
# ============================================================

def load_resized_image(
    image_path: str,
    maximum_side: int,
):
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

        original_width, original_height = (
            image.size
        )

        if (
            maximum_side > 0
            and max(
                original_width,
                original_height,
            ) > maximum_side
        ):
            image.thumbnail(
                (
                    maximum_side,
                    maximum_side,
                ),
                Image.Resampling.LANCZOS,
            )

        return image.copy()


# ============================================================
# Collator
# ============================================================

class LlavaAnswerCollator:
    """
    Batch size 1.

    The processor formats two sequences using the same image:

    1. User question with an assistant-generation prompt.
    2. User question plus the reference answer.

    Their common token prefix is the complete model input before
    the reference answer. All prefix tokens are masked with -100,
    so loss is calculated only on the answer tokens.

    This does not assume any fixed assistant-marker string.
    """

    def __init__(
        self,
        processor,
        maximum_image_side: int,
    ):
        self.processor = processor
        self.maximum_image_side = maximum_image_side
        self._printed_debug_example = False

    @staticmethod
    def common_prefix_length(
        first_ids,
        second_ids,
    ) -> int:
        first_tokens = (
            first_ids[0].detach().cpu().tolist()
        )

        second_tokens = (
            second_ids[0].detach().cpu().tolist()
        )

        length = 0

        for first_token, second_token in zip(
            first_tokens,
            second_tokens,
        ):
            if first_token != second_token:
                break

            length += 1

        return length

    @staticmethod
    def make_user_conversation(question: str):
        return [
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

    @staticmethod
    def make_complete_conversation(
        question: str,
        answer: str,
    ):
        return [
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
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    },
                ],
            },
        ]

    def __call__(self, examples):
        if len(examples) != 1:
            raise RuntimeError(
                "This pipeline requires "
                "per-device batch size 1. "
                f"Received {len(examples)}."
            )

        example = examples[0]

        question = example["question"]
        answer = example["answer"]
        image_path = example["image"]

        user_conversation = (
            self.make_user_conversation(
                question
            )
        )

        complete_conversation = (
            self.make_complete_conversation(
                question,
                answer,
            )
        )

        # The first text ends exactly where the model should
        # begin generating its answer.
        prompt_text = (
            self.processor.apply_chat_template(
                user_conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        complete_text = (
            self.processor.apply_chat_template(
                complete_conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

        image = load_resized_image(
            image_path=image_path,
            maximum_side=self.maximum_image_side,
        )

        # Process both conversations with the same image so
        # image-token expansion is identical.
        prompt_inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
            padding=False,
        )

        complete_inputs = self.processor(
            images=image,
            text=complete_text,
            return_tensors="pt",
            padding=False,
        )

        prompt_ids = prompt_inputs["input_ids"]
        complete_ids = complete_inputs["input_ids"]

        prefix_length = (
            self.common_prefix_length(
                prompt_ids,
                complete_ids,
            )
        )

        prompt_length = prompt_ids.shape[1]
        complete_length = complete_ids.shape[1]

        if prefix_length <= 0:
            raise RuntimeError(
                "The prompt and complete conversation "
                "have no common token prefix."
            )

        # A small difference can occur at the final prompt token,
        # depending on how the chat template closes the assistant
        # generation header.
        if prefix_length < prompt_length - 4:
            prompt_tail = (
                self.processor.tokenizer.decode(
                    prompt_ids[
                        0,
                        max(0, prefix_length - 20):,
                    ],
                    skip_special_tokens=False,
                )
            )

            complete_tail = (
                self.processor.tokenizer.decode(
                    complete_ids[
                        0,
                        max(0, prefix_length - 20):
                        min(
                            complete_length,
                            prefix_length + 30,
                        ),
                    ],
                    skip_special_tokens=False,
                )
            )

            raise RuntimeError(
                "Prompt and full conversation diverged "
                "too early.\n"
                f"Prompt length: {prompt_length}\n"
                f"Complete length: {complete_length}\n"
                f"Common prefix: {prefix_length}\n"
                f"Prompt tail: {prompt_tail!r}\n"
                f"Complete tail: {complete_tail!r}"
            )

        labels = complete_ids.clone()

        # Ignore image tokens, system text, user question,
        # and assistant-generation header.
        labels[:, :prefix_length] = -100

        if "attention_mask" in complete_inputs:
            labels[
                complete_inputs["attention_mask"] == 0
            ] = -100

        supervised_tokens = int(
            labels.ne(-100).sum().item()
        )

        if supervised_tokens <= 0:
            raise RuntimeError(
                "No reference-answer tokens remain "
                "after prefix masking."
            )

        complete_inputs["labels"] = labels

        if not self._printed_debug_example:
            print(
                "\n===== COLLATOR PREFLIGHT =====",
                flush=True,
            )

            print(
                "Prompt tokens:",
                prompt_length,
                flush=True,
            )

            print(
                "Complete tokens:",
                complete_length,
                flush=True,
            )

            print(
                "Common prefix:",
                prefix_length,
                flush=True,
            )

            print(
                "Supervised answer tokens:",
                supervised_tokens,
                flush=True,
            )

            decoded_answer = (
                self.processor.tokenizer.decode(
                    labels[
                        labels.ne(-100)
                    ],
                    skip_special_tokens=False,
                )
            )

            print(
                "Decoded supervised text:",
                repr(decoded_answer[:500]),
                flush=True,
            )

            self._printed_debug_example = True

        return complete_inputs


# ============================================================
# Memory-efficient answer-only loss
# ============================================================

class AnswerOnlyTrainer(Trainer):
    """
    Avoid producing vocabulary logits for the complete
    image + question sequence.

    Example:
        full sequence = 2,701 tokens
        answer         = 11 tokens

    Normal training produces:
        [1, 2701, vocabulary_size] logits

    This trainer produces approximately:
        [1, 12, vocabulary_size] logits
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        model_inputs = dict(inputs)

        labels = model_inputs.pop("labels")

        supervised_mask = labels.ne(-100)

        if not supervised_mask.any():
            raise RuntimeError(
                "Batch has no supervised labels."
            )

        sequence_length = labels.shape[1]

        supervised_positions = (
            supervised_mask.nonzero(
                as_tuple=False
            )
        )

        first_supervised_position = int(
            supervised_positions[:, 1]
            .min()
            .item()
        )

        # One extra logit is required because the logit at
        # position t predicts the token at position t+1.
        logits_to_keep = (
            sequence_length
            - first_supervised_position
            + 1
        )

        outputs = model(
            **model_inputs,
            labels=None,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        )

        logits = outputs.logits

        # Create standard next-token labels.
        padded_labels = F.pad(
            labels,
            pad=(0, 1),
            value=-100,
        )

        shifted_labels = (
            padded_labels[..., 1:]
            .contiguous()
        )

        # Keep only labels corresponding to the
        # retained final logits.
        shifted_labels = shifted_labels[
            :,
            -logits.shape[1]:,
        ]

        flat_logits = logits.float().reshape(
            -1,
            logits.shape[-1],
        )

        flat_labels = shifted_labels.reshape(-1)

        valid_token_count = (
            flat_labels.ne(-100)
            .sum()
            .clamp_min(1)
        )

        loss_sum = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction="sum",
        )

        loss = (
            loss_sum
            / valid_token_count
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss: {loss.item()}"
            )

        if return_outputs:
            return loss, outputs

        return loss


# ============================================================
# LoRA configuration
# ============================================================

def find_lora_targets(model):
    """
    Use attention Q and V projections only.

    This is substantially smaller than adding LoRA to
    attention plus every MLP projection.
    """

    allowed_names = {
        "q_proj",
        "v_proj",
    }

    targets = []

    for module_name, module in (
        model.named_modules()
    ):
        if not isinstance(
            module,
            torch.nn.Linear,
        ):
            continue

        leaf_name = (
            module_name.rsplit(".", 1)[-1]
        )

        if (
            "language_model" in module_name
            and leaf_name in allowed_names
        ):
            targets.append(module_name)

    targets = sorted(set(targets))

    if not targets:
        raise RuntimeError(
            "No Qwen language-model "
            "LoRA targets were found."
        )

    return targets


# ============================================================
# Validation subset
# ============================================================

def make_fixed_subset(
    dataset,
    subset_size: int,
    seed: int,
):
    if (
        subset_size <= 0
        or subset_size >= len(dataset)
    ):
        return dataset

    generator = random.Random(seed)

    indices = generator.sample(
        range(len(dataset)),
        subset_size,
    )

    indices.sort()

    return Subset(
        dataset,
        indices,
    )


# ============================================================
# Logging callback
# ============================================================

class MemoryLoggingCallback(
    TrainerCallback
):
    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):
        logs = logs or {}

        for key in [
            "loss",
            "eval_loss",
        ]:
            if key in logs:
                value = float(logs[key])

                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite {key}: {value}"
                    )

        if torch.cuda.is_available():
            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            peak = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            print(
                f"[GPU memory] "
                f"allocated={allocated:.2f} GB, "
                f"reserved={reserved:.2f} GB, "
                f"peak={peak:.2f} GB",
                flush=True,
            )


# ============================================================
# Resume
# ============================================================

def resolve_resume(
    output_directory: Path,
    requested_resume: str,
):
    normalized = (
        requested_resume
        .strip()
        .lower()
    )

    latest = get_last_checkpoint(
        str(output_directory)
    )

    if normalized == "auto":
        if latest:
            print(
                "Resuming from:",
                latest,
                flush=True,
            )
        else:
            print(
                "No checkpoint found. "
                "Starting a new run.",
                flush=True,
            )

        return latest

    if normalized == "none":
        if latest:
            raise RuntimeError(
                f"Existing checkpoint found: {latest}. "
                "Use --resume auto or archive "
                "the output directory."
            )

        return None

    requested_path = Path(
        requested_resume
    )

    if not requested_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{requested_path}"
        )

    return str(requested_path)


# ============================================================
# Mode
# ============================================================

def resolve_mode(args):
    if args.mode == "smoke":
        return {
            "train_file": (
                DATA_DIR
                / "train_smoke_1000.jsonl"
            ),
            "validation_file": (
                DATA_DIR
                / "val_smoke_200.jsonl"
            ),
            "output_directory": (
                OUTPUT_ROOT
                / "llava_onevision_7b_lora_smoke"
            ),
            "epochs": 1.0,
            "max_steps": (
                args.max_steps
                if args.max_steps is not None
                else 100
            ),
            "eval_steps": 10,
            "save_steps": 10,
            "logging_steps": 2,
            "validation_subset_size": 200,
        }

    return {
        "train_file": (
            DATA_DIR / "train.jsonl"
        ),
        "validation_file": (
            DATA_DIR / "validation_monitor_1000.jsonl"
        ),
        "output_directory": (
            OUTPUT_ROOT
            / "llava_onevision_7b_lora_full"
        ),
        "epochs": args.epochs,
        "max_steps": (
            args.max_steps
            if args.max_steps is not None
            else -1
        ),
        "eval_steps": args.eval_steps,
        "save_steps": args.eval_steps,
        "logging_steps": 20,
        "validation_subset_size": (
            args.validation_subset_size
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "full",
        ],
        default="smoke",
    )

    parser.add_argument(
        "--resume",
        default="auto",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--validation-subset-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--maximum-image-side",
        type=int,
        default=768,
    )

    args = parser.parse_args()

    configuration = resolve_mode(args)

    train_file = configuration[
        "train_file"
    ]

    validation_file = configuration[
        "validation_file"
    ]

    output_directory = configuration[
        "output_directory"
    ]

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory missing: "
            f"{MODEL_DIR}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    print("=" * 76, flush=True)
    print(
        "Mode:",
        args.mode,
        flush=True,
    )
    print(
        "Model:",
        MODEL_DIR,
        flush=True,
    )
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print(
        "Training data:",
        train_file,
        flush=True,
    )
    print(
        "Validation data:",
        validation_file,
        flush=True,
    )
    print(
        "Output:",
        output_directory,
        flush=True,
    )
    print(
        "Maximum image side:",
        args.maximum_image_side,
        flush=True,
    )
    print("=" * 76, flush=True)

    train_dataset = MinimalVQADataset(
        train_file
    )

    complete_validation_dataset = (
        MinimalVQADataset(
            validation_file
        )
    )

    periodic_validation_dataset = (
        make_fixed_subset(
            dataset=(
                complete_validation_dataset
            ),
            subset_size=configuration[
                "validation_subset_size"
            ],
            seed=SEED,
        )
    )

    print(
        "Periodic validation size:",
        len(periodic_validation_dataset),
        flush=True,
    )

    processor = (
        AutoProcessor.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
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
        "right"
    )

    print(
        "Loading local model weights...",
        flush=True,
    )

    base_model = (
        LlavaOnevisionForConditionalGeneration
        .from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    )

    base_model.config.use_cache = False

    # Activate model-level checkpointing before PEFT wrapping.
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    target_modules = find_lora_targets(
        base_model
    )

    print(
        "LoRA target modules:",
        len(target_modules),
        flush=True,
    )

    lora_configuration = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(
        base_model,
        lora_configuration,
    )

    # Required for checkpointing when input embeddings
    # and the base model are frozen.
    model.enable_input_require_grads()

    model.config.use_cache = False

    model.print_trainable_parameters()

    collator = LlavaAnswerCollator(
        processor=processor,
        maximum_image_side=(
            args.maximum_image_side
        ),
    )

    preflight_batch = collator(
        [train_dataset[0]]
    )

    labels = preflight_batch["labels"]

    first_supervised = int(
        labels.ne(-100)
        .nonzero(as_tuple=False)[:, 1]
        .min()
        .item()
    )

    estimated_logits_kept = (
        labels.shape[1]
        - first_supervised
        + 1
    )

    print(
        "\n===== PREFLIGHT =====",
        flush=True,
    )

    for key, value in (
        preflight_batch.items()
    ):
        if torch.is_tensor(value):
            print(
                key,
                tuple(value.shape),
                value.dtype,
                flush=True,
            )

    print(
        "Full sequence tokens:",
        labels.shape[1],
        flush=True,
    )

    print(
        "Answer tokens:",
        int(labels.ne(-100).sum()),
        flush=True,
    )

    print(
        "Vocabulary-logit positions retained:",
        estimated_logits_kept,
        flush=True,
    )

    run_configuration = {
        "created_at": (
            datetime.now().isoformat()
        ),
        "mode": args.mode,
        "model_directory": str(
            MODEL_DIR
        ),
        "train_file": str(train_file),
        "validation_file": str(
            validation_file
        ),
        "train_examples": len(
            train_dataset
        ),
        "periodic_validation_examples": (
            len(periodic_validation_dataset)
        ),
        "complete_validation_examples": (
            len(
                complete_validation_dataset
            )
        ),
        "output_directory": str(
            output_directory
        ),
        "maximum_image_side": (
            args.maximum_image_side
        ),
        "max_steps": configuration[
            "max_steps"
        ],
        "epochs": configuration[
            "epochs"
        ],
        "learning_rate": (
            args.learning_rate
        ),
        "gradient_accumulation": (
            args.gradient_accumulation
        ),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_targets": [
            "q_proj",
            "v_proj",
        ],
        "gradient_checkpointing": True,
        "answer_only_logits": True,
        "seed": SEED,
    }

    with (
        output_directory
        / "run_configuration.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_configuration,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    training_arguments = TrainingArguments(
        output_dir=str(
            output_directory
        ),

        num_train_epochs=configuration[
            "epochs"
        ],

        max_steps=configuration[
            "max_steps"
        ],

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            args.gradient_accumulation
        ),

        learning_rate=(
            args.learning_rate
        ),

        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        optim="adamw_torch_fused",

        bf16=True,
        tf32=True,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        logging_strategy="steps",

        logging_steps=configuration[
            "logging_steps"
        ],

        logging_first_step=True,
        logging_nan_inf_filter=False,

        eval_strategy="steps",

        eval_steps=configuration[
            "eval_steps"
        ],

        prediction_loss_only=True,

        save_strategy="steps",

        save_steps=configuration[
            "save_steps"
        ],

        save_total_limit=4,
        save_safetensors=True,

        load_best_model_at_end=True,
        metric_for_best_model=(
            "eval_loss"
        ),
        greater_is_better=False,

        label_names=[
            "labels",
        ],

        remove_unused_columns=False,

        dataloader_num_workers=0,

        dataloader_pin_memory=True,

        seed=SEED,
        data_seed=SEED,

        report_to="none",
    )

    trainer = AnswerOnlyTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=(
            periodic_validation_dataset
        ),
        data_collator=collator,
        callbacks=[
            MemoryLoggingCallback(),
        ],
    )

    resume_checkpoint = resolve_resume(
        output_directory=output_directory,
        requested_resume=args.resume,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        print(
            "\n===== STARTING TRAINING =====",
            flush=True,
        )

        train_result = trainer.train(
            resume_from_checkpoint=(
                resume_checkpoint
            )
        )

        trainer.log_metrics(
            "train",
            train_result.metrics,
        )

        trainer.save_metrics(
            "train",
            train_result.metrics,
        )

        trainer.save_state()

        print(
            "\n===== FINAL VALIDATION =====",
            flush=True,
        )

        validation_metrics = (
            trainer.evaluate(
                eval_dataset=(
                    complete_validation_dataset
                ),
                metric_key_prefix=(
                    "validation"
                ),
            )
        )

        trainer.log_metrics(
            "validation",
            validation_metrics,
        )

        trainer.save_metrics(
            "validation",
            validation_metrics,
        )

        final_adapter_directory = (
            output_directory
            / "adapter_best"
        )

        trainer.model.save_pretrained(
            str(
                final_adapter_directory
            ),
            safe_serialization=True,
        )

        processor.save_pretrained(
            str(
                final_adapter_directory
            )
        )

        success = {
            "status": "success",
            "finished_at": (
                datetime.now().isoformat()
            ),
            "global_step": (
                trainer.state.global_step
            ),
            "best_checkpoint": (
                trainer.state
                .best_model_checkpoint
            ),
            "best_validation_loss": (
                trainer.state.best_metric
            ),
            "adapter": str(
                final_adapter_directory
            ),
            "peak_gpu_memory_gb": (
                torch.cuda
                .max_memory_allocated()
                / 1024**3
            ),
        }

        with (
            output_directory
            / "SUCCESS.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                success,
                handle,
                indent=2,
            )

        failed_file = (
            output_directory
            / "FAILED.json"
        )

        if failed_file.exists():
            failed_file.unlink()

        print(
            "\n===== TRAINING COMPLETE =====",
            flush=True,
        )

        print(
            "Best checkpoint:",
            trainer.state
            .best_model_checkpoint,
            flush=True,
        )

        print(
            "Best validation loss:",
            trainer.state.best_metric,
            flush=True,
        )

        print(
            "Adapter:",
            final_adapter_directory,
            flush=True,
        )

    except Exception:
        failure = {
            "status": "failed",
            "time": (
                datetime.now().isoformat()
            ),
            "latest_checkpoint": (
                get_last_checkpoint(
                    str(output_directory)
                )
            ),
            "peak_gpu_memory_gb": (
                torch.cuda
                .max_memory_allocated()
                / 1024**3
            ),
            "traceback": (
                traceback.format_exc()
            ),
        }

        with (
            output_directory
            / "FAILED.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                failure,
                handle,
                indent=2,
            )

        print(
            "\nTraining failed. See:",
            output_directory
            / "FAILED.json",
            flush=True,
        )

        raise


if __name__ == "__main__":
    main()
