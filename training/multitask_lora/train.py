#!/usr/bin/env python3

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from PIL import Image

from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)


# ============================================================
# IO
# ============================================================

def read_json(path):
    return json.loads(Path(path).read_text())


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def save_torch_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(str(path) + ".tmp")

    torch.save(obj, tmp)
    os.replace(tmp, path)


def count_nonempty_lines(path):
    n = 0

    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1

    return n


def load_rank_shard(path, rank, world_size):
    rows = []

    logical_index = 0

    with Path(path).open(encoding="utf-8") as f:
        for line in f:

            if not line.strip():
                continue

            if logical_index % world_size == rank:
                rows.append(json.loads(line))

            logical_index += 1

    return rows


# ============================================================
# DISTRIBUTED
# ============================================================

def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl"
    )

    device = torch.device(
        "cuda",
        local_rank,
    )

    return local_rank, rank, world_size, device


def distributed_barrier(device):
    dist.barrier(
        device_ids=[device.index]
    )


def reduce_sum_int(value, device):
    x = torch.tensor(
        [int(value)],
        dtype=torch.long,
        device=device,
    )

    dist.all_reduce(
        x,
        op=dist.ReduceOp.SUM,
    )

    return int(x.item())


def reduce_sum_float(value, device):
    x = torch.tensor(
        [float(value)],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        x,
        op=dist.ReduceOp.SUM,
    )

    return float(x.item())


# ============================================================
# RNG / RESUME
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():

        for key, value in state.items():

            if torch.is_tensor(value):
                state[key] = value.to(device)


# ============================================================
# LoRA
# ============================================================

def find_lora_targets(model, suffixes):
    targets = []

    for name, module in model.named_modules():

        if "language_model" not in name:
            continue

        if not isinstance(
            module,
            torch.nn.Linear,
        ):
            continue

        if any(
            name.endswith(suffix)
            for suffix in suffixes
        ):
            targets.append(name)

    return sorted(targets)


def load_model_and_processor(
    config,
    device,
    resume_adapter=None,
):
    model_path = config["model_path"]

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    processor.tokenizer.padding_side = "right"

    base_model = (
        LlavaOnevisionForConditionalGeneration
        .from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    )

    base_model.config.use_cache = False

    # Freeze every original parameter.
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    if config["gradient_checkpointing"]:
        base_model.gradient_checkpointing_enable()

        if hasattr(
            base_model,
            "enable_input_require_grads",
        ):
            base_model.enable_input_require_grads()

    if resume_adapter is None:

        targets = find_lora_targets(
            base_model,
            config["target_module_suffixes"],
        )

        if len(targets) != 196:
            raise RuntimeError(
                "Expected 196 language-model LoRA "
                f"target modules; found {len(targets)}"
            )

        peft_config = LoraConfig(
            r=int(config["lora_rank"]),
            lora_alpha=int(
                config["lora_alpha"]
            ),
            lora_dropout=float(
                config["lora_dropout"]
            ),
            bias=config["lora_bias"],
            target_modules=targets,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(
            base_model,
            peft_config,
        )

    else:

        # MANUAL_PEFT_RESUME
        # DDP-only experiment: construct the LoRA adapter from its saved
        # configuration, then load the adapter tensors manually. This avoids
        # PEFT 0.19.1's incompatible Tensor-Parallel checkpoint path.

        from peft import PeftConfig, get_peft_model
        from safetensors.torch import load_file
        import peft.utils.save_and_load as _peft_save_load

        peft_config = PeftConfig.from_pretrained(
            str(resume_adapter)
        )

        model = get_peft_model(
            base_model,
            peft_config,
        )

        # Disable the unused TP sharding helper for this DDP-only run.
        _peft_save_load._maybe_shard_state_dict_for_tp = (
            lambda model, state_dict, adapter_name: None
        )

        adapter_weights = load_file(
            str(
                Path(resume_adapter)
                / "adapter_model.safetensors"
            )
        )

        load_result = _peft_save_load.set_peft_model_state_dict(
            model,
            adapter_weights,
            adapter_name="default",
        )

        if hasattr(load_result, "missing_keys"):
            unexpected = list(
                getattr(load_result, "unexpected_keys", [])
            )

            bad_unexpected = [
                x for x in unexpected
                if "lora_" in x
            ]

            if bad_unexpected:
                raise RuntimeError(
                    "Unexpected LoRA keys during resume: "
                    + str(bad_unexpected[:20])
                )

    # Ensure only LoRA parameters can train.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            "lora_" in name
        )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters: {trainable:,}",
        flush=True,
    )

    print(
        f"Total parameters: {total:,}",
        flush=True,
    )

    # Exact adapter size expected for this configuration.
    if trainable != 80_740_352:
        raise RuntimeError(
            "Unexpected LoRA trainable parameter count: "
            f"{trainable:,}; expected 80,740,352"
        )

    model.to(device)

    return model, processor


# ============================================================
# PROMPT PREPARATION
# ============================================================

def build_prompt_texts(
    processor,
    question,
    answer,
):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": str(question),
                },
            ],
        }
    ]

    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": str(answer),
                }
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return prompt_text, full_text


def prepare_qa_chunk(
    processor,
    image,
    qas,
    max_length,
):
    prompt_texts = []
    full_texts = []
    qa_types = []

    for qa in qas:

        prompt_text, full_text = (
            build_prompt_texts(
                processor,
                qa["question"],
                qa["answer"],
            )
        )

        prompt_texts.append(prompt_text)
        full_texts.append(full_text)
        qa_types.append(qa["qa_type"])

    # IMPORTANT:
    # one image entry for every QA.
    # No visual representation is shared.
    repeated_images = [
        image.copy()
        for _ in qas
    ]

    full = processor(
        images=repeated_images,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    prompt = processor(
        images=repeated_images,
        text=prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = full["input_ids"].clone()

    full_attention = full["attention_mask"]
    prompt_attention = prompt["attention_mask"]

    valid = []

    for i in range(labels.shape[0]):

        full_len = int(
            full_attention[i].sum().item()
        )

        prompt_len = int(
            prompt_attention[i].sum().item()
        )

        prompt_len = min(
            prompt_len,
            full_len,
        )

        labels[i, :prompt_len] = -100

        labels[
            i,
            full_attention[i] == 0
        ] = -100

        supervised = int(
            (labels[i] != -100)
            .sum()
            .item()
        )

        valid.append(
            supervised > 0
        )

    return {
        "inputs": full,
        "labels": labels,
        "qa_types": qa_types,
        "valid": valid,
    }


def prepare_group(
    processor,
    group,
    max_length,
    micro_batch_size,
):
    image_path = Path(group["image_path"])

    if not image_path.is_file():
        raise FileNotFoundError(
            str(image_path)
        )

    qas = group.get("qas", [])

    if not qas:
        return []

    image = Image.open(
        image_path
    ).convert("RGB")

    chunks = []

    for start in range(
        0,
        len(qas),
        micro_batch_size,
    ):

        qa_chunk = qas[
            start:
            start + micro_batch_size
        ]

        prepared = prepare_qa_chunk(
            processor,
            image,
            qa_chunk,
            max_length,
        )

        chunks.append(prepared)

    image.close()

    return chunks


def move_inputs_to_device(
    inputs,
    device,
):
    output = {}

    for key, value in inputs.items():

        if torch.is_tensor(value):
            output[key] = value.to(
                device,
                non_blocking=True,
            )

        else:
            output[key] = value

    return output


# ============================================================
# LOSS
# ============================================================

def per_sample_answer_loss(
    logits,
    labels,
):
    shift_logits = (
        logits[:, :-1, :]
        .contiguous()
        .float()
    )

    shift_labels = (
        labels[:, 1:]
        .contiguous()
    )

    vocab_size = shift_logits.shape[-1]

    token_losses = F.cross_entropy(
        shift_logits.reshape(
            -1,
            vocab_size,
        ),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )

    token_losses = token_losses.view(
        shift_labels.shape
    )

    valid_tokens = (
        shift_labels != -100
    )

    denominator = (
        valid_tokens.sum(dim=1)
        .clamp_min(1)
    )

    losses = (
        (
            token_losses
            * valid_tokens
        )
        .sum(dim=1)
        / denominator
    )

    return losses


def task_weights_from_counts(task_counts):
    # The common scale factor is irrelevant because
    # each group is normalized by its total task weight.
    return {
        task: 1.0 / float(count)
        for task, count
        in task_counts.items()
    }


# ============================================================
# FORWARD
# ============================================================

def forward_full_multimodal(
    model,
    prepared,
    device,
):
    inputs = move_inputs_to_device(
        prepared["inputs"],
        device,
    )

    labels = prepared[
        "labels"
    ].to(
        device,
        non_blocking=True,
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):

        outputs = model(
            **inputs,
            use_cache=False,
        )

        losses = per_sample_answer_loss(
            outputs.logits,
            labels,
        )

    return losses


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def run_validation(
    model,
    processor,
    config,
    device,
):
    model.eval()

    task_sum = defaultdict(float)
    task_n = Counter()

    groups_seen = 0

    max_groups = int(
        config["validation_max_groups"]
    )

    micro_batch_size = int(
        config["qa_micro_batch_size"]
    )

    max_length = int(
        config["max_sequence_length"]
    )

    with Path(
        config["validation_data"]
    ).open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            if groups_seen >= max_groups:
                break

            group = json.loads(line)

            groups_seen += 1

            try:
                chunks = prepare_group(
                    processor,
                    group,
                    max_length,
                    micro_batch_size,
                )

            except Exception:
                continue

            for prepared in chunks:

                if not any(
                    prepared["valid"]
                ):
                    continue

                losses = (
                    forward_full_multimodal(
                        model,
                        prepared,
                        device,
                    )
                )

                for i, loss in enumerate(
                    losses
                ):

                    if not prepared["valid"][i]:
                        continue

                    qa_type = (
                        prepared[
                            "qa_types"
                        ][i]
                    )

                    task_sum[qa_type] += float(
                        loss.item()
                    )

                    task_n[qa_type] += 1

    per_type = {}

    for task in sorted(task_n):

        per_type[task] = (
            task_sum[task]
            / task_n[task]
        )

    equal_type_loss = (
        sum(per_type.values())
        / len(per_type)
        if per_type
        else float("inf")
    )

    model.train()

    return {
        "equal_type_loss": equal_type_loss,
        "per_type_loss": per_type,
        "per_type_count": dict(task_n),
        "validation_groups": groups_seen,
    }


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def checkpoint_parent(config):
    return (
        Path(config["checkpoint_root"])
        / config["run_name"]
    )


def find_latest_checkpoint(config):
    pointer = (
        checkpoint_parent(config)
        / "latest_checkpoint.txt"
    )

    if not pointer.is_file():
        return None

    value = pointer.read_text().strip()

    if not value:
        return None

    path = Path(value)

    if not path.is_dir():
        return None

    if not (
        path / "training_state.pt"
    ).is_file():
        return None

    if not (
        path / "adapter"
    ).is_dir():
        return None

    return path


def save_checkpoint(
    ddp_model,
    optimizer,
    scheduler,
    config,
    global_step,
    local_cursor,
    best_validation_loss,
    device,
    rank,
    world_size,
):
    cursor_list = [
        None
        for _ in range(world_size)
    ]

    dist.all_gather_object(
        cursor_list,
        int(local_cursor),
    )

    local_rng = capture_rng_state()

    rng_states = [
        None
        for _ in range(world_size)
    ]

    dist.all_gather_object(
        rng_states,
        local_rng,
    )

    if rank != 0:
        return

    parent = checkpoint_parent(
        config
    )

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = (
        parent
        / f"checkpoint-step-{global_step:06d}"
    )

    checkpoint.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter_dir = (
        checkpoint
        / "adapter"
    )

    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    ddp_model.module.save_pretrained(
        adapter_dir
    )

    state = {
        "global_step": global_step,
        "cursors": cursor_list,
        "best_validation_loss":
            best_validation_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_states": rng_states,
    }

    save_torch_atomic(
        checkpoint / "training_state.pt",
        state,
    )

    write_json_atomic(
        checkpoint / "checkpoint_summary.json",
        {
            "global_step": global_step,
            "cursors": cursor_list,
            "best_validation_loss":
                best_validation_loss,
            "training_strategy":
                "pooled_repeated_image_per_qa",
            "reuse_visual_features": False,
            "use_vision_cache": False,
        },
    )

    pointer = (
        parent
        / "latest_checkpoint.txt"
    )

    pointer.write_text(
        str(checkpoint) + "\n"
    )

    # Also maintain a convenient latest state file.
    weight_dir = (
        Path(config["weight_root"])
        / config["run_name"]
    )

    weight_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        checkpoint / "training_state.pt",
        weight_dir
        / "training_state_latest.pt",
    )

    keep = int(
        config["keep_last_checkpoints"]
    )

    all_checkpoints = sorted(
        parent.glob(
            "checkpoint-step-*"
        )
    )

    while len(all_checkpoints) > keep:

        old = all_checkpoints.pop(0)

        shutil.rmtree(
            old,
            ignore_errors=True,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    config = read_json(
        args.config
    )

    (
        local_rank,
        rank,
        world_size,
        device,
    ) = init_distributed()

    if world_size != int(
        config["expected_world_size"]
    ):
        raise RuntimeError(
            "Unexpected world size: "
            f"{world_size}"
        )

    seed = int(config["seed"])

    set_seed(
        seed + rank
    )

    run_dir = (
        Path(config["output_root"])
        / config["run_name"]
    )

    adapter_run_dir = (
        Path(config["adapter_root"])
        / config["run_name"]
    )

    weight_run_dir = (
        Path(config["weight_root"])
        / config["run_name"]
    )

    if rank == 0:

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        adapter_run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        weight_run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_parent(
            config
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        write_json_atomic(
            run_dir
            / "training_configuration.json",
            config,
        )

    distributed_barrier(
        device
    )

    resume_checkpoint = None

    if config.get(
        "resume",
        "auto",
    ) == "auto":

        resume_checkpoint = (
            find_latest_checkpoint(
                config
            )
        )

    resume_adapter = None

    if resume_checkpoint is not None:
        resume_adapter = (
            resume_checkpoint
            / "adapter"
        )

    model, processor = (
        load_model_and_processor(
            config,
            device,
            resume_adapter,
        )
    )

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    trainable_parameters = [
        parameter
        for parameter
        in ddp_model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(
            config["learning_rate"]
        ),
        weight_decay=float(
            config["weight_decay"]
        ),
    )

    dataset_size = count_nonempty_lines(
        config["train_data"]
    )

    rank_groups = load_rank_shard(
        config["train_data"],
        rank,
        world_size,
    )

    expected_max_steps = math.ceil(
        dataset_size / world_size
    )

    configured_max_steps = int(
        config["max_optimizer_steps"]
    )

    if configured_max_steps > 0:
        scheduler_total_steps = min(
            expected_max_steps,
            configured_max_steps,
        )
    else:
        scheduler_total_steps = (
            expected_max_steps
        )

    warmup_steps = int(
        round(
            scheduler_total_steps
            * float(
                config["warmup_ratio"]
            )
        )
    )

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                warmup_steps,
            num_training_steps=
                scheduler_total_steps,
        )
    )

    global_step = 0
    cursor = 0
    best_validation_loss = math.inf

    if resume_checkpoint is not None:

        state = torch.load(
            resume_checkpoint
            / "training_state.pt",
            map_location="cpu",
        )

        optimizer.load_state_dict(
            state["optimizer"]
        )

        optimizer_to(
            optimizer,
            device,
        )

        scheduler.load_state_dict(
            state["scheduler"]
        )

        global_step = int(
            state["global_step"]
        )

        cursor = int(
            state["cursors"][rank]
        )

        best_validation_loss = float(
            state.get(
                "best_validation_loss",
                math.inf,
            )
        )

        if "rng_states" in state:
            restore_rng_state(
                state[
                    "rng_states"
                ][rank]
            )

    if rank == 0:

        processor.save_pretrained(
            run_dir / "processor"
        )

        print(
            "=" * 80,
            flush=True,
        )

        print(
            "POOLED MULTITASK LoRA",
            flush=True,
        )

        print(
            "Separate full image forward "
            "for every QA",
            flush=True,
        )

        print(
            "Visual reuse: FALSE",
            flush=True,
        )

        print(
            "Vision cache: FALSE",
            flush=True,
        )

        print(
            f"Training records: "
            f"{dataset_size:,}",
            flush=True,
        )

        print(
            f"DDP maximum steps: "
            f"{expected_max_steps:,}",
            flush=True,
        )

        print(
            f"Scheduler steps: "
            f"{scheduler_total_steps:,}",
            flush=True,
        )

        print(
            f"Warmup steps: "
            f"{warmup_steps:,}",
            flush=True,
        )

        print(
            f"Starting step: "
            f"{global_step:,}",
            flush=True,
        )

        print(
            f"Rank cursors resume: "
            f"{cursor:,}",
            flush=True,
        )

        print(
            "=" * 80,
            flush=True,
        )

    task_weights = (
        task_weights_from_counts(
            config["task_counts"]
        )
    )

    micro_batch_size = int(
        config["qa_micro_batch_size"]
    )

    max_length = int(
        config["max_sequence_length"]
    )

    training_metrics = (
        run_dir
        / "training_metrics.jsonl"
    )

    validation_metrics = (
        run_dir
        / "validation_metrics.jsonl"
    )

    skipped_log = (
        run_dir
        / f"skipped_rank{rank}.jsonl"
    )

    while True:

        if (
            configured_max_steps > 0
            and global_step
            >= configured_max_steps
        ):
            break

        has_group = (
            cursor
            < len(rank_groups)
        )

        group = (
            rank_groups[cursor]
            if has_group
            else None
        )

        if has_group:
            cursor += 1

        global_remaining = (
            reduce_sum_int(
                int(has_group),
                device,
            )
        )

        if global_remaining == 0:
            break

        prepared_chunks = []
        local_error = None

        if group is not None:

            try:

                prepared_chunks = (
                    prepare_group(
                        processor,
                        group,
                        max_length,
                        micro_batch_size,
                    )
                )

            except Exception as exc:

                local_error = repr(exc)

                append_jsonl(
                    skipped_log,
                    {
                        "group_id":
                            group.get(
                                "group_id",
                                "",
                            ),
                        "reason":
                            local_error,
                    },
                )

        valid_records = []

        for chunk_index, prepared in enumerate(
            prepared_chunks
        ):

            for qa_index, valid in enumerate(
                prepared["valid"]
            ):

                if valid:

                    valid_records.append(
                        (
                            chunk_index,
                            qa_index,
                            prepared[
                                "qa_types"
                            ][qa_index],
                        )
                    )

        local_active = int(
            len(valid_records) > 0
        )

        active_groups = (
            reduce_sum_int(
                local_active,
                device,
            )
        )

        if (
            group is not None
            and not local_active
            and local_error is None
        ):

            append_jsonl(
                skipped_log,
                {
                    "group_id":
                        group.get(
                            "group_id",
                            "",
                        ),
                    "reason":
                        "no_supervised_tokens_"
                        "after_truncation",
                },
            )

        if active_groups == 0:
            continue

        ddp_model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        local_raw_sum = 0.0
        local_raw_n = 0
        local_weighted_group_loss = 0.0

        active_scale = (
            float(world_size)
            / float(active_groups)
        )

        if local_active:

            denominator = sum(
                task_weights[qa_type]
                for _, _, qa_type
                in valid_records
            )

            active_chunk_indices = sorted(
                set(
                    chunk_index
                    for chunk_index, _, _
                    in valid_records
                )
            )

            for chunk_position, chunk_index in enumerate(
                active_chunk_indices
            ):

                prepared = (
                    prepared_chunks[
                        chunk_index
                    ]
                )

                last_sync = (
                    chunk_position
                    == len(
                        active_chunk_indices
                    ) - 1
                )

                sync_context = (
                    contextlib.nullcontext()
                    if last_sync
                    else ddp_model.no_sync()
                )

                with sync_context:

                    losses = (
                        forward_full_multimodal(
                            ddp_model,
                            prepared,
                            device,
                        )
                    )

                    component = (
                        losses.sum()
                        * 0.0
                    )

                    for qa_index, loss in enumerate(
                        losses
                    ):

                        if not prepared[
                            "valid"
                        ][qa_index]:
                            continue

                        qa_type = (
                            prepared[
                                "qa_types"
                            ][qa_index]
                        )

                        weight = (
                            task_weights[
                                qa_type
                            ]
                        )

                        coefficient = (
                            weight
                            / denominator
                        )

                        component = (
                            component
                            + (
                                loss
                                * coefficient
                                * active_scale
                            )
                        )

                        local_weighted_group_loss += (
                            float(
                                loss.detach().item()
                            )
                            * coefficient
                        )

                        local_raw_sum += float(
                            loss.detach().item()
                        )

                        local_raw_n += 1

                    component.backward()

        else:
            # Participate in DDP gradient synchronization
            # using one zero-valued real multimodal loss.

            fallback = rank_groups[0]

            dummy_chunks = prepare_group(
                processor,
                fallback,
                max_length,
                1,
            )

            dummy = None

            for candidate in dummy_chunks:
                if any(candidate["valid"]):
                    dummy = candidate
                    break

            if dummy is None:
                raise RuntimeError(
                    "Could not create DDP dummy "
                    "forward on inactive rank."
                )

            losses = forward_full_multimodal(
                ddp_model,
                dummy,
                device,
            )

            (losses.sum() * 0.0).backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                trainable_parameters,
                float(
                    config[
                        "max_grad_norm"
                    ]
                ),
            )
        )

        optimizer.step()
        scheduler.step()

        global_step += 1

        global_qa = reduce_sum_int(
            local_raw_n,
            device,
        )

        global_raw_sum = reduce_sum_float(
            local_raw_sum,
            device,
        )

        global_weighted = (
            reduce_sum_float(
                local_weighted_group_loss,
                device,
            )
        )

        grad_max = torch.tensor(
            [float(grad_norm)],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            grad_max,
            op=dist.ReduceOp.MAX,
        )

        if rank == 0:

            record = {
                "global_step":
                    global_step,

                "active_image_records":
                    active_groups,

                "qa_pairs_this_step":
                    global_qa,

                "train_weighted_loss":
                    (
                        global_weighted
                        / active_groups
                    ),

                "train_raw_mean_loss":
                    (
                        global_raw_sum
                        / global_qa
                        if global_qa > 0
                        else 0.0
                    ),

                "gradient_norm_max":
                    float(
                        grad_max.item()
                    ),

                "learning_rate":
                    float(
                        scheduler
                        .get_last_lr()[0]
                    ),

                "vision_inputs_this_step":
                    global_qa,

                "visual_reuse":
                    False,

                "cursor_rank0":
                    cursor,

                "time":
                    time.time(),
            }

            append_jsonl(
                training_metrics,
                record,
            )

            if (
                global_step
                % int(
                    config[
                        "log_every_steps"
                    ]
                )
                == 0
            ):

                print(
                    json.dumps(record),
                    flush=True,
                )

        # -----------------------------------------------
        # CHECKPOINT
        # -----------------------------------------------

        if (
            global_step
            % int(
                config[
                    "save_every_steps"
                ]
            )
            == 0
        ):

            save_checkpoint(
                ddp_model,
                optimizer,
                scheduler,
                config,
                global_step,
                cursor,
                best_validation_loss,
                device,
                rank,
                world_size,
            )

            distributed_barrier(
                device
            )

        # -----------------------------------------------
        # VALIDATION
        # -----------------------------------------------

        if (
            global_step
            % int(
                config[
                    "validation_every_steps"
                ]
            )
            == 0
        ):

            distributed_barrier(
                device
            )

            if rank == 0:

                validation = (
                    run_validation(
                        ddp_model.module,
                        processor,
                        config,
                        device,
                    )
                )

                record = {
                    "global_step":
                        global_step,
                    "validation":
                        validation,
                }

                append_jsonl(
                    validation_metrics,
                    record,
                )

                print(
                    "VALIDATION "
                    + json.dumps(record),
                    flush=True,
                )

                current = float(
                    validation[
                        "equal_type_loss"
                    ]
                )

                if (
                    current
                    < best_validation_loss
                ):

                    best_validation_loss = (
                        current
                    )

                    best_dir = (
                        adapter_run_dir
                        / "best_adapter"
                    )

                    if best_dir.exists():
                        shutil.rmtree(
                            best_dir
                        )

                    ddp_model.module.save_pretrained(
                        best_dir
                    )

                    write_json_atomic(
                        run_dir
                        / "best_validation.json",
                        {
                            "global_step":
                                global_step,
                            "equal_type_loss":
                                current,
                        },
                    )

            # Synchronize updated best value.
            best_tensor = torch.tensor(
                [best_validation_loss],
                dtype=torch.float64,
                device=device,
            )

            dist.broadcast(
                best_tensor,
                src=0,
            )

            best_validation_loss = float(
                best_tensor.item()
            )

            distributed_barrier(
                device
            )

    # ========================================================
    # FINAL CHECKPOINT + ADAPTER
    # ========================================================

    save_checkpoint(
        ddp_model,
        optimizer,
        scheduler,
        config,
        global_step,
        cursor,
        best_validation_loss,
        device,
        rank,
        world_size,
    )

    distributed_barrier(
        device
    )

    if rank == 0:

        final_adapter = (
            adapter_run_dir
            / "final_adapter"
        )

        if final_adapter.exists():
            shutil.rmtree(
                final_adapter
            )

        ddp_model.module.save_pretrained(
            final_adapter
        )

        latest_checkpoint = (
            find_latest_checkpoint(
                config
            )
        )

        if latest_checkpoint:

            shutil.copy2(
                latest_checkpoint
                / "training_state.pt",
                weight_run_dir
                / "training_state_final.pt",
            )

        write_json_atomic(
            run_dir
            / "final_summary.json",
            {
                "global_step":
                    global_step,
                "completed":
                    True,
                "best_validation_loss":
                    best_validation_loss,
                "training_strategy":
                    "pooled_repeated_image_per_qa",
                "visual_reuse":
                    False,
                "vision_cache":
                    False,
                "final_adapter":
                    str(final_adapter),
            },
        )

        print(
            "=" * 80,
            flush=True,
        )

        print(
            "TRAINING COMPLETE",
            flush=True,
        )

        print(
            f"Final step: {global_step}",
            flush=True,
        )

        print(
            f"Final adapter: "
            f"{final_adapter}",
            flush=True,
        )

        print(
            "=" * 80,
            flush=True,
        )

    distributed_barrier(
        device
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
