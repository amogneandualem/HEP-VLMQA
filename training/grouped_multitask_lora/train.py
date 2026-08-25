#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
)

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)


import peft.utils.save_and_load as peft_save_and_load


# PEFT 0.19.x invokes its tensor-parallel adapter sharding
# path whenever torch.distributed is initialized. This run
# uses ordinary DDP and has no tensor-parallel plan.
_ORIGINAL_PEFT_TP_SHARD = (
    peft_save_and_load
    ._maybe_shard_state_dict_for_tp
)


def _model_has_tensor_parallel_plan(model) -> bool:
    for module in model.modules():
        get_base_layer = getattr(
            module,
            "get_base_layer",
            None,
        )

        if get_base_layer is None:
            continue

        try:
            base_layer = get_base_layer()
        except Exception:
            continue

        tp_plan = getattr(
            base_layer,
            "_hf_tp_plan",
            None,
        )

        device_mesh = getattr(
            base_layer,
            "_hf_device_mesh",
            None,
        )

        if (
            tp_plan is not None
            and device_mesh is not None
        ):
            return True

    return False


def _safe_peft_tp_shard(
    model,
    state_dict,
    adapter_name,
):
    if not _model_has_tensor_parallel_plan(model):
        # Standard DDP: adapter tensors are already complete
        # on every rank and require no TP sharding.
        return None

    return _ORIGINAL_PEFT_TP_SHARD(
        model,
        state_dict,
        adapter_name,
    )


peft_save_and_load._maybe_shard_state_dict_for_tp = (
    _safe_peft_tp_shard
)

from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def load_helpers(project: Path):
    helper_path = (
        project
        / "scripts"
        / "10_train_grouped_encode_once.py"
    )

    if not helper_path.is_file():
        raise FileNotFoundError(helper_path)

    spec = importlib.util.spec_from_file_location(
        "grouped_single_gpu_helpers",
        helper_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import helper script: {helper_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0

    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def truncate_lines(path: Path, keep: int) -> None:
    if not path.is_file():
        return

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    path.write_text(
        "\n".join(lines[:keep])
        + ("\n" if keep > 0 else ""),
        encoding="utf-8",
    )


def find_latest_checkpoint(
    checkpoint_root: Path,
) -> Path | None:
    candidates = []

    for path in checkpoint_root.glob(
        "checkpoint-step-*"
    ):
        if not path.is_dir():
            continue

        state_path = path / "training_state.pt"

        if not state_path.is_file():
            continue

        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue

        candidates.append((step, path))

    if not candidates:
        return None

    candidates.sort()

    return candidates[-1][1]


class GroupedLossModel(nn.Module):
    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        image_features: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        core = self.peft_model.get_base_model()

        input_embeddings = (
            core.get_input_embeddings()(input_ids)
        )

        image_token_id = (
            core.config.image_token_index
        )

        image_mask = (
            input_ids == image_token_id
        )

        expected_per_sample = (
            image_mask.sum(dim=1)
        )

        actual_features = int(
            image_features.shape[0]
        )

        for expected in expected_per_sample.tolist():
            if int(expected) != actual_features:
                raise RuntimeError(
                    "Image token/feature mismatch: "
                    f"tokens={expected}, "
                    f"features={actual_features}"
                )

        repeated_features = (
            image_features
            .unsqueeze(0)
            .expand(
                input_ids.shape[0],
                -1,
                -1,
            )
            .reshape(
                -1,
                image_features.shape[-1],
            )
            .to(
                device=input_embeddings.device,
                dtype=input_embeddings.dtype,
            )
        )

        expanded_mask = (
            image_mask.unsqueeze(-1)
            .expand_as(input_embeddings)
        )

        input_embeddings = (
            input_embeddings.masked_scatter(
                expanded_mask,
                repeated_features.reshape(-1),
            )
        )

        outputs = core.model(
            input_ids=None,
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        losses = []

        for sample_index in range(
            input_ids.shape[0]
        ):
            shifted_labels = (
                labels[sample_index, 1:]
            )

            valid = shifted_labels != -100

            if not bool(valid.any()):
                raise RuntimeError(
                    "No supervised answer tokens "
                    f"in sample {sample_index}."
                )

            selected_hidden = (
                hidden_states[
                    sample_index,
                    :-1,
                    :
                ][valid]
            )

            selected_targets = (
                shifted_labels[valid]
            )

            selected_logits = (
                core.lm_head(selected_hidden)
            )

            loss = F.cross_entropy(
                selected_logits.float(),
                selected_targets,
                reduction="mean",
            )

            losses.append(loss)

        loss_vector = torch.stack(losses)

        weighted_sum = (
            loss_vector
            * sample_weights.float()
        ).sum()

        raw_sum = loss_vector.detach().sum()

        return weighted_sum, raw_sum


def prepare_next_usable(
    helper,
    dataset,
    local_order: list[int],
    cursor: int,
    processor,
    max_sequence_length: int,
    rank: int,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]] | None,
    torch.Tensor | None,
    torch.Tensor | None,
    int,
    list[dict[str, Any]],
]:
    skipped = []

    while cursor < len(local_order):
        dataset_index = local_order[cursor]
        cursor += 1

        group = dataset[dataset_index]

        (
            prepared,
            pixel_values,
            image_sizes,
        ) = helper.prepare_group(
            processor,
            group,
            max_sequence_length,
        )

        if prepared:
            return (
                group,
                prepared,
                pixel_values,
                image_sizes,
                cursor,
                skipped,
            )

        skipped.append(
            {
                "rank": rank,
                "dataset_index": dataset_index,
                "group_id": group["group_id"],
                "paper_id": group.get("paper_id"),
                "image_path": group.get("image_path"),
                "original_qa_count":
                    len(group.get("qas", [])),
                "reason":
                    "no_supervised_tokens_after_truncation",
                "max_sequence_length":
                    max_sequence_length,
            }
        )

    return (
        None,
        None,
        None,
        None,
        cursor,
        skipped,
    )


def prepare_dummy(
    helper,
    dataset,
    processor,
    max_sequence_length: int,
):
    for index in range(len(dataset)):
        group = dataset[index]

        (
            prepared,
            pixel_values,
            image_sizes,
        ) = helper.prepare_group(
            processor,
            group,
            max_sequence_length,
        )

        if prepared:
            return (
                group,
                prepared,
                pixel_values,
                image_sizes,
            )

    raise RuntimeError(
        "Could not find a usable dummy group."
    )


def save_checkpoint(
    *,
    rank: int,
    world_size: int,
    peft_model,
    processor,
    optimizer,
    scheduler,
    checkpoint_root: Path,
    metrics_path: Path,
    skipped_path: Path,
    global_step: int,
    cursor: int,
    processed_real_groups: int,
    skipped_total: int,
    best_validation_loss: float,
    dataset_size: int,
    keep_last: int,
    reason: str,
) -> Path:
    local_rank_state = {
        "rank": rank,
        "cursor": cursor,
        "skipped_total": skipped_total,
        "rng_state": capture_rng_state(),
    }

    rank_states = [None] * world_size

    dist.all_gather_object(
        rank_states,
        local_rank_state,
    )

    final_path = (
        checkpoint_root
        / f"checkpoint-step-{global_step:06d}"
    )

    if rank == 0:
        temporary_path = (
            checkpoint_root
            / f".checkpoint-step-{global_step:06d}.tmp"
        )

        if temporary_path.exists():
            shutil.rmtree(temporary_path)

        temporary_path.mkdir(
            parents=True,
            exist_ok=False,
        )

        peft_model.save_pretrained(
            temporary_path / "adapter",
            safe_serialization=True,
        )

        processor.save_pretrained(
            temporary_path / "processor"
        )

        state = {
            "global_step": global_step,
            "processed_real_groups":
                processed_real_groups,
            "best_validation_loss":
                best_validation_loss,
            "dataset_size": dataset_size,
            "world_size": world_size,
            "optimizer":
                optimizer.state_dict(),
            "scheduler":
                scheduler.state_dict(),
            "rank_states":
                rank_states,
            "metrics_lines":
                count_lines(metrics_path),
            "skipped_lines":
                count_lines(skipped_path),
            "reason":
                reason,
        }

        torch.save(
            state,
            temporary_path
            / "training_state.pt",
        )

        (
            temporary_path
            / "checkpoint_summary.json"
        ).write_text(
            json.dumps(
                {
                    "global_step":
                        global_step,
                    "processed_real_groups":
                        processed_real_groups,
                    "world_size":
                        world_size,
                    "best_validation_loss":
                        best_validation_loss,
                    "reason":
                        reason,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if final_path.exists():
            shutil.rmtree(final_path)

        temporary_path.rename(final_path)

        (
            checkpoint_root
            / "latest_checkpoint.txt"
        ).write_text(
            str(final_path) + "\n",
            encoding="utf-8",
        )

        checkpoints = []

        for path in checkpoint_root.glob(
            "checkpoint-step-*"
        ):
            try:
                step = int(
                    path.name.rsplit("-", 1)[-1]
                )
            except ValueError:
                continue

            checkpoints.append((step, path))

        checkpoints.sort()

        while len(checkpoints) > keep_last:
            _, old_path = checkpoints.pop(0)

            if old_path != final_path:
                shutil.rmtree(old_path)

        print(
            f"Checkpoint saved: {final_path}",
            flush=True,
        )

    dist.barrier()

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).resolve()

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    dist.init_process_group(
        backend="nccl"
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    torch.cuda.set_device(local_rank)

    device = torch.device(
        "cuda",
        local_rank,
    )

    expected_world_size = int(
        config["expected_world_size"]
    )

    if world_size != expected_world_size:
        raise RuntimeError(
            f"Expected {expected_world_size} ranks, "
            f"received {world_size}."
        )

    project = config_path.parent.parent
    helper = load_helpers(project)

    run_name = config["run_name"]

    run_root = (
        Path(config["output_root"])
        / run_name
    )

    checkpoint_root = (
        Path(config["checkpoint_root"])
        / run_name
    )

    metrics_path = (
        run_root
        / "training_metrics.jsonl"
    )

    validation_path = (
        run_root
        / "validation_metrics.jsonl"
    )

    skipped_path = (
        run_root
        / "skipped_groups.jsonl"
    )

    stop_file = (
        run_root
        / "STOP_REQUESTED"
    )

    manual_stop_file = (
        run_root
        / "MANUAL_STOP_REQUESTED"
    )

    if rank == 0:
        run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    dist.barrier()

    seed = int(config["seed"])
    set_seed(seed + rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(
        config["model_path"],
        local_files_only=True,
        use_fast=False,
    )

    processor.tokenizer.padding_side = "right"

    base_model = (
        LlavaOnevisionForConditionalGeneration
        .from_pretrained(
            config["model_path"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    )

    base_model.config.use_cache = False

    for parameter in base_model.parameters():
        parameter.requires_grad = False

    latest_checkpoint = None

    if config.get("resume") == "auto":
        latest_checkpoint = (
            find_latest_checkpoint(
                checkpoint_root
            )
        )

    if latest_checkpoint is None:
        target_modules = (
            helper.discover_lora_targets(
                base_model,
                config[
                    "target_module_suffixes"
                ],
            )
        )

        lora_config = LoraConfig(
            r=int(config["lora_rank"]),
            lora_alpha=int(
                config["lora_alpha"]
            ),
            lora_dropout=float(
                config["lora_dropout"]
            ),
            bias=config["lora_bias"],
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )

        peft_model = get_peft_model(
            base_model,
            lora_config,
        )

        if rank == 0:
            print(
                "Created clean DDP LoRA from base model.",
                flush=True,
            )

    else:
        peft_model = PeftModel.from_pretrained(
            base_model,
            latest_checkpoint / "adapter",
            is_trainable=True,
        )

        if rank == 0:
            print(
                f"Resuming from {latest_checkpoint}",
                flush=True,
            )

    if bool(
        config.get(
            "gradient_checkpointing",
            True,
        )
    ):
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )

        peft_model.enable_input_require_grads()

    peft_model.to(device)
    peft_model.train()

    wrapper = GroupedLossModel(
        peft_model
    ).to(device)

    ddp_model = DDP(
        wrapper,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    trainable_parameters = [
        parameter
        for parameter in ddp_model.parameters()
        if parameter.requires_grad
    ]

    if rank == 0:
        trainable_count = sum(
            parameter.numel()
            for parameter in trainable_parameters
        )

        total_count = sum(
            parameter.numel()
            for parameter in peft_model.parameters()
        )

        print(
            f"World size: {world_size}",
            flush=True,
        )

        print(
            f"Trainable parameters: "
            f"{trainable_count:,} / "
            f"{total_count:,}",
            flush=True,
        )

    train_dataset = (
        helper.JsonlOffsetDataset(
            config["train_data"]
        )
    )

    validation_dataset = (
        helper.JsonlOffsetDataset(
            config["validation_data"]
        )
    )

    dataset_size = len(train_dataset)

    global_order = list(
        range(dataset_size)
    )

    random.Random(seed).shuffle(
        global_order
    )

    local_order = global_order[
        rank::world_size
    ]

    task_counts = config["task_counts"]
    qa_types = tuple(helper.QA_TYPES)

    total_qa_count = sum(
        int(task_counts[qa_type])
        for qa_type in qa_types
    )

    task_scales = {
        qa_type:
            total_qa_count
            / (
                len(qa_types)
                * int(task_counts[qa_type])
            )
        for qa_type in qa_types
    }

    estimated_total_steps = math.ceil(
        dataset_size / world_size
    )

    configured_max_steps = int(
        config.get(
            "max_optimizer_steps",
            0,
        )
    )

    if configured_max_steps > 0:
        scheduler_total_steps = min(
            configured_max_steps,
            estimated_total_steps,
        )
    else:
        scheduler_total_steps = (
            estimated_total_steps
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(
            config["learning_rate"]
        ),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(
            config["weight_decay"]
        ),
    )

    warmup_steps = max(
        1,
        int(
            scheduler_total_steps
            * float(config["warmup_ratio"])
        ),
    )

    if config.get("scheduler") == "linear":
        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=
                    scheduler_total_steps,
            )
        )
    else:
        scheduler = (
            get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=
                    scheduler_total_steps,
            )
        )

    global_step = 0
    processed_real_groups = 0
    best_validation_loss = math.inf
    cursor = 0
    skipped_total = 0

    if latest_checkpoint is not None:
        state = torch.load(
            latest_checkpoint
            / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )

        if int(state["world_size"]) != world_size:
            raise RuntimeError(
                "Checkpoint world size differs from "
                "the current DDP world size."
            )

        if int(state["dataset_size"]) != dataset_size:
            raise RuntimeError(
                "Dataset size differs from checkpoint."
            )

        optimizer.load_state_dict(
            state["optimizer"]
        )

        scheduler.load_state_dict(
            state["scheduler"]
        )

        global_step = int(
            state["global_step"]
        )

        processed_real_groups = int(
            state["processed_real_groups"]
        )

        best_validation_loss = float(
            state["best_validation_loss"]
        )

        rank_state = (
            state["rank_states"][rank]
        )

        cursor = int(
            rank_state["cursor"]
        )

        skipped_total = int(
            rank_state["skipped_total"]
        )

        restore_rng_state(
            rank_state["rng_state"]
        )

        if rank == 0:
            truncate_lines(
                metrics_path,
                int(state["metrics_lines"]),
            )

            truncate_lines(
                skipped_path,
                int(state["skipped_lines"]),
            )

    dist.barrier()

    (
        dummy_group,
        dummy_prepared,
        dummy_pixel_values,
        dummy_image_sizes,
    ) = prepare_dummy(
        helper,
        train_dataset,
        processor,
        int(config["max_sequence_length"]),
    )

    pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    if pad_token_id is None:
        pad_token_id = (
            processor.tokenizer.eos_token_id
        )

    qa_micro_batch_size = int(
        config["qa_micro_batch_size"]
    )

    keep_last = int(
        config.get(
            "keep_last_checkpoints",
            3,
        )
    )

    save_every = int(
        config["save_every_steps"]
    )

    validation_every = int(
        config["validation_every_steps"]
    )

    log_every = int(
        config["log_every_steps"]
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    while True:
        if (
            configured_max_steps > 0
            and global_step
            >= configured_max_steps
        ):
            break

        step_start = time.time()

        (
            group,
            prepared,
            pixel_values,
            image_sizes,
            cursor,
            skip_records,
        ) = prepare_next_usable(
            helper,
            train_dataset,
            local_order,
            cursor,
            processor,
            int(config["max_sequence_length"]),
            rank,
        )

        all_skip_records = [
            None
        ] * world_size

        dist.all_gather_object(
            all_skip_records,
            skip_records,
        )

        skipped_this_step = sum(
            len(records)
            for records in all_skip_records
        )

        skipped_total += len(skip_records)

        if rank == 0 and skipped_this_step:
            with skipped_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                for rank_records in all_skip_records:
                    for record in rank_records:
                        handle.write(
                            json.dumps(record)
                            + "\n"
                        )

        has_real_group = group is not None

        local_real_qa = (
            len(prepared)
            if has_real_group
            else 0
        )

        global_real_qa_tensor = torch.tensor(
            [local_real_qa],
            device=device,
            dtype=torch.long,
        )

        dist.all_reduce(
            global_real_qa_tensor,
            op=dist.ReduceOp.SUM,
        )

        global_real_qa = int(
            global_real_qa_tensor.item()
        )

        if global_real_qa == 0:
            break

        if not has_real_group:
            group = dummy_group
            prepared = dummy_prepared
            pixel_values = dummy_pixel_values
            image_sizes = dummy_image_sizes
            active_weight = 0.0
        else:
            active_weight = 1.0

        image_features = (
            helper.encode_image_once(
                peft_model,
                pixel_values,
                image_sizes,
                device,
            )
        )

        chunks = [
            prepared[start:
                     start + qa_micro_batch_size]
            for start in range(
                0,
                len(prepared),
                qa_micro_batch_size,
            )
        ]

        # Every DDP rank must execute the same number of
        # backward calls. Images have different QA counts, so
        # synchronize the maximum number of micro-batches and
        # pad shorter ranks with zero-weight dummy batches.
        local_chunk_count_tensor = torch.tensor(
            [len(chunks)],
            device=device,
            dtype=torch.long,
        )

        dist.all_reduce(
            local_chunk_count_tensor,
            op=dist.ReduceOp.MAX,
        )

        synchronized_chunk_count = int(
            local_chunk_count_tensor.item()
        )

        if synchronized_chunk_count <= 0:
            raise RuntimeError(
                "Synchronized micro-batch count is zero."
            )

        # A padded DDP micro-batch must use tokens from the
        # same image whose features are supplied below.
        dummy_records = prepared[:1]

        local_weighted_sum = 0.0
        local_raw_sum = 0.0
        local_type_counts = Counter()

        for chunk_index in range(
            synchronized_chunk_count
        ):
            if chunk_index < len(chunks):
                records = chunks[chunk_index]
                chunk_is_real = has_real_group
            else:
                records = dummy_records
                chunk_is_real = False

            chunk_active_weight = (
                1.0 if chunk_is_real else 0.0
            )
            batch = helper.pad_microbatch(
                records,
                pad_token_id,
            )

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            sample_weights = torch.tensor(
                [
                    task_scales[
                        record["qa_type"]
                    ]
                    * chunk_active_weight
                    for record in records
                ],
                device=device,
                dtype=torch.float32,
            )

            context = (
                ddp_model.no_sync()
                if chunk_index
                < synchronized_chunk_count - 1
                else nullcontext()
            )

            with context:
                (
                    weighted_sum,
                    raw_sum,
                ) = ddp_model(
                    input_ids=input_ids,
                    attention_mask=
                        attention_mask,
                    labels=labels,
                    image_features=
                        image_features,
                    sample_weights=
                        sample_weights,
                )

                # DDP averages gradients across ranks.
                # Multiplying by world_size restores the
                # exact global QA-normalized objective.
                backward_loss = (
                    weighted_sum
                    * world_size
                    / global_real_qa
                )

                backward_loss.backward()

            if chunk_is_real:
                local_weighted_sum += float(
                    weighted_sum.detach().item()
                )

                local_raw_sum += float(
                    raw_sum.item()
                )

                for record in records:
                    local_type_counts[
                        record["qa_type"]
                    ] += 1

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=float(
                    config["max_grad_norm"]
                ),
            )
        )

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        global_step += 1

        local_real_groups = (
            1 if has_real_group else 0
        )

        real_group_tensor = torch.tensor(
            [local_real_groups],
            device=device,
            dtype=torch.long,
        )

        dist.all_reduce(
            real_group_tensor,
            op=dist.ReduceOp.SUM,
        )

        global_real_groups = int(
            real_group_tensor.item()
        )

        processed_real_groups += (
            global_real_groups
        )

        metric_tensor = torch.tensor(
            [
                local_weighted_sum,
                local_raw_sum,
                float(local_real_qa),
                float(skipped_this_step),
                float(gradient_norm),
                time.time() - step_start,
                torch.cuda.max_memory_allocated()
                / 1024**3,
            ],
            device=device,
            dtype=torch.float64,
        )

        dist.all_reduce(
            metric_tensor[0:4],
            op=dist.ReduceOp.SUM,
        )

        dist.all_reduce(
            metric_tensor[4:7],
            op=dist.ReduceOp.MAX,
        )

        type_tensor = torch.tensor(
            [
                local_type_counts[qa_type]
                for qa_type in qa_types
            ],
            device=device,
            dtype=torch.long,
        )

        dist.all_reduce(
            type_tensor,
            op=dist.ReduceOp.SUM,
        )

        if rank == 0:
            metric_record = {
                "global_step": global_step,
                "processed_real_groups":
                    processed_real_groups,
                "global_image_groups":
                    global_real_groups,
                "global_qa_pairs":
                    int(metric_tensor[2].item()),
                "train_weighted_loss":
                    float(
                        metric_tensor[0].item()
                        / max(
                            metric_tensor[2].item(),
                            1.0,
                        )
                    ),
                "train_raw_mean_loss":
                    float(
                        metric_tensor[1].item()
                        / max(
                            metric_tensor[2].item(),
                            1.0,
                        )
                    ),
                "gradient_norm_max":
                    float(metric_tensor[4].item()),
                "learning_rate":
                    float(
                        scheduler.get_last_lr()[0]
                    ),
                "step_seconds_max":
                    float(metric_tensor[5].item()),
                "gpu_memory_allocated_max_gb":
                    float(metric_tensor[6].item()),
                "skipped_groups":
                    skipped_this_step,
                "qa_type_counts": {
                    qa_type:
                        int(type_tensor[index].item())
                    for index, qa_type
                    in enumerate(qa_types)
                },
            }

            with metrics_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(metric_record)
                    + "\n"
                )

            if global_step % log_every == 0:
                print(
                    json.dumps(metric_record),
                    flush=True,
                )

        validation_ran = False

        if (
            validation_every > 0
            and global_step
            % validation_every == 0
        ):
            dist.barrier()

            if rank == 0:
                validation_result = (
                    helper.validate(
                        model=peft_model,
                        processor=processor,
                        dataset=
                            validation_dataset,
                        config=config,
                        task_scales=
                            task_scales,
                        device=device,
                    )
                )

                validation_record = {
                    "global_step": global_step,
                    "validation":
                        validation_result,
                }

                with validation_path.open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(
                        json.dumps(
                            validation_record
                        )
                        + "\n"
                    )

                current_loss = float(
                    validation_result[
                        "equal_type_loss"
                    ]
                )

                if current_loss < best_validation_loss:
                    best_validation_loss = current_loss

                    best_adapter = (
                        run_root
                        / "best_adapter"
                    )

                    if best_adapter.exists():
                        shutil.rmtree(best_adapter)

                    peft_model.save_pretrained(
                        best_adapter,
                        safe_serialization=True,
                    )

                    (
                        run_root
                        / "best_validation.json"
                    ).write_text(
                        json.dumps(
                            validation_record,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                validation_ran = True

            best_tensor = torch.tensor(
                [best_validation_loss],
                device=device,
                dtype=torch.float64,
            )

            dist.broadcast(
                best_tensor,
                src=0,
            )

            best_validation_loss = float(
                best_tensor.item()
            )

            dist.barrier()

        stop_tensor = torch.tensor(
            [
                int(
                    stop_file.exists()
                    or manual_stop_file.exists()
                )
            ],
            device=device,
            dtype=torch.long,
        )

        dist.all_reduce(
            stop_tensor,
            op=dist.ReduceOp.MAX,
        )

        stop_requested = bool(
            stop_tensor.item()
        )

        should_save = (
            global_step % save_every == 0
            or stop_requested
            or (
                configured_max_steps > 0
                and global_step
                >= configured_max_steps
            )
        )

        if should_save:
            save_checkpoint(
                rank=rank,
                world_size=world_size,
                peft_model=peft_model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_root=
                    checkpoint_root,
                metrics_path=metrics_path,
                skipped_path=skipped_path,
                global_step=global_step,
                cursor=cursor,
                processed_real_groups=
                    processed_real_groups,
                skipped_total=skipped_total,
                best_validation_loss=
                    best_validation_loss,
                dataset_size=dataset_size,
                keep_last=keep_last,
                reason=(
                    "stop_requested"
                    if stop_requested
                    else "scheduled"
                ),
            )

        if stop_requested:
            if rank == 0:
                print(
                    "STOPPED_AFTER_RESUMABLE_CHECKPOINT",
                    flush=True,
                )

            dist.barrier()
            dist.destroy_process_group()
            return

        del image_features

    save_checkpoint(
        rank=rank,
        world_size=world_size,
        peft_model=peft_model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_root=checkpoint_root,
        metrics_path=metrics_path,
        skipped_path=skipped_path,
        global_step=global_step,
        cursor=cursor,
        processed_real_groups=
            processed_real_groups,
        skipped_total=skipped_total,
        best_validation_loss=
            best_validation_loss,
        dataset_size=dataset_size,
        keep_last=keep_last,
        reason="complete",
    )

    if rank == 0:
        final_adapter = (
            run_root
            / "final_adapter"
        )

        if final_adapter.exists():
            shutil.rmtree(final_adapter)

        peft_model.save_pretrained(
            final_adapter,
            safe_serialization=True,
        )

        processor.save_pretrained(
            run_root / "processor"
        )

        summary = {
            "status": "PASS",
            "global_step": global_step,
            "processed_real_groups":
                processed_real_groups,
            "dataset_size": dataset_size,
            "world_size": world_size,
            "best_validation_loss":
                best_validation_loss,
            "final_adapter":
                str(final_adapter),
        }

        (
            run_root
            / "training_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        (
            project
            / "state"
            / f"{run_name}_COMPLETED"
        ).write_text(
            json.dumps(summary, indent=2)
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(summary, indent=2),
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
