import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoModelForCausalLM, AutoTokenizer
import modal
from tqdm import tqdm
import random
import os
import subprocess
from contextlib import nullcontext
from datetime import timedelta

from consts import (
    BATCH_SIZE,
    CHECKPOINT,
    GRAD_ACCUM_STEPS,
    HF_REPO_SAVE_PATH,
    LEARNING_RATE,
    LETTERS,
    LOG_EVERY_TRAIN,
    LOG_EVERY_VALID,
    LORA_ALPHA,
    LORA_RANK,
    LR_PERCENTAGE_START,
    N_EPOCHS,
    SEED,
    MODAL_APP_NAME,
    CHECKPOINT_DIR,
)
from data import *
from load_data import load_train, load_valid
from lora import LoRALinear, apply_lora, merge_lora

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name("decision-checkpoints", create_if_missing=True)
CKPT_DIR = CHECKPOINT_DIR


def get_device():
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def save_checkpoint(model, optimizer, scheduler, step):
    state = {
        "model": {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }

    torch.save(state, os.path.join(CKPT_DIR, "checkpoint.pth"))
    volume.commit()


def init_instances():
    torch.manual_seed(SEED)
    random.seed(SEED)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=get_device(),
        timeout=timedelta(hours=2),
    )

    is_main = rank == 0
    per_device_batch_size = BATCH_SIZE // world_size

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype="bfloat16", device_map=get_device()
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    LETTER_IDS = torch.cat(
        [tokenizer.encode(letter, return_tensors="pt").squeeze(0) for letter in LETTERS]
    ).to(get_device())
    LETTERS_RANGE = torch.arange(len(LETTER_IDS), device=get_device()).unsqueeze(0)

    (
        data_anli_train,
        data_mmlu_train,
        data_sciq_train,
        data_hellaswag_train,
        data_tasksource_train,
        data_openjev_train,
        data_ledgar_train,
        data_casehold_train,
        data_nli_train,
        data_vitamin_train,
        data_onejev_train,
    ) = load_train()
    (
        data_tasksource_valid,
        data_openjev_valid,
        data_ledgar_valid,
        data_casehold_valid,
        data_nli_valid,
        data_hellaswag_valid,
    ) = load_valid()

    train_dataset = ConcatDataset(
        [
            ANLIDataWrapper(data_anli_train, tokenizer),
            MMLUDataWrapper(data_mmlu_train, tokenizer),
            SCIQDataWrapper(data_sciq_train, tokenizer),
            JEVDataWrapper(data_tasksource_train, tokenizer),
            LEDGARDataWrapper(data_ledgar_train, tokenizer),
            CASEHOLDDataWrapper(data_casehold_train, tokenizer),
            JEVDataWrapper(data_openjev_train, tokenizer, is_context_json=True),
            NLIDataWrapper(data_nli_train, tokenizer),
            HELLASWAGDataWrapper(data_hellaswag_train, tokenizer),
            VITAMINDataWrapper(data_vitamin_train, tokenizer),
            # Triple duplicate to enhance impact on train.
            ONEJEVDataWrapper(data_onejev_train, tokenizer),
            ONEJEVDataWrapper(data_onejev_train, tokenizer),
            ONEJEVDataWrapper(data_onejev_train, tokenizer),
        ]
    )

    valid_dataset = ConcatDataset(
        [
            JEVDataWrapper(data_tasksource_valid, tokenizer),
            LEDGARDataWrapper(data_ledgar_valid, tokenizer),
            CASEHOLDDataWrapper(data_casehold_valid, tokenizer),
            JEVDataWrapper(data_openjev_valid, tokenizer, is_context_json=True),
            NLIDataWrapper(data_nli_valid, tokenizer),
            HELLASWAGDataWrapper(data_hellaswag_valid, tokenizer),
        ]
    )

    sampler_train = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
        sampler=sampler_train,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    apply_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    print(f"Total arams: {total_params}")
    print(f"Trainable params: {trainable_params}")
    print(f"Trainable percentage: {round((trainable_params / total_params), 3)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, fused=True
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=int(len(train_loader) // GRAD_ACCUM_STEPS),
        epochs=N_EPOCHS,
        pct_start=LR_PERCENTAGE_START,
        cycle_momentum=False,
    )

    model = DDP(model, device_ids=[local_rank])

    return (
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        LETTERS_RANGE,
        LETTER_IDS,
        tokenizer,
    )


def train(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    LETTERS_RANGE,
    LETTER_IDS,
):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    n_ok = 0
    is_main = dist.get_rank() == 0

    iterator = tqdm(train_loader) if is_main else train_loader
    for step, batch in enumerate(iterator):
        sync_now = (step + 1) % GRAD_ACCUM_STEPS == 0

        with nullcontext() if sync_now else model.no_sync():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens_padded, attention_mask, labels, K = (
                    b.to(get_device()) for b in batch
                )
                output = model(
                    tokens_padded, attention_mask=attention_mask, logits_to_keep=1
                )
                pred_token_logits = output.logits[:, -1]
                only_selected_ids = pred_token_logits[:, LETTER_IDS]
                only_selected_ids.masked_fill_(
                    LETTERS_RANGE >= K.unsqueeze(1), float("-inf")
                )
                loss_raw = criterion(only_selected_ids, labels)

            loss = loss_raw / GRAD_ACCUM_STEPS
            loss.backward()

        if is_main:
            if torch.isfinite(loss_raw):
                total_loss += loss_raw.item()
                n_ok += 1
            else:
                print(
                    f"Step {step + 1}: loss={loss_raw.item()} | K={K.tolist()} "
                    f"| labels={labels.tolist()} | seq_len={tokens_padded.shape[1]}"
                )

        if sync_now:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            elif is_main:
                print(f"Step {step + 1}: non-finite grad_norm, optimizer step skipped.")
            scheduler.step()
            optimizer.zero_grad()


        if is_main and (step + 1) % LOG_EVERY_TRAIN == 0:
            print(f"Step: {step + 1} | train_loss: {total_loss / max(n_ok, 1)}")

        if (step + 1) % LOG_EVERY_VALID == 0:
            if is_main:
                val_loss, val_acc = evaluate(
                    model.module, valid_loader, LETTERS_RANGE, LETTER_IDS, criterion
                )
                print(
                    f"Step: {step + 1} | valid_loss: {val_loss:.3f} | valid_acc: {val_acc:.3f}"
                )
                save_checkpoint(model.module, optimizer, scheduler, step + 1)
            dist.barrier()
            model.train()


def evaluate(model, valid_loader, LETTERS_RANGE, LETTER_IDS, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0

    for step, batch in enumerate(valid_loader):
        with torch.no_grad():
            tokens_padded, attention_mask, labels, K = (
                b.to(get_device()) for b in batch
            )
            output = model(
                tokens_padded, attention_mask=attention_mask, logits_to_keep=1
            )
            pred_token_logits = output.logits[:, -1]
            only_selected_ids = pred_token_logits[:, LETTER_IDS]
            only_selected_ids.masked_fill_(
                LETTERS_RANGE >= K.unsqueeze(1), float("-inf")
            )

        loss = criterion(only_selected_ids, labels)
        total_loss += loss.item()

        acc = torch.sum(torch.argmax(only_selected_ids, dim=1) == labels) / len(labels)
        total_acc += acc.item()

    return total_loss / len(valid_loader), total_acc / len(valid_loader)


def launch_training():
    (
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        LETTERS_RANGE,
        LETTER_IDS,
        tokenizer,
    ) = init_instances()

    train(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        LETTERS_RANGE,
        LETTER_IDS,
    )

    if dist.get_rank() == 0:
        base = model.module
        save_checkpoint(base, optimizer, scheduler, step=-1)
        base.eval()
        merge_lora(base)
        print(sum(isinstance(m, LoRALinear) for m in base.modules()))

        base.gradient_checkpointing_disable()
        base.config.use_cache = True

        base.push_to_hub(HF_REPO_SAVE_PATH, private=True)
        tokenizer.push_to_hub(HF_REPO_SAVE_PATH, private=True)
        print("Model saved to HF.")

        _, val_acc = evaluate(base, valid_loader, LETTERS_RANGE, LETTER_IDS, criterion)
        print(val_acc)

    dist.barrier()
    destroy_process_group()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.20.2",
        "datasets==4.8.5",
        "accelerate",
        "tqdm",
        "huggingface_hub",
        "flash-linear-attention==0.6.0",
    )
    .env({"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("consts", "data", "load_data", "lora", "prompt", "train")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=["B200:4"],
    cpu=32,
    memory=131072,
    volumes={CKPT_DIR: volume, "/root/.cache/huggingface": hf_cache},
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run():
    subprocess.run(["torchrun", "--nproc_per_node=4", "-m", "train"], check=True)


if __name__ == "__main__":
    launch_training()
