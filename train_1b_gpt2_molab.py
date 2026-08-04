import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 1B GPT-2 Trainer — Blackwell B6000 (96GB VRAM)

        **Base**: GPT-2 architecture (MIT license, OpenAI) · **Size**: ~1.1B params
        **Framework**: Transformers Trainer · **Platform**: molab (marimo)

        Trains a 1B-parameter causal LM from scratch on CodeParrot-style code data,
        and **automatically uploads every checkpoint to your Hugging Face account**
        (time-based uploads, so nothing is lost even if the 12h session dies).
        """
    )
    return


@app.cell
def _(mo):
    import sys
    import os
    import time
    import gc
    import warnings
    import torch

    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"Python {sys.version.split()[0]} · PyTorch {torch.__version__}")

    if not torch.cuda.is_available():
        mo.stop(True, "No GPU detected. Enable the GPU in molab's notebook specs.")

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name} · VRAM: {vram:.0f} GB")
    if vram < 48:
        mo.stop(True, f"Need >=48 GB VRAM, got {vram:.0f} GB.")
    return gc, os, sys, time, torch


@app.cell
def _(sys):
    import subprocess

    def install(pkg, label=None):
        label = label or pkg
        print(f"Installing {label} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    install("transformers>=4.45.0", "transformers")
    install("accelerate>=0.34.0", "accelerate")
    install("datasets>=2.20.0", "datasets")
    install("huggingface_hub>=0.25.0", "huggingface_hub")

    print("All training dependencies installed and verified.")
    _pkgs = True
    return _pkgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 1. Credentials

        Your Hugging Face token is used to (a) log in for gated datasets and
        (b) upload checkpoints to your repo. A **write token** is required.
        You can also set the `HF_TOKEN` environment variable instead of typing it.
        """
    )
    return


@app.cell
def _(mo, os):
    import os as _os

    token = _os.environ.get("HF_TOKEN")
    if not token:
        token = mo.ui.text(
            label="Hugging Face Token (write access)",
            kind="password",
            placeholder="hf_...",
        )
        mo.output.replace(token)
        mo.stop(not token.value, "Enter your HF token to continue.")
        token = token.value

    from huggingface_hub import login, whoami

    try:
        login(token=token)
        user = whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception as exc:
        mo.stop(True, f"HF authentication failed: {exc}. Check your token.")
    return token


@app.cell
def _(_pkgs):
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        DataCollatorForLanguageModeling,
    )
    from transformers.trainer_callback import TrainerControl, TrainerState
    from datasets import load_dataset, Dataset
    from huggingface_hub import (
        create_repo,
        upload_folder,
        list_repo_tree,
        snapshot_download,
    )
    import numpy as np
    import threading

    return (
        GPT2Config,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        DataCollatorForLanguageModeling,
        load_dataset,
        Dataset,
        create_repo,
        upload_folder,
        list_repo_tree,
        snapshot_download,
        np,
        threading,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2. Configuration

        Edit the values below. The checkpoint repo is created automatically if it
        does not exist (e.g. `pinkelephantlimited/1b-gpt2`).
        """
    )
    return


@app.cell
def _(mo, os):
    repo_id = mo.ui.text(
        value=os.environ.get("CHECKPOINT_REPO", "pinkelephantlimited/1b-gpt2"),
        label="HF repo ID (checkpoints + final model)",
    )
    max_steps = mo.ui.number(start=100, stop=50000, step=100, value=2000, label="Total training steps")
    save_steps = mo.ui.number(start=50, stop=5000, step=50, value=200, label="Save checkpoint every N steps")
    upload_min = mo.ui.number(start=5, stop=180, step=5, value=40, label="Force-upload at least every N minutes")
    max_tokens = mo.ui.number(
        start=5_000_000, stop=200_000_000, step=5_000_000,
        value=50_000_000, label="Max tokens to load (RAM safety cap, 32GB)",
    )
    mo.hstack([repo_id, max_steps, save_steps, upload_min, max_tokens])
    return repo_id, max_steps, save_steps, upload_min, max_tokens


@app.cell
def _(max_steps, max_tokens, os, repo_id, save_steps, upload_min):
    REPO_ID = repo_id.value.strip()
    MAX_STEPS = int(max_steps.value)
    SAVE_STEPS = int(save_steps.value)
    UPLOAD_INTERVAL_MIN = int(upload_min.value)
    MAX_TOKENS = int(max_tokens.value)
    OUTPUT_DIR = os.path.abspath("./train_out")
    SEQ_LEN = 1024
    SEED = 42
    SAVE_TOTAL_LIMIT = 3  # keep only 3 checkpoints locally (all are on HF)
    EVAL_STEPS = 250
    RESUME_LOCAL = True  # resume from latest local checkpoint if it exists
    RESUME_FROM_HF = True  # if no local checkpoint, pull the newest from HF repo
    # Add "optimizer.pt" here to skip uploading 2GB optimizer states (faster uploads).
    IGNORE_PATTERNS = []

    MODEL_CFG = dict(
        vocab_size=50257,
        n_positions=SEQ_LEN,
        n_ctx=SEQ_LEN,
        n_embd=1600,
        n_layer=36,
        n_head=25,
        n_inner=6400,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
    )

    # Weighted mix of datasets: (dataset id, text column, weight).
    # Swap in any streaming dataset; total weight is normalized.
    DATASETS = [
        ("codeparrot/codeparrot-clean-subset", "content", 0.7),
        ("open-web-math/open-web-math", "text", 0.3),
    ]

    print(
        f"Repo: {REPO_ID} · Steps: {MAX_STEPS} · Save every {SAVE_STEPS} · "
        f"Force upload every {UPLOAD_INTERVAL_MIN} min · Tokens: {MAX_TOKENS/1e6:.0f}M"
    )
    return (
        REPO_ID,
        MAX_STEPS,
        SAVE_STEPS,
        UPLOAD_INTERVAL_MIN,
        MAX_TOKENS,
        OUTPUT_DIR,
        SEQ_LEN,
        SEED,
        SAVE_TOTAL_LIMIT,
        EVAL_STEPS,
        RESUME_LOCAL,
        RESUME_FROM_HF,
        IGNORE_PATTERNS,
        MODEL_CFG,
        DATASETS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Model

        Builds the ~1.1B-param GPT-2 (MIT) architecture from scratch.
        """
    )
    return


@app.cell
def _(GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, MODEL_CFG, torch):
    import time as _t

    _t0 = _t.time()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    cfg = GPT2Config(**MODEL_CFG)
    model = GPT2LMHeadModel(cfg)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model created: {n_params:.2f}B params ({_t.time() - _t0:.1f}s)")
    print(f"Device: {next(model.parameters()).device}")
    return cfg, model, n_params, tok


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Data

        Streams the configured datasets, tokenizes on the fly into a numpy buffer
        (RAM-safe: capped at MAX_TOKENS), and chunks into `SEQ_LEN` blocks.
        The last 500 blocks are held out for evaluation.
        """
    )
    return


@app.cell
def _(DATASETS, MAX_TOKENS, SEQ_LEN, SEED, Dataset, load_dataset, np, tok):
    import time as _t

    buf = np.zeros(MAX_TOKENS, dtype=np.int32)
    filled = 0
    targets = []
    streams = []
    for ds_id, col, w in DATASETS:
        try:
            s = load_dataset(ds_id, split="train", streaming=True)
            s = s.shuffle(seed=SEED, buffer_size=10_000)
            streams.append([iter(s), col, w])
            targets.append(int(MAX_TOKENS * w / sum(x[2] for x in DATASETS)))
            print(f"Streaming: {ds_id} (column '{col}', target {targets[-1]/1e6:.0f}M tokens)")
        except Exception as exc:
            print(f"Skip {ds_id}: {exc}")

    if not streams:
        raise RuntimeError("No dataset could be streamed. Check DATASETS / token.")

    counts = [0] * len(streams)
    done = [False] * len(streams)
    t_start = _t.time()
    while filled < MAX_TOKENS:
        advanced = False
        for i, (it, col, w) in enumerate(streams):
            if done[i]:
                continue
            if counts[i] >= targets[i] and all(
                counts[j] >= targets[j] or done[j] for j in range(len(streams))
            ):
                done[i] = True
                continue
            if counts[i] >= targets[i]:
                continue
            try:
                ex = next(it)
            except StopIteration:
                done[i] = True
                print(f"Stream {i} exhausted ({counts[i]/1e6:.0f}M tokens)")
                continue
            ids = tok(
                ex[col],
                truncation=True,
                max_length=SEQ_LEN,
            )["input_ids"]
            room = MAX_TOKENS - filled
            ids = ids[:room]
            if not ids:
                continue
            buf[filled : filled + len(ids)] = ids
            filled += len(ids)
            counts[i] += len(ids)
            advanced = True
        if not advanced or all(done):
            break

    buf = buf[:filled]
    n_blocks = filled // SEQ_LEN
    usable = n_blocks * SEQ_LEN
    blocks = buf[:usable].reshape(n_blocks, SEQ_LEN).astype(np.int64)

    n_eval = min(500, n_blocks // 10)
    train_blocks = blocks[: n_blocks - n_eval]
    eval_blocks = blocks[n_blocks - n_eval :]

    ones = lambda b: np.ones_like(b)
    train_ds = Dataset.from_dict(
        {"input_ids": train_blocks.tolist(), "attention_mask": ones(train_blocks).tolist()}
    )
    eval_ds = Dataset.from_dict(
        {"input_ids": eval_blocks.tolist(), "attention_mask": ones(eval_blocks).tolist()}
    )

    print(f"\n{filled/1e6:.1f}M tokens in {_t.time() - t_start:.0f}s")
    print(f"Blocks: {n_blocks} of {SEQ_LEN} · train {len(train_ds)} · eval {len(eval_ds)}")
    print("Per-stream tokens (M):", [round(c / 1e6, 1) for c in counts])
    return buf, counts, eval_ds, train_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Checkpoint Uploader

        A Trainer callback that:
        1. Uploads every saved checkpoint to `REPO_ID/checkpoints/checkpoint-N`
           (in a background thread, so training never stalls).
        2. Forces a save+upload if no checkpoint has been saved in
           `UPLOAD_INTERVAL_MIN` minutes — insurance against the 12h session timeout.
        """
    )
    return


@app.cell
def _(IGNORE_PATTERNS, TrainerCallback, REPO_ID, os, threading, time, upload_folder):
    class CheckpointUploader(TrainerCallback):
        def __init__(self, repo_id, token, interval_min=40, ignore=()):
            self.repo_id = repo_id
            self.token = token
            self.interval = interval_min * 60
            self.last_save = time.time()
            self.uploaded = set()

        def on_save(self, args, state, control, **kwargs):
            step = state.global_step
            src = os.path.join(args.output_dir, f"checkpoint-{step}")
            if not os.path.isdir(src) or step in self.uploaded:
                return
            self.uploaded.add(step)
            self.last_save = time.time()

            def _upload():
                try:
                    upload_folder(
                        repo_id=self.repo_id,
                        folder_path=src,
                        path_in_repo=f"checkpoints/checkpoint-{step}",
                        token=self.token,
                        commit_message=f"checkpoint-{step}",
                        ignore_patterns=list(self.ignore),
                        repo_type="model",
                    )
                    print(f"\n[uploader] checkpoint-{step} uploaded to {self.repo_id}", flush=True)
                except Exception as exc:
                    print(f"\n[uploader] upload of step {step} FAILED: {exc}", flush=True)

            threading.Thread(target=_upload, daemon=True).start()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.last_save > self.interval:
                self.last_save = time.time()
                control.should_save = True
                print(f"\n[uploader] forcing checkpoint save (interval {self.interval/60:.0f} min)", flush=True)

    print("CheckpointUploader callback defined.")
    return CheckpointUploader


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Resume

        Picks the newest checkpoint to resume from:
        1. latest local `train_out/checkpoint-*`, or
        2. newest `checkpoints/checkpoint-N` in the HF repo (downloaded locally).

        Set `RESUME_LOCAL` / `RESUME_FROM_HF` in the config cell to control this.
        """
    )
    return


@app.cell
def _(OUTPUT_DIR, REPO_ID, RESUME_FROM_HF, RESUME_LOCAL, list_repo_tree, os, snapshot_download, token):
    def find_latest_local(out_dir):
        import glob
        cps = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=os.path.getmtime)
        return cps[-1] if cps else None

    def find_latest_remote(repo_id, tok):
        try:
            tree = list_repo_tree(repo_id, path_in_repo="checkpoints", repo_type="model", token=tok)
            names = sorted({e.path.split("/")[-1] for e in tree if "checkpoint-" in e.path})
            return names[-1] if names else None
        except Exception:
            return None

    resume_path = None
    if RESUME_LOCAL:
        resume_path = find_latest_local(OUTPUT_DIR)
        if resume_path:
            print(f"Resuming from local: {os.path.basename(resume_path)}")
    if not resume_path and RESUME_FROM_HF:
        latest = find_latest_remote(REPO_ID, token)
        if latest:
            dl = snapshot_download(
                REPO_ID,
                allow_patterns=f"checkpoints/{latest}/**",
                token=token,
                repo_type="model",
            )
            resume_path = os.path.join(dl, "checkpoints", latest)
            print(f"Resuming from HF: {latest}")
    if not resume_path:
        print("No checkpoint found — starting fresh.")
    return find_latest_local, find_latest_remote, resume_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 7. Trainer

        Constructs the Trainer. Checkpoints go to `train_out/` and are mirrored
        to HF by the uploader callback. Runs with bf16 on the 96GB GPU.
        """
    )
    return


@app.cell
def _(
    CheckpointUploader,
    DataCollatorForLanguageModeling,
    EVAL_STEPS,
    IGNORE_PATTERNS,
    MAX_STEPS,
    OUTPUT_DIR,
    REPO_ID,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
    SEQ_LEN,
    Trainer,
    TrainingArguments,
    UPLOAD_INTERVAL_MIN,
    create_repo,
    eval_ds,
    model,
    tok,
    token,
    train_ds,
):
    create_repo(REPO_ID, repo_type="model", token=token, exist_ok=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.01,
        max_steps=MAX_STEPS,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=[],
        seed=SEED,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=None,
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[
            CheckpointUploader(
                repo_id=REPO_ID,
                token=token,
                interval_min=UPLOAD_INTERVAL_MIN,
                ignore=IGNORE_PATTERNS,
            )
        ],
    )
    print(
        f"Trainer ready · steps {MAX_STEPS} · batch 16 x2 accum = 32K tok/step · "
        f"~{(32 * SEQ_LEN):,} tokens/step"
    )
    return collator, trainer, trainer_args


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 8. Train

        Run the cell below and leave it alone — checkpoints auto-upload.
        If the 12h session dies, restart this notebook from GitHub and the
        **Resume** cell will pick up the newest checkpoint from HF.
        """
    )
    return


@app.cell
def _(gc, os, resume_path, time, torch, trainer):
    gc.collect()
    torch.cuda.empty_cache()
    _t0 = time.time()

    result = trainer.train(resume_from_checkpoint=resume_path)

    _elapsed = time.time() - _t0
    print(f"\nTraining finished in {_elapsed/60:.1f} min · final loss {result.training_loss:.3f}")
    print(f"Best step: {result.global_step}")

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak VRAM used: {peak:.1f} GB")
    return result


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 9. Test Generation

        Quick sanity check after training.
        """
    )
    return


@app.cell
def _(model, tok, torch):
    import time as _t

    prompt = "def is_prime(n):\n    "
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    _t0 = _t.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    _elapsed = _t.time() - _t0
    text = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    n_new = out.shape[1] - inputs.input_ids.shape[1]
    print(f"[{n_new} tok in {_elapsed:.1f}s · {n_new/max(_elapsed,1e-6):.1f} tok/s]\n")
    print(prompt + text)
    return text


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 10. Push Final Model

        Saves the final model to `REPO_ID/final/` (weights + tokenizer + config),
        and writes a short README.
        """
    )
    return


@app.cell
def _(OUTPUT_DIR, REPO_ID, os, token, trainer, upload_folder):
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)

    trainer.save_model(final_dir)
    with open(os.path.join(final_dir, "README.md"), "w") as f:
        f.write(
            "# 1B GPT-2 (MIT)\n\n"
            "Trained on molab (Blackwell B6000, 96GB). Checkpoints in `/checkpoints/`.\n"
        )

    try:
        upload_folder(
            repo_id=REPO_ID,
            folder_path=final_dir,
            path_in_repo="final",
            token=token,
            commit_message="final model",
            repo_type="model",
        )
        print(f"Final model pushed to https://huggingface.co/{REPO_ID}/tree/main/final")
    except Exception as exc:
        print(f"Final push FAILED: {exc}")

    print(f"\nDone. Model: {REPO_ID}")
    return final_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---
        **Base**: GPT-2 (MIT) · **Size**: ~1.1B params · **License**: MIT
        **Checkpoints**: auto-uploaded to HF every save + every N minutes
        **Platform**: molab (marimo) · **RAM safety**: token cap for 32GB
        """
    )
    return


if __name__ == "__main__":
    app.run()
