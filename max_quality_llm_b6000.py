import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Maximum Quality LLM — Blackwell B6000 (96GB VRAM)

        **Model**: Meta-Llama-3.1-70B-Instruct · **Precision**: 8-bit (near-lossless)
        **Framework**: Transformers + bitsandbytes + Flash Attention 2

        Runs the best open-source LLM at maximum quality on a 96GB VRAM GPU.
        """
    )
    return


@app.cell
def _(mo):
    import torch
    import sys
    import os
    import time
    import gc
    import warnings

    warnings.filterwarnings("once", category=UserWarning)
    print(f"Python {sys.version.split()[0]} · PyTorch {torch.__version__}")

    if not torch.cuda.is_available():
        mo.stop(True, "No GPU detected. Enable the GPU in molab's notebook specs.")

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_capability()
    print(f"GPU: {name} · VRAM: {vram:.0f} GB · CUDA {cap[0]}.{cap[1]}")
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
    install("bitsandbytes>=0.44.0", "bitsandbytes")
    install("huggingface_hub>=0.25.0 gradio>=5.0", "huggingface_hub + gradio")

    flash_ok = False
    try:
        import flash_attn
        flash_ok = True
    except ImportError:
        try:
            install("flash-attn --no-build-isolation", "flash-attn")
            import flash_attn
            flash_ok = True
        except Exception:
            print(
                "Warning: flash-attn installation failed. "
                "Falling back to SDPA attention (slower, higher VRAM)."
            )

    print("All dependencies installed and verified.")
    _pkgs = True
    return _pkgs, flash_ok


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Model Setup
        Loads Llama 3.1 70B with 8-bit quantization.

        **VRAM budget**: ~70 GB weights + ~15 GB KV cache = fits in 96 GB.

        You need a Hugging Face token and must accept the license at
        [huggingface.co/meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct).
        """
    )
    return


@app.cell
def _(mo):
    import os

    token = os.environ.get("HF_TOKEN")
    if not token:
        token = mo.ui.text(
            label="Hugging Face Token",
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
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        GenerationConfig,
        TextIteratorStreamer,
    )
    from huggingface_hub import login, whoami
    import threading

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
        TextIteratorStreamer,
        login,
        threading,
        whoami,
    )


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    flash_ok,
    time,
    token,
    torch,
):
    MODEL = "meta-llama/Llama-3.1-70B-Instruct"
    print(f"Loading: {MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    tok.pad_token = tok.eos_token

    attn_impl = "flash_attention_2" if flash_ok else "sdpa"
    if not flash_ok:
        print("Using SDPA attention (flash-attn unavailable)")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        token=token,
        low_cpu_mem_usage=True,
        use_cache=True,
    )
    elapsed = (time.time() - t0) / 60

    torch.cuda.reset_peak_memory_stats()
    peak = torch.cuda.max_memory_allocated() / 1e9
    current = torch.cuda.memory_allocated() / 1e9

    print(f"Loaded in {elapsed:.1f} min")
    print(f"VRAM: {current:.1f} GB current, {peak:.1f} GB peak")
    free = 96 - current
    if free < 10:
        print(f"Warning: only {free:.1f} GB VRAM free — reduce context length.")
    return model, tok


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Chat Interface
        Multi-turn conversation with streaming output.
        """
    )
    return


@app.cell
def _(GenerationConfig, TextIteratorStreamer, model, threading, tok):
    def build_prompt(conversation, system=None):
        if system:
            conversation = [{"role": "system", "content": system}] + conversation
        return tok.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

    def chat(conversation, system=None, max_tokens=2048, temp=0.7, top_p=0.9):
        prompt = build_prompt(conversation, system)
        inputs = tok(prompt, return_tensors="pt", truncation=True).to(model.device)

        gen = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            do_sample=temp > 0,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=gen,
            streamer=streamer,
            use_cache=True,
        )
        t = threading.Thread(target=model.generate, kwargs=kwargs)
        t.start()

        out = []
        for chunk in streamer:
            out.append(chunk)
            print(chunk, end="", flush=True)
        t.join()
        return "".join(out)

    print("Chat function ready.")
    return build_prompt, chat


@app.cell
def _(mo):
    mo.md(
        """
        ## Quick Test
        Run the next cell to verify inference works before using the Gradio UI.
        """
    )
    return


@app.cell
def _(chat, time, tok):
    import torch

    test = "Write a Python function to check if a string is a palindrome."
    print(f"Prompt: {test}")
    t0 = time.time()
    with torch.inference_mode():
        reply = chat(
            [{"role": "user", "content": test}], max_tokens=256, temp=0.5
        )
    elapsed = time.time() - t0
    nt = len(tok.encode(reply))
    print(f"\n[{nt} tok · {elapsed:.1f}s · {nt/elapsed:.1f} tok/s]")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Benchmark
        Measures throughput across 3 prompts (512 tokens each, greedy decoding).
        """
    )
    return


@app.cell
def _(chat, gc, time, tok, torch):
    import random

    torch.manual_seed(42)
    random.seed(42)

    prompts = [
        "Explain how a transformer attention mechanism works.",
        "Write a Python async web scraper with error handling.",
        "Compare arrays vs linked lists with Big-O analysis.",
    ]
    results = []
    for i, p in enumerate(prompts):
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        with torch.inference_mode():
            reply = chat(
                [{"role": "user", "content": p}],
                max_tokens=512,
                temp=0.0,
            )
        elapsed = max(time.time() - t0, 1e-6)
        nt = len(tok.encode(reply))
        tok_s = nt / elapsed
        results.append({"prompt": p[:50], "tokens": nt, "seconds": elapsed, "tok_s": tok_s})
        print(f"[{i+1}/3] {nt} tok · {elapsed:.1f}s · {tok_s:.1f} tok/s")

    avg = sum(r["tok_s"] for r in results) / len(results)
    print(f"\nAverage: {avg:.1f} tokens/second")
    results.append({"prompt": "AVERAGE", "tokens": "", "seconds": "", "tok_s": avg})
    return results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Gradio Web UI (Optional)
        Launch a shareable chat interface. The UI runs in a separate thread.
        """
    )
    return


@app.cell
def _(mo, model, tok, TextIteratorStreamer, build_prompt):
    import gradio as gr
    import threading as _t
    import torch as _torch

    def respond(message, history_gradio):
        hist = []
        for u, a in history_gradio:
            hist.append({"role": "user", "content": u})
            hist.append({"role": "assistant", "content": a})
        hist.append({"role": "user", "content": message})

        prompt = build_prompt(hist)
        inputs = tok(prompt, return_tensors="pt", truncation=True).to(model.device)

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            streamer=streamer,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        t = _t.Thread(target=model.generate, kwargs=kwargs)
        t.start()

        out = []
        for chunk in streamer:
            out.append(chunk)
            yield "".join(out)

    demo = gr.ChatInterface(
        respond,
        title="Llama 3.1 70B (8-bit) — Max Quality",
        description="Blackwell B6000 · 96GB VRAM",
    )

    try:
        demo.launch(share=True, debug=False, prevent_thread_lock=True)
    except Exception as exc:
        print(f"Gradio launch failed: {exc}")
        print("The Gradio UI cell can be safely skipped; use the chat() function above.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---
        **Model**: Meta-Llama-3.1-70B-Instruct · **Quantization**: 8-bit Int8
        **VRAM**: ~70 GB w + ~15 GB cache · **Attention**: Flash Attn 2 (fallback: SDPA)
        **Platform**: molab (marimo) · **License**: MIT (notebook)
        """
    )
    return


if __name__ == "__main__":
    app.run()
