import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Maximum Quality Open-Source LLM on Google Colab
    ---
    **Hardware**: Blackwell B6000 (96GB VRAM) | **System RAM**: 32GB
    **Model**: Meta-Llama-3.1-70B-Instruct (8-bit) | **Framework**: Transformers + bitsandbytes + Flash Attention 2
    ---
    This notebook runs the **largest possible open-source LLM at maximum quality** on a 96GB VRAM GPU.
    Using 8-bit quantization (near-lossless), we load **Llama 3.1 70B Instruct** with high throughput.
    """)
    return


@app.cell
def _():
    # Verify GPU
    import torch, sys, subprocess, os, time, gc, json

    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    else:
        print("WARNING: No GPU detected! This notebook needs a GPU with >=48GB VRAM.")
        print('Go to Runtime > Change runtime type > A100 / B6000 GPU')
        raise SystemExit(0)
    return gc, os, subprocess, sys, time, torch


@app.cell
def _(subprocess, sys):
    # Install dependencies
    import importlib, warnings
    warnings.filterwarnings('ignore')

    def install_if_missing(pkg, pip_name=None):
        pip_name = pip_name or pkg
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f'Installing {pip_name}...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_name])
    install_if_missing('transformers', 'transformers>=4.45.0')
    # Core ML
    install_if_missing('accelerate', 'accelerate>=0.34.0')
    install_if_missing('bitsandbytes', 'bitsandbytes>=0.44.0')
    try:
        import flash_attn
    # Flash Attention
        print(f'Flash Attention: {flash_attn.__version__}')
    except ImportError:
        print('Installing Flash Attention...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'flash-attn', '--no-build-isolation'])
    install_if_missing('gradio', 'gradio>=5.0')
    install_if_missing('huggingface_hub', 'huggingface_hub>=0.25.0')
    print('All dependencies installed successfully!')
    import bitsandbytes as bnb
    print(f'bitsandbytes: {bnb.__version__}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the Model (8-bit, Max Quality)
    We load **Meta-Llama-3.1-70B-Instruct** with 8-bit quantization.
    - 8-bit is **near-lossless** (<1% quality degradation vs FP16)
    - Fits in 96GB VRAM: ~70GB weights + ~15GB KV cache
    - Flash Attention 2 for memory-efficient attention
    - We use **bfloat16** compute dtype for best numerical stability
    """)
    return


@app.cell
def _():
    # Import libraries
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
    from accelerate import dispatch_model, infer_auto_device_map
    MODEL_ID = 'meta-llama/Llama-3.1-70B-Instruct'
    # Alternative models (uncomment to use):
    # MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
    # MODEL_ID = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    print(f'Target model: {MODEL_ID}')
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
        MODEL_ID,
    )


@app.cell
def _(os):
    # Hugging Face login (required for Llama models)
    from huggingface_hub import login, whoami
    import getpass
    HF_TOKEN = os.environ.get('HF_TOKEN', None)
    if not HF_TOKEN:
        print('Enter your Hugging Face token (get one at huggingface.co/settings/tokens)')
        print('You need to accept the license at: huggingface.co/meta-llama/Llama-3.1-70B-Instruct')
        HF_TOKEN = getpass.getpass('HF Token: ')
    if HF_TOKEN:
        login(token=HF_TOKEN)
        user = whoami()
        print(f"Logged in as: {user['name']}")
    else:
        print('No HF token provided. Loading gated models will fail.')
        print('See: huggingface.co/meta-llama/Llama-3.1-70B-Instruct')
    return (HF_TOKEN,)


@app.cell
def _(BitsAndBytesConfig, torch):
    # Configure 8-bit quantization for maximum quality
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,                # 8-bit = near-lossless
        llm_int8_threshold=6.0,           # Threshold for outlier detection
        llm_int8_has_fp16_weight=False,   # Use int8 weights (not mixed)
        llm_int8_enable_fp32_cpu_offload=False,  # Keep all on GPU
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Quantization config:")
    print(f"  load_in_8bit: {bnb_config.load_in_8bit}")
    print(f"  compute_dtype: bfloat16")
    return (bnb_config,)


@app.cell
def _(AutoTokenizer, HF_TOKEN, MODEL_ID):
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    print(f"Max length: {tokenizer.model_max_length}")
    return (tokenizer,)


@app.cell
def _(AutoModelForCausalLM, HF_TOKEN, MODEL_ID, bnb_config, time, torch):
    _t0 = time.time()
    print('Downloading and loading model (this will take a while)...')
    print('Estimated VRAM after load: ~75-80 GB')
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', token=HF_TOKEN, low_cpu_mem_usage=True, use_cache=True)
    _t1 = time.time()
    print(f'\nModel loaded in {(_t1 - _t0) / 60:.1f} minutes')
    print(f'Model parameters: {model.num_parameters() / 1000000000.0:.1f}B')
    vram_used = torch.cuda.memory_allocated() / 1000000000.0
    vram_reserved = torch.cuda.memory_reserved() / 1000000000.0
    print(f'VRAM used: {vram_used:.1f} GB')
    print(f'VRAM reserved: {vram_reserved:.1f} GB')
    print(f'VRAM free: {96 - vram_reserved:.1f} GB')
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference Pipeline
    Optimized chat function with:
    - Streaming output (token-by-token)
    - KV cache reuse for multi-turn conversations
    - Dynamic context length management
    - Temperature, top-p, and other sampling controls
    """)
    return


@app.cell
def _(GenerationConfig, model, tokenizer, torch):
    # Define optimized chat function with streaming
    from transformers import TextIteratorStreamer
    from threading import Thread

    def create_chat_prompt(messages, system_prompt=None):
        if system_prompt:
            messages = [{'role': 'system', 'content': system_prompt}] + messages
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_response(messages, system_prompt='You are a helpful, knowledgeable AI assistant.', max_new_tokens=2048, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05, do_stream=True):
        prompt = create_chat_prompt(messages, system_prompt)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        print(f'Prompt tokens: {input_ids.shape[1]}')
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, do_sample=temperature > 0, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        if do_stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config, streamer=streamer, use_cache=True)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            full_response = ''
            for new_token in streamer:
                full_response += new_token
                print(new_token, end='', flush=True)
            print()
            thread.join()
        else:
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config, use_cache=True)
            full_response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        _new_tokens = len(tokenizer.encode(full_response)) if full_response else 0
        return (full_response, _new_tokens)
    print('Chat function defined. Ready for inference!')
    return TextIteratorStreamer, Thread, create_chat_prompt, generate_response


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Chat Demo
    Run the cell below to start a conversation with Llama 3.1 70B at maximum quality.
    """)
    return


@app.cell
def _(generate_response, time):
    # Simple interactive chat
    messages_history = []
    system = 'You are a helpful, knowledgeable AI assistant with deep expertise in science, technology, and the arts. Be concise but thorough.'
    print('=' * 60)
    print('Llama 3.1 70B (8-bit) - Interactive Chat')
    print('=' * 60)
    print("Type 'quit' to exit, 'clear' to reset conversation")
    while True:
        user_input = input('\n>>> ')
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if user_input.lower() == 'clear':
            messages_history = []
            print('Conversation cleared.')
            continue
        if not user_input.strip():
            continue
        messages_history.append({'role': 'user', 'content': user_input})
        _t0 = time.time()
        _response, _new_tokens = generate_response(messages_history, system_prompt=system, max_new_tokens=2048, temperature=0.7, do_stream=True)
        _t1 = time.time()
        messages_history.append({'role': 'assistant', 'content': _response})
        _elapsed = _t1 - _t0
        tokens_per_sec = _new_tokens / _elapsed if _elapsed > 0 else 0
        print(f'\n[Generated {_new_tokens} tokens in {_elapsed:.1f}s ({tokens_per_sec:.1f} tok/s)]')
    print('Session ended.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark: Inference Speed & Quality
    Measure tokens per second across different generation settings.
    """)
    return


@app.cell
def _(gc, generate_response, time, torch):
    # Benchmark different configurations
    test_prompts = ['Explain quantum entanglement in simple terms.', 'Write a Python function to merge k sorted linked lists.', 'Summarize the key differences between transformer and CNN architectures.']
    results = []
    for i, prompt in enumerate(test_prompts):
        print(f'\n--- Benchmark {i + 1}: {prompt[:50]}... ---')
        msgs = [{'role': 'user', 'content': prompt}]
        gc.collect()
        torch.cuda.empty_cache()
        _t0 = time.time()
        _response, _new_tokens = generate_response(msgs, max_new_tokens=512, temperature=0.7, do_stream=True)
        _t1 = time.time()
        _elapsed = _t1 - _t0
        tok_s = _new_tokens / _elapsed if _elapsed > 0 else 0
        results.append({'prompt': prompt[:60], 'tokens': _new_tokens, 'time': _elapsed, 'tok_s': tok_s})
        print(f'Result: {_new_tokens} tokens in {_elapsed:.1f}s = {tok_s:.1f} tok/s')
    print('\n' + '=' * 60)
    print('BENCHMARK SUMMARY')
    print('=' * 60)
    for r in results:
        print(f"  {r['prompt']}: {r['tok_s']:.1f} tok/s ({r['tokens']} tokens in {r['time']:.1f}s)")
    avg_tok_s = sum((r['tok_s'] for r in results)) / len(results)
    print(f'\nAverage throughput: {avg_tok_s:.1f} tokens/second')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradio Web Interface (Optional)
    Launch a chat UI in your browser. Share the public link to let others interact with the model.
    """)
    return


@app.cell
def _(TextIteratorStreamer, Thread, create_chat_prompt, model, tokenizer):
    # Launch Gradio chat interface
    import gradio as gr

    def chat_fn(message, history):
        history_with_role = []
        for user, assistant in history:
            history_with_role.append({"role": "user", "content": user})
            history_with_role.append({"role": "assistant", "content": assistant})
        history_with_role.append({"role": "user", "content": message})

        prompt = create_chat_prompt(history_with_role, "You are a helpful AI assistant.")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            streamer=streamer,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            yield partial_message

    # Create and launch interface
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Llama 3.1 70B (8-bit) - Max Quality Open-Source LLM",
        description="Running on Blackwell B6000 with 96GB VRAM. 8-bit quantization for near-lossless quality.",
        theme="soft",
        examples=[
            "Explain the transformer architecture in detail.",
            "Write a Python implementation of a vector database.",
            "Compare and contrast LoRA vs QLoRA fine-tuning.",
        ],
    )

    print("Launching Gradio interface...")
    print("URL will appear below (both local and public share link)")
    demo.launch(share=True, debug=False, server_port=7860)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performance Summary
    | Setting | Value |
    |---------|-------|
    | Model | Meta-Llama-3.1-70B-Instruct |
    | Parameters | 70.6B |
    | Quantization | 8-bit (Int8) - near-lossless |
    | Compute dtype | bfloat16 |
    | Attention | Flash Attention 2 |
    | VRAM usage | ~70 GB (weights) + ~15 GB (cache) |
    | Context length | Up to 128K tokens |
    | Framework | Transformers + bitsandbytes + accelerate |

    ### Next Steps
    - **Larger models**: For 96GB VRAM, try Mixtral-8x22B at 8-bit (~50GB) or DeepSeek-V2 at 4-bit
    - **Fine-tuning**: Use LoRA/QLoRA for task-specific adaptation
    - **Serving**: Deploy with vLLM for production-grade inference
    - **vLLM backend**: Use `vllm serve meta-llama/Llama-3.1-70B-Instruct --quantization bitsandbytes --dtype bfloat16`
    """)
    return


if __name__ == "__main__":
    app.run()
