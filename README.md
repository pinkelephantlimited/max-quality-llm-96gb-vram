# Max Quality LLM — Blackwell B6000 (96GB)

Run maximum-quality open-source LLMs on **molab** (marimo) with an RTX PRO 6000 Blackwell GPU (96GB GDDR7).

[![Open on molab](https://img.shields.io/badge/molab-open-blue)](https://molab.marimo.io/github/pinkelephantlimited/max-quality-llm-96gb-vram/blob/main/train_1b_gpt2_molab.py)

## Notebooks

| Notebook | Purpose |
|---|---|
| `train_1b_gpt2_molab.py` | Train a 1B-param GPT-2 (MIT) from scratch, **auto-uploading every checkpoint to Hugging Face** (time-based force-saves survive the 12h session limit). Resume from latest HF checkpoint after session restarts. |
| `max_quality_llm_b6000.py` | Chat with Llama 3.1 70B (8-bit) at max quality — flash-attn with SDPA fallback, streaming chat, benchmark, Gradio UI. |
| `Max_Quality_LLM_96GB_VRAM.py` | Older marimo conversion of the original Colab notebook. |
| `Max_Quality_LLM_96GB_VRAM.ipynb` | Original Jupyter notebook (Colab). |

## Using on molab

1. Open a notebook via the badge above (or the Files tab on [molab](https://molab.marimo.io)).
2. Toggle **GPU on** in the notebook header (Enable: `RTX Pro 6000 Blackwell 96GB`).
3. Enter your **Hugging Face write token** when prompted.
4. Run cells in order.

## 1B GPT-2 Trainer

- Architecture: GPT-2 (MIT, OpenAI), ~1.1B params (36 layers, 1600 hidden, 1024 ctx).
- Data: weighted stream of `codeparrot/codeparrot-clean-subset` (70%) + `open-web-math/open-web-math` (30%) — editable in the config cell; RAM-safe token cap for 32GB.
- Checkpoints: saved to `train_out/`, uploaded to `checkpoints/checkpoint-N` in your HF repo after every save **and** force-saved+uploaded every N minutes (default 40).
- Resume: picks the newest local checkpoint, else downloads the newest from HF automatically.
- Final model pushed to `<repo>/final`.

## License

MIT (notebooks). Model weights: GPT-2 is MIT.
