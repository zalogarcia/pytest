#!/usr/bin/env python3
"""
Expose fine-tuned Qwen3.5-122B-A10B as an OpenAI-compatible API.

Runs on port 8080. Access via RunPod proxy:
  https://<pod-id>-8080.proxy.runpod.net/v1/chat/completions

Usage:
  python3 serve_model.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
from flask import Flask, request, jsonify

app = Flask(__name__)
model = None
tokenizer = None
_enc = None


def load_model():
    global model, tokenizer, _enc

    # Import unsloth FIRST -- it patches transformers to register qwen3_5_moe model type
    from unsloth import FastLanguageModel

    # THEN monkey-patch safetensors to load adapter weights to CPU (avoids OOM)
    import safetensors.torch
    _orig_sf_load = safetensors.torch.load_file
    def _cpu_load(filename, device=None):
        return _orig_sf_load(filename, device="cpu")
    safetensors.torch.load_file = _cpu_load
    import peft.utils.save_and_load
    peft.utils.save_and_load.safe_load_file = _cpu_load

    print("Loading model with LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/workspace/lora",
        max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    _enc = getattr(tokenizer, "tokenizer", tokenizer)
    print("Model loaded and ready.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 256)
    top_p = data.get("top_p", 0.9)

    text = _enc.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = _enc(text, return_tensors="pt")["input_ids"].to("cuda:0")

    t0 = time.time()
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=max_tokens,
        temperature=temperature, top_p=top_p,
        repetition_penalty=1.1, do_sample=True,
    )
    gen_time = time.time() - t0

    response = _enc.decode(
        outputs[0][inputs.shape[-1]:], skip_special_tokens=True
    ).strip()

    return jsonify({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs.shape[-1],
            "completion_tokens": outputs.shape[-1] - inputs.shape[-1],
        },
        "gen_time": round(gen_time, 2),
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080)
