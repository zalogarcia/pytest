#!/usr/bin/env python3
"""
Test the fine-tuned model with sample conversations.

Usage:
  # Test LoRA adapters (default)
  python3 test_model.py

  # Test merged model
  python3 test_model.py --model ./output/merged_bf16

  # Test with custom system prompt (simulates real deployment)
  python3 test_model.py --system-prompt "You are Sarah, the assistant for Apex Dental..."

  # Interactive mode
  python3 test_model.py --interactive
"""

import argparse
import json

# ─── TEST SCENARIOS ─────────────────────────────────────────

# These simulate real deployment with filled-in template variables
DEFAULT_SYSTEM_PROMPT = """You are Sarah, the assistant for Apex Dental. You handle inbound messages and route prospects to the right solution.

CRITICAL: Every response MUST be under 800 characters. Keep it short, punchy, and conversational -- like texting a friend who happens to be really good at sales. One idea per message. Vary your length naturally: sometimes a few words, sometimes a couple sentences.

Tone: Witty, direct, confident but not pushy. Match the lead's energy. Never over-enthusiastic. Think confident friend giving advice, not salesperson.

Products:
- Done-For-You AI Agent: Custom-built AI agent that handles lead follow-up, qualification, and booking 24/7. Setup $3,000 + $99/month.
- DIY Marketing Platform: Self-serve platform for running ads and automating follow-ups. $199/month with free trial.
- AI Agency Course: Learn to build and sell AI chatbots to other businesses. $37 one-time.

Links:
- DFY info page: https://apexdental.com/ai-agent
- DIY free trial: https://apexdental.com/platform
- Course signup: https://apexdental.com/course
- Book a demo: https://apexdental.com/demo

Rules:
- Same question 3x = hand off to human
- Video/link permission: ask max 2x, then try different angle
- Booking link: share max 3x total
- If asked if you're AI, tell the truth
- Never use emojis, hashtags, semicolons, em dashes, or asterisks
- Never use: Additionally, However, Moreover, Nevertheless, Transformative, Seamless, Delve, Facilitate, Utilize, Robust, Innovative, Optimization"""

TEST_CONVERSATIONS = [
    {
        "name": "DFY -- Dental practice (warm lead)",
        "messages": [
            "Hi I saw your ad about AI for dental practices",
        ],
    },
    {
        "name": "DFY -- Objection handling (price)",
        "messages": [
            "What do you guys do?",
            "$3k is way too expensive",
        ],
    },
    {
        "name": "CMAA -- Course buyer",
        "messages": [
            "I want to learn how to sell AI chatbots to businesses",
        ],
    },
    {
        "name": "Edge -- AI disclosure",
        "messages": [
            "Are you a real person or a bot?",
        ],
    },
    {
        "name": "Edge -- Hostile lead",
        "messages": [
            "This is a scam right",
        ],
    },
    {
        "name": "Ghost recovery",
        "messages": [
            "Hey",
            "yeah sorry been busy",
        ],
    },
    {
        "name": "Multi-turn qualification",
        "messages": [
            "I run a plumbing company",
            "About 150 leads a month",
            "My guys take like 3-4 hours to call back",
            "What's it cost",
        ],
    },
]


def run_tests(model_path, system_prompt):
    """Load model and run test conversations."""
    from unsloth import FastLanguageModel

    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    print(f"\n{'='*60}")
    print("RUNNING TEST CONVERSATIONS")
    print(f"{'='*60}\n")

    for test in TEST_CONVERSATIONS:
        print(f"--- {test['name']} ---")
        messages = [{"role": "system", "content": system_prompt}]

        for user_msg in test["messages"]:
            messages.append({"role": "user", "content": user_msg})
            print(f"  User: {user_msg}")

            # Generate response
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
            )

            # Decode only the new tokens
            response = tokenizer.decode(
                outputs[0][inputs.shape[-1]:], skip_special_tokens=True
            ).strip()

            messages.append({"role": "assistant", "content": response})
            print(f"  Assistant: {response}")
            print(f"  [{len(response)} chars]")
            print()

        print()

    return model, tokenizer


def interactive_mode(model_path, system_prompt):
    """Interactive chat with the fine-tuned model."""
    from unsloth import FastLanguageModel

    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n{'='*60}")
    print("INTERACTIVE MODE (type 'quit' to exit, 'reset' to restart)")
    print(f"{'='*60}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("[Conversation reset]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )

        response = tokenizer.decode(
            outputs[0][inputs.shape[-1]:], skip_special_tokens=True
        ).strip()

        messages.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}")
        print(f"[{len(response)} chars]\n")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument("--model", type=str, default="./output/lora_adapters",
                        help="Path to model (LoRA adapters or merged)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Custom system prompt (simulates real deployment)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat mode")
    args = parser.parse_args()

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    if args.interactive:
        interactive_mode(args.model, system_prompt)
    else:
        run_tests(args.model, system_prompt)


if __name__ == "__main__":
    main()
