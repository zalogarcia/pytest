#!/usr/bin/env python3
"""
Benchmark test: 30 conversations with automated ratings.
Compare fine-tuned model output against Sonnet baseline.

Ratings (per response):
  - length_ok: under 800 chars
  - no_banned: no emojis, hashtags, semicolons, em dashes, asterisks, banned words
  - no_thinking: no chain-of-thought leakage
  - tone_human: reads like a real SMS (subjective, logged for manual review)

Usage:
  python3 benchmark_test.py 2>&1 | tee benchmark_results.log
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import json
import time

# ─── SYSTEM PROMPT (same as deployment) ────────────────────

SYSTEM_PROMPT = """You are Sarah, the assistant for Apex Dental. You handle inbound messages and route prospects to the right solution.

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

# ─── 30 TEST CONVERSATIONS ─────────────────────────────────

TESTS = [
    # --- DFY WARM LEADS (6) ---
    {"id": 1, "cat": "DFY-warm", "msgs": ["Hi I saw your ad about AI for dental practices"]},
    {"id": 2, "cat": "DFY-warm", "msgs": ["Hey do you guys help with patient follow up"]},
    {"id": 3, "cat": "DFY-warm", "msgs": ["I own a med spa and Im losing leads because we cant call them back fast enough"]},
    {"id": 4, "cat": "DFY-warm", "msgs": ["Someone told me you build AI agents for local businesses"]},
    {"id": 5, "cat": "DFY-warm", "msgs": ["We get about 200 leads a month from Google ads but my front desk cant keep up"]},
    {"id": 6, "cat": "DFY-warm", "msgs": ["Im a chiropractor looking for help with lead follow up"]},

    # --- DFY MULTI-TURN / QUALIFICATION (5) ---
    {"id": 7, "cat": "DFY-qualify", "msgs": ["I run a plumbing company", "About 150 leads a month", "My guys take like 3-4 hours to call back", "What's it cost"]},
    {"id": 8, "cat": "DFY-qualify", "msgs": ["We're a roofing company in Texas", "Maybe 80 leads a month from ads", "How does your AI thing work"]},
    {"id": 9, "cat": "DFY-qualify", "msgs": ["I have a pest control business", "We get leads from Thumbtack and Google", "Whats the setup process like"]},
    {"id": 10, "cat": "DFY-qualify", "msgs": ["Hey I run an HVAC company", "Probably 100 calls a month we miss", "Can your AI handle phone calls too"]},
    {"id": 11, "cat": "DFY-qualify", "msgs": ["Im an orthodontist with 3 locations", "We need help qualifying leads before they talk to my treatment coordinators"]},

    # --- DFY OBJECTIONS (4) ---
    {"id": 12, "cat": "DFY-objection", "msgs": ["What do you guys do?", "$3k is way too expensive"]},
    {"id": 13, "cat": "DFY-objection", "msgs": ["How is this different from just hiring another receptionist"]},
    {"id": 14, "cat": "DFY-objection", "msgs": ["Sounds cool but I need to talk to my business partner first"]},
    {"id": 15, "cat": "DFY-objection", "msgs": ["We already use a CRM that does automated follow ups"]},

    # --- CMAA / COURSE (4) ---
    {"id": 16, "cat": "CMAA", "msgs": ["I want to learn how to sell AI chatbots to businesses"]},
    {"id": 17, "cat": "CMAA", "msgs": ["Is the AI agency course still available"]},
    {"id": 18, "cat": "CMAA", "msgs": ["I saw your course about building AI agents. Do I need coding experience"]},
    {"id": 19, "cat": "CMAA", "msgs": ["How much is the course and what do I get"]},

    # --- DIY (3) ---
    {"id": 20, "cat": "DIY", "msgs": ["I just need something simple to send follow up texts to my leads"]},
    {"id": 21, "cat": "DIY", "msgs": ["Do you have anything cheaper than the full AI agent setup"]},
    {"id": 22, "cat": "DIY", "msgs": ["I want to try running my own ads and automating follow ups"]},

    # --- EDGE CASES (8) ---
    {"id": 23, "cat": "edge-AI", "msgs": ["Are you a real person or a bot?"]},
    {"id": 24, "cat": "edge-hostile", "msgs": ["This is a scam right"]},
    {"id": 25, "cat": "edge-hostile", "msgs": ["Stop texting me"]},
    {"id": 26, "cat": "edge-ghost", "msgs": ["Hey", "yeah sorry been busy"]},
    {"id": 27, "cat": "edge-ghost", "msgs": ["Hi"]},
    {"id": 28, "cat": "edge-offtopic", "msgs": ["What's the weather like today"]},
    {"id": 29, "cat": "edge-wrongnum", "msgs": ["Is this the pizza place"]},
    {"id": 30, "cat": "edge-friendly", "msgs": ["Lol you're good at this", "No seriously though what do you actually offer"]},
]

# ─── RATING FUNCTIONS ──────────────────────────────────────

BANNED_WORDS = [
    "additionally", "however", "moreover", "nevertheless",
    "transformative", "seamless", "delve", "facilitate",
    "utilize", "robust", "innovative", "optimization",
]
BANNED_CHARS = set(";\u2014\u2013*#")  # semicolon, em dash, en dash, asterisk, hashtag
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
    flags=re.UNICODE,
)
THINKING_MARKERS = ["thinking process:", "step 1:", "**analyze", "**determine", "chain of thought", "let me think"]


def rate_response(text):
    scores = {}
    # Length
    scores["length_ok"] = len(text) <= 800
    scores["char_count"] = len(text)

    # Banned words
    lower = text.lower()
    found_banned = [w for w in BANNED_WORDS if w in lower]
    scores["no_banned_words"] = len(found_banned) == 0
    scores["banned_words_found"] = found_banned

    # Banned chars + emoji
    found_chars = [c for c in text if c in BANNED_CHARS]
    has_emoji = bool(EMOJI_RE.search(text))
    scores["no_banned_chars"] = len(found_chars) == 0 and not has_emoji
    scores["banned_chars_found"] = found_chars
    scores["has_emoji"] = has_emoji

    # Thinking leakage
    has_thinking = any(m in lower for m in THINKING_MARKERS)
    scores["no_thinking"] = not has_thinking

    # Overall pass
    scores["pass"] = all([
        scores["length_ok"],
        scores["no_banned_words"],
        scores["no_banned_chars"],
        scores["no_thinking"],
    ])

    return scores


# ─── MAIN ──────────────────────────────────────────────────

def main():
    # Patch safetensors to load adapter weights to CPU (avoids OOM)
    import safetensors.torch
    _orig_sf_load = safetensors.torch.load_file
    def _cpu_load(filename, device=None):
        return _orig_sf_load(filename, device="cpu")
    safetensors.torch.load_file = _cpu_load
    import peft.utils.save_and_load
    peft.utils.save_and_load.safe_load_file = _cpu_load

    from unsloth import FastLanguageModel

    print("Loading model with LoRA adapters...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/workspace/lora",
        max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.\n")

    _enc = getattr(tokenizer, "tokenizer", tokenizer)

    results = []
    total_pass = 0
    total_responses = 0

    print("=" * 70)
    print(f"  BENCHMARK: 30 conversations, {sum(len(t['msgs']) for t in TESTS)} total responses")
    print("=" * 70)

    for test in TESTS:
        print(f"\n--- [{test['id']:02d}] {test['cat']} ---")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for user_msg in test["msgs"]:
            messages.append({"role": "user", "content": user_msg})
            print(f"  User: {user_msg}")

            text = _enc.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            inputs = _enc(text, return_tensors="pt")["input_ids"].to("cuda:0")

            t0 = time.time()
            outputs = model.generate(
                input_ids=inputs, max_new_tokens=256,
                temperature=0.7, top_p=0.9,
                repetition_penalty=1.1, do_sample=True,
            )
            gen_time = time.time() - t0

            response = _enc.decode(
                outputs[0][inputs.shape[-1]:], skip_special_tokens=True
            ).strip()

            messages.append({"role": "assistant", "content": response})
            scores = rate_response(response)
            total_responses += 1
            if scores["pass"]:
                total_pass += 1

            flag = "PASS" if scores["pass"] else "FAIL"
            print(f"  Sarah: {response}")
            print(f"  [{scores['char_count']} chars | {gen_time:.1f}s | {flag}]")

            if not scores["pass"]:
                fails = []
                if not scores["length_ok"]: fails.append(f"over 800 ({scores['char_count']})")
                if not scores["no_banned_words"]: fails.append(f"banned words: {scores['banned_words_found']}")
                if not scores["no_banned_chars"]: fails.append(f"banned chars/emoji")
                if not scores["no_thinking"]: fails.append("thinking leakage")
                print(f"  FAILURES: {', '.join(fails)}")

            results.append({
                "test_id": test["id"],
                "category": test["cat"],
                "user": user_msg,
                "response": response,
                "scores": {k: v for k, v in scores.items() if k != "banned_chars_found"},
                "gen_time": round(gen_time, 2),
            })

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total responses: {total_responses}")
    print(f"  Passed:          {total_pass}/{total_responses} ({100*total_pass/total_responses:.0f}%)")
    print(f"  Failed:          {total_responses - total_pass}/{total_responses}")

    # Category breakdown
    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"total": 0, "pass": 0, "avg_chars": 0}
        cats[c]["total"] += 1
        cats[c]["pass"] += 1 if r["scores"]["pass"] else 0
        cats[c]["avg_chars"] += r["scores"]["char_count"]

    print(f"\n  {'Category':<20} {'Pass':>6} {'Avg Chars':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*10}")
    for c, v in sorted(cats.items()):
        avg = v["avg_chars"] / v["total"]
        print(f"  {c:<20} {v['pass']}/{v['total']:>3}   {avg:>7.0f}")

    # Avg generation time
    avg_time = sum(r["gen_time"] for r in results) / len(results)
    print(f"\n  Avg generation time: {avg_time:.1f}s per response")

    # Save JSON results
    with open("/workspace/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to /workspace/benchmark_results.json")
    print("  DONE")


if __name__ == "__main__":
    main()
