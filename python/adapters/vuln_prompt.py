"""The fixed prompt of the vulnerability environment (tasks/vuln), in one place.

Used by scripts/prepare_vuln.py (token-length filter) and python/adapters/vuln_worker.py (the model's input).
tasks/vuln/prelude.py runs inside the sandbox and cannot import this file, so it repeats chat_text;
tests/test_vuln_worker.py checks that the two agree."""
QUESTION = "Can calling entry() reach a NULL pointer dereference on any execution path? Answer YES or NO."
YES, NO = 13602, 8725  # single tokens "YES" and "NO" in the Qwen3.5 tokenizer


def question(code):
    """The user message for one program."""
    return f"Read this C program.\n\n```c\n{code}\n```\n\n{QUESTION}"


def chat_text(code):
    """The exact text the model reads: Qwen3.5's chat template with thinking off (equal to
    tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False))."""
    return "<|im_start|>user\n" + question(code) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
