"""Utility functions for answer extraction, verification, and prompt formatting."""

import re


def extract_boxed_answer(text):
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    matches = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        start = idx + 7
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    matches.append(text[start:j])
                    break
                depth -= 1
        i = idx + 1
    return matches[-1] if matches else None


def extract_answer(text):
    """Extract final answer from model output. Tries \\boxed{} first, then fallback patterns."""
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(answer):
    """Normalize answer string for comparison."""
    if answer is None:
        return None
    s = str(answer).strip()
    s = s.rstrip(".")
    s = s.replace("$", "").replace(" ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    return s


def verify_answer(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)
    if pred is None or gt is None:
        return False
    if pred == gt:
        return True
    # Try numeric comparison
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        pass
    return False


def format_base_prompt(question):
    """Format the base (unconditioned) prompt."""
    return (
        "You are a helpful math assistant. Solve the following problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"{question}"
    )


def format_conditioned_prompt(question, demonstration):
    """Format the solution-conditioned prompt (SDFT style)."""
    return (
        f"You are a helpful math assistant. The following is a math question:\n\n"
        f"{question}\n\n"
        f"This is an example for a response to the question:\n"
        f"{demonstration}\n\n"
        f"Now answer with a response of your own, including the thinking process. "
        f"Put your final answer in \\boxed{{}}."
    )
