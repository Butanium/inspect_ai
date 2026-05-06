"""Minimal eval to verify LoRA adapter name appears in logs.

Uses SmolLM2-135M-Instruct + real LoRA adapters (same as test suite).
"""

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

SMOL_BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOL_LORA = "jekunz/smollm-135m-lora-fineweb-swedish"

SAMPLES = [Sample(input="Hej! Hur mår du?", target="bra")]


@task
def tiny_eval():
    return Task(dataset=SAMPLES, solver=generate(), scorer=includes())


if __name__ == "__main__":
    # 1. Without adapter
    results = eval(
        tiny_eval(),
        model=f"vllm/{SMOL_BASE}",
        model_args=dict(gpu_memory_utilization=0.5),
        max_tokens=10,
        limit=1,
    )
    log = results[0]
    print("\n=== WITHOUT ADAPTER ===")
    print(f"EvalSpec.model = {log.eval.model}")

    # 2. With adapter
    results = eval(
        tiny_eval(),
        model=f"vllm/{SMOL_BASE}:{SMOL_LORA}",
        model_args=dict(gpu_memory_utilization=0.5),
        max_tokens=10,
        limit=1,
    )
    log = results[0]
    print("\n=== WITH ADAPTER ===")
    print(f"EvalSpec.model = {log.eval.model}")
    print(f"Log location   = {log.location}")
