from __future__ import annotations

import argparse
import re
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from training.action_space import ActionSpec, build_action_space
from training.openenv_compat import ensure_training_runtime

ensure_training_runtime()

from models import SpaceFaultAction
from server.space_fault_recovery_environment import MAX_STEPS, SpaceFaultRecoveryEnvironment

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_ACTION = "diagnostic_scan:power"

_ACTION_SPACE: list[ActionSpec] = build_action_space()
_ACTION_LABELS: list[str] = [a.label for a in _ACTION_SPACE]
_ACTION_LOOKUP: dict[str, ActionSpec] = {a.label: a for a in _ACTION_SPACE}
_MODEL: AutoModelForCausalLM | None = None
_TOKENIZER: AutoTokenizer | None = None


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text"):
            if key in value and isinstance(value[key], str):
                return value[key]
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value)


def obs_to_prompt(obs: Any, *, seed: int) -> str:
    return (
        "You are a spacecraft fault-recovery controller.\n"
        "Choose exactly one valid action for the CURRENT telemetry.\n"
        f"Episode seed: {seed}\n"
        f"Step: {int(getattr(obs, 'step', 0))}\n"
        f"Battery: {float(getattr(obs, 'battery_pct', 0.0)):.2f}\n"
        f"Battery drain rate: {float(getattr(obs, 'battery_drain_rate', 0.0)):.3f}\n"
        f"Solar A output W: {float(getattr(obs, 'solar_a_sensor_output_w', 0.0)):.2f}\n"
        f"Solar B output W: {float(getattr(obs, 'solar_b_sensor_output_w', 0.0)):.2f}\n"
        f"Bus voltage: {float(getattr(obs, 'bus_voltage', 0.0)):.2f}\n"
        f"Star tracker err deg: {float(getattr(obs, 'star_tracker_deg', 0.0)):.3f}\n"
        f"Gyro err deg: {float(getattr(obs, 'gyro_deg', 0.0)):.3f}\n"
        f"Sun sensor err deg: {float(getattr(obs, 'sun_sensor_deg', 0.0)):.3f}\n"
        f"Fuel units: {float(getattr(obs, 'fuel_units', 0.0)):.2f}\n"
        f"Signal strength db: {float(getattr(obs, 'signal_strength_db', 0.0)):.2f}\n"
        f"Battery temp C: {float(getattr(obs, 'battery_temp_c', 0.0)):.2f}\n"
        f"Attitude mode: {getattr(obs, 'attitude_mode', 'unknown')}\n"
        f"RW status: {getattr(obs, 'rw_status', 'unknown')}\n"
        f"Transponder: {getattr(obs, 'transponder_status', 'unknown')}\n"
        f"Link bandwidth: {getattr(obs, 'link_bandwidth', 'unknown')}\n"
        f"Mission status: {getattr(obs, 'mission_status', 'unknown')}\n"
        f"Subsystems online: {', '.join(getattr(obs, 'subsystems_online', []) or [])}\n"
        f"Last action result: {getattr(obs, 'last_action_result', 'none')}\n"
        f"Valid actions: {', '.join(_ACTION_LABELS)}\n"
        "Respond with exactly one action label and no explanation."
    )


def build_prompt_dataset(*, num_prompts: int, seed_base: int) -> Dataset:
    env = SpaceFaultRecoveryEnvironment()
    prompts: list[str] = []
    episode_seeds: list[int] = []
    for idx in range(num_prompts):
        seed = seed_base + idx
        obs = env.reset(seed=seed)
        prompts.append(obs_to_prompt(obs, seed=seed))
        episode_seeds.append(seed)
    return Dataset.from_dict({"prompt": prompts, "episode_seed": episode_seeds})


def parse_action(text: str) -> ActionSpec:
    norm = _extract_text(text).strip().lower()
    for label in sorted(_ACTION_LABELS, key=len, reverse=True):
        if label in norm:
            return _ACTION_LOOKUP[label]
    return _ACTION_LOOKUP[FALLBACK_ACTION]


def _action_from_model(prompt: str) -> ActionSpec:
    if _MODEL is None or _TOKENIZER is None:
        return _ACTION_LOOKUP[FALLBACK_ACTION]
    encoded = _TOKENIZER(prompt, return_tensors="pt")
    encoded = {k: v.to(_MODEL.device) for k, v in encoded.items()}
    with torch.no_grad():
        generated = _MODEL.generate(
            **encoded,
            max_new_tokens=24,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=_TOKENIZER.eos_token_id,
        )
    new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
    text = _TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    return parse_action(text)


def _seed_from_prompt(prompt: str, fallback_seed: int = 0) -> int:
    match = re.search(r"Episode seed:\s*(\d+)", prompt)
    if match:
        return int(match.group(1))
    return fallback_seed


def _rollout_reward(seed: int, first_action: ActionSpec) -> float:
    env = SpaceFaultRecoveryEnvironment()
    obs = env.reset(seed=seed)
    total_reward = 0.0

    obs = env.step(first_action.to_action())
    total_reward += float(obs.reward or 0.0)

    while not bool(getattr(obs, "done", False)) and int(getattr(obs, "step", 0)) < MAX_STEPS:
        prompt = obs_to_prompt(obs, seed=seed)
        action = _action_from_model(prompt)
        obs = env.step(action.to_action())
        total_reward += float(obs.reward or 0.0)

    status = str(getattr(obs, "mission_status", ""))
    if status == "recovered":
        total_reward += 5.0
    elif status == "lost":
        total_reward -= 5.0
    return total_reward


def reward_fn(completions: list[Any], prompts: list[Any] | None = None, **kwargs: Any) -> list[float]:
    prompt_values = prompts or kwargs.get("prompt") or kwargs.get("prompts")
    if prompt_values is None:
        prompt_values = [""] * len(completions)

    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        prompt_text = _extract_text(prompt_values[idx])
        completion_text = _extract_text(completion)
        seed = _seed_from_prompt(prompt_text, fallback_seed=idx)
        first_action = parse_action(completion_text)
        try:
            reward = _rollout_reward(seed, first_action)
        except Exception as exc:
            print(f"reward_fn rollout error: {exc}")
            reward = -5.0
        rewards.append(float(reward))
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRL GRPO policy for space fault recovery.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default="trl-space-agent")
    parser.add_argument("--final-dir", default="trl-space-agent-final")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=40)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    global _MODEL, _TOKENIZER
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _TOKENIZER = tokenizer
    _MODEL = model

    dataset = build_prompt_dataset(num_prompts=args.num_prompts, seed_base=args.seed_base)
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.final_dir)
    tokenizer.save_pretrained(args.final_dir)


if __name__ == "__main__":
    main()
