import os
from openai import OpenAI, AuthenticationError
from typing import Dict
from env_server import TASKS, KernelOptimization_env, grade_episode
from models import Action
import json
import sys
from dotenv import load_dotenv

load_dotenv()
def extract_code(text: str) -> str:
    if "```" not in text:
        return text
    start = text.find("```")
    end = text.rfind("```")
    chunk = text[start + 3 : end]
    if chunk.startswith("cuda") or chunk.startswith("cpp"):
        return chunk.split("\n", 1)[1]
    return chunk

def choose_action(client: OpenAI, model: str, observation: Dict) -> Action:
    prompt = f"""Optimize this CUDA kernel.
    Task: {observation['task_name']}
    Pending checks: {observation['pending_checks']}
    Baseline:
    {observation['baseline_code']}
    Current best speedup: {observation['current_best_speedup']}x
    Return only optimized CUDA code.
"""
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a CUDA optimization expert. Return code only."},
            {"role": "user", "content": prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    code = extract_code(text).strip() or observation["current_best_code"]
    return Action(optimized_code=code, strategy="llm_proposed")

def run_task(client: OpenAI, model: str, task_id: str) -> float:
    env = KernelOptimization_env()
    obs = env.reset(task_id=task_id)["observation"]
    done = False
    while not done:
        action = choose_action(client, model, obs)
        step_result = env.step(action)
        obs = step_result.observation.model_dump()
        done = step_result.done
    return grade_episode(task_id, env.state.completed_checks, env.state.best_speedup, env.state.step_count, env.state.max_steps)
def main()->int:
    if not os.getenv("OPENAI_API_KEY"):
        print("openai key not set")

    model =os.getenv("MODEL_NAME", "gemma-3-4b")
    client =OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url =os.getenv("API_BASE_URL", "https://api.openai.com/v1"))

    scores: Dict[str, float] = {}
    try:
        for task_id in TASKS:
            scores[task_id] = run_task(client, model, task_id)
            print(f"[TASK] {task_id} score={scores[task_id]:.4f}")
    except AuthenticationError:
        print("ERROR: OpenAI authentication failed. Check OPENAI_API_KEY.", file=sys.stderr)
        return 1

    avg = sum(scores.values()) / len(scores)
    print(f"[BASELINE] model={model} average_score={avg:.4f}")
    print(json.dumps({"scores": scores, "average": round(avg, 4)}))
    return 0
if __name__=="__main__":
    sys.exit(main())
