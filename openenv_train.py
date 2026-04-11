from env_server import KernelOptimization_env, TASKS
from trl import GRPOConfig, GRPOTrainer
from models import Action
from typing import List
from datasets import Dataset
import os

class KernelOptTool:
    def __init__(self):
        self.env = KernelOptimization_env()
        self.reward = 0.0
        self.done = False
    
    def reset(self, **kwargs) -> str:
        task_id = kwargs.get("task_id")
        if task_id is None and isinstance(kwargs.get("sample"), dict):
            task_id = kwargs["sample"].get("task_id")
        result = self.env.reset(task_id=task_id)
        obs = result["observation"]
        self.reward = 0.0
        self.done = False
        return (
            f"Task: {obs['task_name']}\n"
            f"Baseline CUDA kernel:\n{obs['baseline_code']}\n"
            f"Pending checks: {obs['pending_checks']}\n"
            "Use tools to submit improved code."
        )

    def submit_optimization(self, optimized_code: str, strategy: str = "", expected_speedup: float | None = None) -> str:
        if self.done:
            raise ValueError("Episode is already done.")
        result = self.env.step(
            Action(optimized_code=optimized_code, strategy=strategy, expected_speedup=expected_speedup)
        )
        self.reward = result.reward.value
        self.done = result.done
        obs = result.observation
        return (
            f"reward={result.reward.value:.4f}, "
            f"best_speedup={obs.current_best_speedup:.3f}x, "
            f"pending_checks={obs.pending_checks}, done={result.done}"
        )

    # Backward-compatible alias
    def submit_optiization(self, optimized_code: str, strategy: str = "") -> str:
        return self.submit_optimization(optimized_code=optimized_code, strategy=strategy)

def reward_func(environments, **kwargs) -> List[float]:
    return [env.reward for env in environments]

def build_dataset(repeats_per_task:int=32)-> Dataset:
    prompts, task_ids = [], []
    for task_id, task in TASKS.items():
        for _ in range(repeats_per_task):
            prompts.append([{"role": "user", "content": f"Optimize CUDA kernel task: {task['name']}"}])
            task_ids.append(task_id)
    return Dataset.from_dict({"prompt": prompts, "task_id": task_ids})

def main():
    model_name =os.getenv("TRAIN_MODEL", "Qwen/Qwen3-0.6B")
    dataset = build_dataset()
    trainer = GRPOTrainer(
        model=model_name,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=KernelOptTool,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            max_completion_length=2048,
            num_generations=4,
            log_completions=True,
        ),
    )
    trainer.train()
# trainer = GRPOTrainer(model =model_name, train_dataset=dataset, reward_funcs =reward_func, env_factory=KernelOptTool)

if __name__ == "__main__":
    main()
