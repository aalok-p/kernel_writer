from typing import List, Optional, Dict, Any
from models import Action, StepResult, ResetRequest, StepRequest, EnvState, Observation, Reward
from fastapi import FastAPI, HTTPException
import random

TASKS: Dict[str, Dict[str, Any]] ={
    "vector_add_easy": {
        "name": "Vector Addition Kernel Optimization",
        "difficulty": "easy",
        "max_steps": 5,
        "target_speedup": 1.8,
        "baseline_code": """extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) 
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) c[idx] = a[idx] + b[idx];
        }""",
        "checks": {
            "coalesced_memory": "Use memory-coalesced indexing",
            "vectorized_loads": "Use vectorized loads/stores (float2/float4)",
            "bounds_safe": "Keep safe boundary checks",
        },

    },
    "matmul_medium": {
        "name": "Matrix Multiplication Kernel Optimization",
        "difficulty": "medium",
        "max_steps": 6,
        "target_speedup": 3.0,
        "baseline_code": """extern "C" __global__ void matmul(const float* A, const float* B, float* C, int N) 
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < N && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];
            C[row * N + col] = sum;
            }
        }""",
        "checks": {
            "shared_tiling": "Use shared-memory tiling",
            "synchronization": "Synchronize tiles with __syncthreads",
            "register_accumulation": "Accumulate partial sums in registers",
        },
    },
    "reduction_hard": {
        "name": "Reduction Kernel Optimization",
        "difficulty": "hard",
        "max_steps":7,
        "target_speedup": 3.5,
        "baseline_code": """extern "C" __global__ void reduce_sum(const float* input, float* output, int n) 
        {
            extern __shared__ float sdata[];
            int tid = threadIdx.x;
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            sdata[tid] = (i < n) ? input[i] : 0.0f;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
            }
            if (tid == 0) output[blockIdx.x] = sdata[0];
        }""",
        "checks": {
            "warp_primitive": "Use warp-level primitive (e.g., __shfl_down_sync)",
            "bank_conflict_reduction": "Reduce shared-memory bank conflicts",
            "unrolled_reduction": "Use partial unrolling for final reduction",
        },
    }
}

def check_passed(check_id:str, code_lower:str) ->bool:
    if check_id =="coalesced_memory":
        return "idx" in code_lower and ("blockidx.x" in code_lower or "threadidx.x" in code_lower)
    if check_id == "vectorized_loads":
        return "float4" in code_lower or "float2" in code_lower
    if check_id == "bounds_safe":
        return "if" in code_lower and "< n" in code_lower
    if check_id == "shared_tiling":
        return "__shared__" in code_lower
    if check_id == "synchronization":
        return "__syncthreads" in code_lower
    if check_id == "register_accumulation":
        return "sum" in code_lower or "acc" in code_lower
    if check_id == "warp_primitive":
        return "__shfl_down_sync" in code_lower or "__shfl_sync" in code_lower
    if check_id =="bank_conflict_reduction":
        return "pad" in code_lower or "bank" in code_lower or "+ 1" in code_lower
    if check_id == "unrolled_reduction":
        return "#pragma unroll" in code_lower or "unroll" in code_lower
    return False

def to_observation(task_id:str, state:EnvState)->Observation:
    task = TASKS[task_id]
    pending = [desc for cid, desc in task["checks"].items() if cid not in set(state.completed_checks)]
    return Observation(task_id=task_id, task_name=task["name"], difficulty=task["difficulty"], baseline_code=task["baseline_code"], current_best_code=state.best_code or task["baseline_code"], current_best_speedup=state.best_speedup, step_count=state.step_count, max_steps=state.max_steps, pending_checks=pending, completed_checks=[task["checks"][cid] for cid in state.completed_checks if cid in task["checks"]], done=(len(pending) == 0 or state.step_count >= state.max_steps))

def grade_episode(task_id:str, completed_checks:List[str], best_speedup:float, step_count:int, max_steps:int)->float:
    task=TASKS[task_id]
    completion =len(completed_checks) / max(len(task["checks"]),1)
    speedup_score = min(best_speedup /task["target_speedup"],1.0)
    efficiency = max(0.0, 1.0 - ((step_count - 1) / max(max_steps, 1)))
    return round(max(0.0, min(1.0, 0.5 * completion + 0.35 * speedup_score + 0.15 * efficiency)), 4)

class KernelOptimization_env:
    def __init__(self):
        self.state =EnvState(initialized=False)
        self.current_task_id: Optional[str]=None

    def reset(self, task_id:Optional[str]=None)->Dict[str, Any]:
        if task_id and task_id not in TASKS:
            raise HTTPException(status_code=400, detail=f"unknown task_id: {task_id}")
        self.current_task_id =task_id or random.choice(list(TASKS.keys()))
        task= TASKS[self.current_task_id]
        self.state =EnvState(initialized=True, task_id=self.current_task_id, step_count=0, max_steps=task["max_steps"], total_reward=0.0, best_code=task["baseline_code"], best_speedup=1.0, completed_checks=[], action_history=[])
        return {
            "observation": to_observation(self.current_task_id, self.state).model_dump(),
            "info": {
                "task_id": self.current_task_id,
                "task_name": task["name"],
                "difficulty": task["difficulty"],
                "max_steps": task["max_steps"],
                "target_speedup": task["target_speedup"],
                "checks": task["checks"],
            },
        }
    
    def step(self, action:Action) ->StepResult:
        if not self.state.initialized or not self.current_task_id:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

        self.state.step_count += 1
        code = action.optimized_code or ""
        code_lower = code.lower()
        compile_ok = "__global__" in code_lower and "{" in code and "}" in code

        completed = set(self.state.completed_checks)
        newly_completed = {cid for cid in TASKS[self.current_task_id]["checks"] if cid not in completed and check_passed(cid, code_lower)}
        completed.update(newly_completed)
        self.state.completed_checks = sorted(completed)

        completion_ratio = len(completed) / max(len(TASKS[self.current_task_id]["checks"]), 1)
        max_reasonable_speedup = 1.0 + completion_ratio * 3.0
        if action.expected_speedup is None:
            est_speedup = round(max_reasonable_speedup, 3)
        else:
            est_speedup = round(max(1.0, min(action.expected_speedup, max_reasonable_speedup)), 3)
        if est_speedup > self.state.best_speedup:
            self.state.best_speedup = est_speedup
            self.state.best_code = code

        progress = 0.22 * len(newly_completed)
        quality = 0.18 * min(self.state.best_speedup / TASKS[self.current_task_id]["target_speedup"], 1.0)
        penalty = 0.0
        if not compile_ok:
            penalty -= 0.25
        if not newly_completed:
            penalty -= 0.08
        reward_value = max(0.0, min(1.0, progress + quality + penalty))
        self.state.total_reward += reward_value

        self.state.action_history.append(
            {
                "step": self.state.step_count,
                "newly_completed": sorted(newly_completed),
                "compile_ok": compile_ok,
                "estimated_speedup": est_speedup,
                "reward": reward_value,
            }
        )

        obs =to_observation(self.current_task_id, self.state)
        info: Dict[str, Any] = { "compile_ok": compile_ok, "estimated_speedup": est_speedup}
        if obs.done:
            info["final_score"] = grade_episode(
                self.current_task_id, self.state.completed_checks, self.state.best_speedup, self.state.step_count, self.state.max_steps
            )

        return StepResult(
            observation=obs,
            reward=Reward(
                value=round(reward_value, 4),
                components={"progress": round(progress, 4), "quality": round(quality, 4), "penalty": round(penalty, 4)},
            ),
            done=obs.done,
            info=info,
        )
    def state_dict(self)->Dict[str, Any]:
        data = self.state.model_dump()
        if self.current_task_id:
            data["task_name"] = TASKS[self.current_task_id]["name"]
            data["difficulty"] = TASKS[self.current_task_id]["difficulty"]
            data["grader_score"] = grade_episode(
                self.current_task_id, self.state.completed_checks, self.state.best_speedup, self.state.step_count, self.state.max_steps
            )
        return data

env=KernelOptimization_env()
app=FastAPI(title="Kernel Optimization", version="1.0.0")

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "kernel-optimization-openenv"}

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    return env.reset(task_id=request.task_id)


@app.post("/step")
def step(request: StepRequest):
    return env.step(request.action).model_dump()


@app.get("/state")
def state():
    return env.state_dict()
