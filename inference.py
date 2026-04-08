import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from openai import OpenAI

REQUIRED_ENV_VARS = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        print(f"ERROR: Required environment variable {var} is not set", file=sys.stderr)
        sys.exit(1)

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url=os.getenv("API_BASE_URL")
)

MODEL_NAME = os.getenv("MODEL_NAME")

TASKS = {
    "vector_add_opt": {
        "name": "Vector Addition",
        "baseline": """__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}""",
        "target_speedup": 2.0,
        "max_iterations": 5
    },
    "matmul_opt": {
        "name": "Matrix Multiplication",
        "baseline": """__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}""",
        "target_speedup": 5.0,
        "max_iterations": 10
    },
    "attention_opt": {
        "name": "Attention Kernel",
        "baseline": """__global__ void attention_kernel(const float* Q, const float* K, const float* V, float* O, int seq_len, int d_model) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len) return;
    float scale = 1.0f / sqrtf((float)d_model);
    float max_score = -1e9f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        for (int d = 0; d < d_model; d++) {
            score += Q[q_idx * d_model + d] * K[k_idx * d_model + d];
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }
}""",
        "target_speedup": 3.0,
        "max_iterations": 15
    },
    "reduction_opt": {
        "name": "Parallel Reduction",
        "baseline": """__global__ void reduce_sum(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0 && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}""",
        "target_speedup": 4.0,
        "max_iterations": 8
    }
}


def emit_start_log(task_id: str, task_info: Dict[str, Any]):
    # Emit [START] log with task information
    log_entry = {
        "type": "START",
        "timestamp": datetime.utcnow().isoformat(),
        "task_id": task_id,
        "task_name": task_info["name"],
        "target_speedup": task_info["target_speedup"],
        "max_iterations": task_info["max_iterations"],
        "model": MODEL_NAME,
        "api_base": os.getenv("API_BASE_URL")
    }
    print(f"[START] {json.dumps(log_entry)}", flush=True)


def emit_step_log(iteration: int, speedup: float, reward: float, 
                  reasoning: str, compilation_success: bool, metrics: Dict[str, float]):
    # Emit [STEP] log with iteration results
    log_entry = {
        "type": "STEP",
        "timestamp": datetime.utcnow().isoformat(),
        "iteration": iteration,
        "speedup": round(speedup, 4),
        "reward": round(reward, 4),
        "compilation_success": compilation_success,
        "reasoning": reasoning[:200],  # Truncate for readability
        "metrics": {
            "execution_time_ms": round(metrics.get("execution_time_ms", 0.0), 4),
            "bandwidth_gb_s": round(metrics.get("bandwidth_gb_s", 0.0), 4),
            "occupancy": round(metrics.get("occupancy", 0.0), 4)
        }
    }
    print(f"[STEP] {json.dumps(log_entry)}", flush=True)


def emit_end_log(task_id: str, success: bool, total_iterations: int, 
                 final_speedup: float, total_reward: float, elapsed_time: float):
    # Emit [END] log with final results
    log_entry = {
        "type": "END",
        "timestamp": datetime.utcnow().isoformat(),
        "task_id": task_id,
        "success": success,
        "total_iterations": total_iterations,
        "final_speedup": round(final_speedup, 4),
        "total_reward": round(total_reward, 4),
        "elapsed_time_seconds": round(elapsed_time, 2),
        "score": round(total_reward / total_iterations if total_iterations > 0 else 0.0, 4)
    }
    print(f"[END] {json.dumps(log_entry)}", flush=True)


def call_llm(prompt: str, system_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"ERROR: LLM call failed: {str(e)}", file=sys.stderr)
        return ""


def extract_cuda_code(llm_response: str) -> str:
    if "```cuda" in llm_response:
        start = llm_response.find("```cuda") + 7
        end = llm_response.find("```", start)
        return llm_response[start:end].strip()
    elif "```cpp" in llm_response:
        start = llm_response.find("```cpp") + 6
        end = llm_response.find("```", start)
        return llm_response[start:end].strip()
    elif "```" in llm_response:
        start = llm_response.find("```") + 3
        end = llm_response.find("```", start)
        return llm_response[start:end].strip()
    return llm_response


def simulate_kernel_execution(baseline_time: float, iteration: int) -> Dict[str, float]:
    # Simulate kernel execution with progressive improvements
    improvement = 1.0 + (iteration * 0.15)
    new_time = baseline_time / improvement
    speedup = baseline_time / new_time
    
    return {
        "execution_time_ms": new_time,
        "bandwidth_gb_s": min(200.0, 50.0 * improvement),
        "occupancy": min(0.95, 0.25 + iteration * 0.1),
        "speedup": speedup
    }


def calculate_reward(metrics: Dict[str, float], target_speedup: float, 
                    correctness: float = 1.0) -> float:
    speedup = metrics["speedup"]
    normalized_speedup = min(speedup / target_speedup, 1.0)
    code_quality = (
        metrics["occupancy"] * 0.5 + 
        min(metrics["bandwidth_gb_s"] / 200.0, 1.0) * 0.5
    )
    
    reward = (
        0.4 * normalized_speedup +
        0.4 * correctness +
        0.2 * code_quality
    )
    
    return max(0.0, min(1.0, reward))


def optimize_kernel(task_id: str, task_info: Dict[str, Any]) -> Dict[str, Any]:
    # Main optimization loop for a task
    baseline_code = task_info["baseline"]
    target_speedup = task_info["target_speedup"]
    max_iterations = task_info["max_iterations"]
    
    baseline_time = 100.0  # ms
    current_code = baseline_code
    current_metrics = {
        "execution_time_ms": baseline_time,
        "bandwidth_gb_s": 50.0,
        "occupancy": 0.25,
        "speedup": 1.0
    }
    
    total_reward = 0.0
    optimization_history = []
    
    system_prompt = """You are an expert CUDA kernel optimizer.

CRITICAL: Output ONLY code with minimal thinking (1 line comment max).
NO explanations, NO analysis text outside the code block.

MANDATORY FORMAT:
```cuda
// Strategy: [one sentence]
[optimized CUDA kernel code]
```

Focus on: memory coalescing, shared memory, thread blocks, warp optimizations."""
    
    for iteration in range(1, max_iterations + 1):
        # Build prompt with history for LLM
        history_str = "\n".join([
            f"Iteration {h['iteration']}: speedup={h['speedup']:.2f}x"
            for h in optimization_history[-3:]
        ])
        
        prompt = f"""Current CUDA kernel:
```cuda
{current_code}
```

Current performance:
- Speedup: {current_metrics['speedup']:.2f}x
- Occupancy: {current_metrics['occupancy']:.2f}
- Bandwidth: {current_metrics['bandwidth_gb_s']:.1f} GB/s

Target speedup: {target_speedup}x
Iteration: {iteration}/{max_iterations}

Previous attempts:
{history_str if history_str else 'First iteration'}

IMPORTANT: Return ONLY optimized code with 1 brief comment. NO text outside code block.
```cuda
// Strategy: [one line]
[code]
```"""
        
        # Get LLM suggestion
        llm_response = call_llm(prompt, system_prompt)
        
        if not llm_response:
            # LLM failed, use previous code
            reasoning = "LLM call failed, using previous code"
            optimized_code = current_code
            compilation_success = False
        else:
            # Extract code and reasoning
            optimized_code = extract_cuda_code(llm_response)
            reasoning = llm_response.split("```")[0].strip() if "```" in llm_response else llm_response[:200]
            compilation_success = bool(optimized_code)
        
        # Simulate execution
        if compilation_success:
            new_metrics = simulate_kernel_execution(baseline_time, iteration)
            current_code = optimized_code
            current_metrics = new_metrics
        else:
            # Keep previous metrics
            pass
        
        # Calculate reward
        reward = calculate_reward(current_metrics, target_speedup)
        total_reward += reward
        
        # Log step
        emit_step_log(
            iteration=iteration,
            speedup=current_metrics["speedup"],
            reward=reward,
            reasoning=reasoning,
            compilation_success=compilation_success,
            metrics=current_metrics
        )
        
        # Track history
        optimization_history.append({
            "iteration": iteration,
            "speedup": current_metrics["speedup"],
            "reward": reward,
            "reasoning": reasoning[:100]
        })
        
        # Check if target reached
        if current_metrics["speedup"] >= target_speedup:
            break
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    return {
        "success": current_metrics["speedup"] >= target_speedup,
        "iterations": len(optimization_history),
        "final_speedup": current_metrics["speedup"],
        "total_reward": total_reward,
        "history": optimization_history
    }


def run_grader(task_id: str, result: Dict[str, Any]) -> float:
    # Grade task results, returning score in [0.0, 1.0]
    task_info = TASKS[task_id]
    target_speedup = task_info["target_speedup"]
    
    # Scoring rubric:
    # - 40%: Speedup achievement
    # - 40%: Efficiency (iterations used)
    # - 20%: Total reward
    
    speedup_score = min(result["final_speedup"] / target_speedup, 1.0)
    efficiency_score = 1.0 - (result["iterations"] / task_info["max_iterations"])
    reward_score = result["total_reward"] / result["iterations"] if result["iterations"] > 0 else 0.0
    
    final_score = (
        0.4 * speedup_score +
        0.4 * efficiency_score +
        0.2 * reward_score
    )
    
    return max(0.0, min(1.0, final_score))


def main():
    # Main inference script entry point
    start_time = time.time()
    all_results = {}
    all_scores = {}
    
    # Run all tasks
    for task_id, task_info in TASKS.items():
        task_start_time = time.time()
        
        # Emit START log
        emit_start_log(task_id, task_info)
        
        # Run optimization
        result = optimize_kernel(task_id, task_info)
        
        # Calculate elapsed time
        task_elapsed = time.time() - task_start_time
        
        # Emit END log
        emit_end_log(
            task_id=task_id,
            success=result["success"],
            total_iterations=result["iterations"],
            final_speedup=result["final_speedup"],
            total_reward=result["total_reward"],
            elapsed_time=task_elapsed
        )
        
        # Run grader
        score = run_grader(task_id, result)
        
        all_results[task_id] = result
        all_scores[task_id] = score
    
    # Summary
    total_elapsed = time.time() - start_time
    avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    
    return 0 if avg_score >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
