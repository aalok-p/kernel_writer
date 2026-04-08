

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from flask import Flask, request, jsonify
from cuda_kernel_env.llm_optimizer import LLMKernelOptimizer
import random


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

TASKS = {
    "vector_add_opt": {
        "name": "Vector Addition Optimization",
        "difficulty": "easy",
        "max_iterations": 5,
        "target_speedup": 2.0,
        "baseline_code": """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
""",
    },
    "matmul_opt": {
        "name": "Matrix Multiplication Optimization",
        "difficulty": "medium",
        "max_iterations": 10,
        "target_speedup": 5.0,
        "baseline_code": """
__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
""",
    },
    "attention_opt": {
        "name": "Attention Kernel Optimization",
        "difficulty": "hard",
        "max_iterations": 15,
        "target_speedup": 3.0,
        "baseline_code": """
__global__ void attention_kernel(
    const float* Q, const float* K, const float* V,
    float* O, int seq_len, int d_model
) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    float scale = 1.0f / sqrtf((float)d_model);
    
    // Compute attention scores
    float max_score = -1e9f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        for (int d = 0; d < d_model; d++) {
            score += Q[q_idx * d_model + d] * K[k_idx * d_model + d];
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }
    
    // Softmax
    float sum_exp = 0.0f;
    float scores[1024]; // Assume max seq_len
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        for (int d = 0; d < d_model; d++) {
            score += Q[q_idx * d_model + d] * K[k_idx * d_model + d];
        }
        score = expf((score * scale) - max_score);
        scores[k_idx] = score;
        sum_exp += score;
    }
    
    // Weighted sum
    for (int d = 0; d < d_model; d++) {
        float output = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            output += (scores[k_idx] / sum_exp) * V[k_idx * d_model + d];
        }
        O[q_idx * d_model + d] = output;
    }
}
""",
    },
    "reduction_opt": {
        "name": "Reduction Kernel Optimization",
        "difficulty": "medium",
        "max_iterations": 8,
        "target_speedup": 4.0,
        "baseline_code": """
__global__ void reduce_sum(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Sequential reduction
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0 && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
""",
    }
}

class Environment:
    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.current_task: Optional[Dict] = None
        self.iteration: int = 0
        self.total_reward: float = 0.0
        self.baseline_metrics: Dict = {}
        self.current_metrics: Dict = {}
        self.optimization_history: list = []
        self.llm_optimizer: Optional[LLMKernelOptimizer] = None
        
    def initialize_optimizers(self):
        # Initialize LLM and GPU optimizers with environment variables
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME", "local-model")
        api_key = os.getenv("HF_TOKEN", "not-needed")
        
        # Determine provider from API_BASE_URL
        if "openai.com" in api_base_url:
            model_provider = "openai"
        else:
            model_provider = "lmstudio"
            os.environ["LM_STUDIO_BASE_URL"] = api_base_url
        
        self.llm_optimizer = LLMKernelOptimizer(
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key,
        )


env = Environment()

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "cuda-kernel-optimizer",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route("/reset", methods=["POST"])
def reset():
    try:
        data = request.get_json() or {}
        task_id = data.get("task_id")
        
        # Select task
        if task_id and task_id in TASKS:
            env.current_task_id = task_id
        else:
            env.current_task_id = random.choice(list(TASKS.keys()))
        
        env.current_task = TASKS[env.current_task_id].copy()
        env.iteration = 0
        env.total_reward = 0.0
        env.optimization_history = []
        
        # Initialize optimizers if not already done
        if env.llm_optimizer is None:
            env.initialize_optimizers()
        
        # Get baseline metrics (simulated)
        baseline_time = 100.0  # ms
        env.baseline_metrics = {
            "execution_time_ms": baseline_time,
            "bandwidth_gb_s": 50.0,
            "occupancy": 0.25,
            "speedup": 1.0
        }
        env.current_metrics = env.baseline_metrics.copy()
        
        observation = {
            "kernel_code": env.current_task["baseline_code"],
            "metrics": env.current_metrics,
            "iteration": env.iteration,
            "task_id": env.current_task_id
        }
        
        info = {
            "task_name": env.current_task["name"],
            "difficulty": env.current_task["difficulty"],
            "max_iterations": env.current_task["max_iterations"],
            "target_speedup": env.current_task["target_speedup"]
        }
        
        logger.info(f"Environment reset to task: {env.current_task_id}")
        
        return jsonify({
            "observation": observation,
            "info": info
        }), 200
        
    except Exception as e:
        logger.error(f"Error in reset: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/step", methods=["POST"])
def step():
    try:
        if env.current_task_id is None:
            return jsonify({"error": "Environment not initialized. Call /reset first."}), 400
        
        data = request.get_json()
        action = data.get("action", {})
        
        if "optimized_code" not in action:
            return jsonify({"error": "Action must contain 'optimized_code'"}), 400
        
        optimized_code = action["optimized_code"]
        reasoning = action.get("reasoning", "")
        
        env.iteration += 1
        
        # Simulate compilation and profiling
        compilation_success = True  # Assume success for simulation
        
        if compilation_success:
            # Simulate improved metrics
            improvement_factor = 1.0 + (env.iteration * 0.15)  # Gradual improvement
            new_time = env.baseline_metrics["execution_time_ms"] / improvement_factor
            speedup = env.baseline_metrics["execution_time_ms"] / new_time
            
            env.current_metrics = {
                "execution_time_ms": new_time,
                "bandwidth_gb_s": min(200.0, 50.0 * improvement_factor),
                "occupancy": min(0.95, 0.25 + env.iteration * 0.1),
                "speedup": speedup
            }
            
            correctness = 1.0  # Assume correct for simulation
        else:
            # Compilation failed
            env.current_metrics = env.baseline_metrics.copy()
            speedup = 1.0
            correctness = 0.0
        
        # Calculate reward (0.0-1.0 range)
        target_speedup = env.current_task["target_speedup"]
        normalized_speedup = min(speedup / target_speedup, 1.0)
        code_quality = env.current_metrics["occupancy"] * 0.5 + \
                      (env.current_metrics["bandwidth_gb_s"] / 200.0) * 0.5
        
        reward = (
            0.4 * normalized_speedup +
            0.4 * correctness +
            0.2 * code_quality
        )
        reward = max(0.0, min(1.0, reward))  # Clamp to [0.0, 1.0]
        
        env.total_reward += reward
        
        # Check if done
        done = (
            env.iteration >= env.current_task["max_iterations"] or
            speedup >= target_speedup
        )
        
        observation = {
            "kernel_code": optimized_code,
            "metrics": env.current_metrics,
            "iteration": env.iteration,
            "task_id": env.current_task_id
        }
        
        info = {
            "reasoning": reasoning,
            "compilation_success": compilation_success,
            "correctness": correctness,
            "target_reached": speedup >= target_speedup
        }
        
        # Log history
        env.optimization_history.append({
            "iteration": env.iteration,
            "speedup": speedup,
            "reward": reward,
            "reasoning": reasoning[:200]  # Truncate for storage
        })
        
        logger.info(f"Step {env.iteration}: speedup={speedup:.2f}x, reward={reward:.3f}")
        
        return jsonify({
            "observation": observation,
            "reward": float(reward),
            "done": done,
            "info": info
        }), 200
        
    except Exception as e:
        logger.error(f"Error in step: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/state", methods=["GET"])
def get_state():
    try:
        if env.current_task_id is None:
            return jsonify({
                "initialized": False,
                "message": "Environment not initialized. Call /reset first."
            }), 200
        
        return jsonify({
            "initialized": True,
            "current_task": env.current_task_id,
            "task_name": env.current_task["name"],
            "iteration": env.iteration,
            "max_iterations": env.current_task["max_iterations"],
            "total_reward": float(env.total_reward),
            "metrics": env.current_metrics,
            "target_speedup": env.current_task["target_speedup"],
            "history_length": len(env.optimization_history)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_state: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"Starting CUDA Kernel Optimizer API server on port {port}")
    logger.info(f"API_BASE_URL: {os.getenv('API_BASE_URL', 'not set')}")
    logger.info(f"MODEL_NAME: {os.getenv('MODEL_NAME', 'not set')}")
    app.run(host="0.0.0.0", port=port, debug=False)
