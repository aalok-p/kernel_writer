from typing import Optional, Dict, Tuple
from llm_optimizer import LLMOptimizer, OptimizationAttempt
import os
import tempfile
import subprocess
import numpy as np

class GPUOptimizer:
    def __init__(self, baseline_code:str, kernel_name:str, model_provider: str ="lmstudio", model_name:str ="local_model", api_key:Optional[str] =None):
        """Args:
            baseline_code: Naive CUDA kernel
            kernel_name: Name of kernel function
            model_provider: LLM provider (lmstudio/openai)
            model_name: Model to use
            api_key: API key (or from env var)
        """
        self.baseline_code = baseline_code
        self.kernel_name =kernel_name

        self.llm_optimizer = LLMOptimizer(model_provider=model_provider, model_name=model_name, api_key= api_key or os.get_env("OPENAI_API"))
        self.baseline_time_ms =None
    
    def compile_kernel(self, cuda_code:str) -> Tuple[bool, Optional[str], Optional[str]]:
        try:
            # write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(cuda_code)
                cu_file = f.name
            
            #cmpile with nvcc
            ptx_file = cu_file.replace('.cu', '.ptx')
            result = subprocess.run(
                ['nvcc', '--ptx', '-o', ptx_file, cu_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            os.unlink(cu_file)
            if result.returncode == 0:
                #read ptx
                with open(ptx_file, 'r') as f:
                    ptx_code = f.read()
                os.unlink(ptx_file)
                return True, ptx_code, None
            else:
                if os.path.exists(ptx_file):
                    os.unlink(ptx_file)
                return False, None, result.stderr
                
        except Exception as e:
            return False, None, str(e)

    def profile_kernel(self, ptx_code:str, inputs:Dict)->Dict:
        return {'execution_time_ms': 0.0, 'occupancy': 0.0, 'memory_efficiency': 0.0, 'warp_divergence': 0.0,'shared_memory_bytes': 0,'registers_per_thread': 0,}
    
    def optimize(self, input_generator, validator, max_iterations: int=10, target_speedup: float=10.0) ->Tuple[str, float]:
        baseline_success, baseline_ptx, baseline_error = self.compile_kernel(self.baseline_code)
        
        if not baseline_success:
            return self.baseline_code, 1.0
        
        #profile baseline
        inputs = input_generator()
        baseline_metrics = self.profile_kernel(baseline_ptx, inputs)
        self.baseline_time_ms = baseline_metrics.get('execution_time_ms', 1.0)
        
        if self.baseline_time_ms == 0.0:
            self.baseline_time_ms = 1.0  
        
        best_code = self.baseline_code
        best_speedup = 1.0
        for iteration in range(max_iterations):
            optimized_code, reasoning = self.llm_optimizer.propose_optimization(
                self.baseline_code,
                current_metrics,
            )
            #compile
            success, ptx_code, error = self.compile_kernel(optimized_code)
            if not success:
                # Record failed attempt
                attempt = OptimizationAttempt(
                    iteration=iteration,
                    code=optimized_code,
                    compilation_success=False,
                    compilation_error=error,
                    correctness_passed=False,
                    execution_time_ms=0.0,
                    speedup=0.0,
                    occupancy=0.0,
                    memory_efficiency=0.0,
                    reasoning=reasoning,
                )
                self.llm_optimizer.add_attempt(attempt)
                #update metrics for error
                current_metrics = {
                    **current_metrics,
                    'compilation_error': error,
                }
                continue
            metrics = self.profile_kernel(ptx_code, inputs)
            exec_time = metrics.get('execution_time_ms', 0.0)
            if exec_time == 0.0:
                exec_time = self.baseline_time_ms 
            
            speedup = self.baseline_time_ms / exec_time
            # Validate correctness
            # TODO: Actually run kernel and validate
            correctness = True  
            
            attempt = OptimizationAttempt(
                iteration=iteration,
                code=optimized_code,
                compilation_success=True,
                compilation_error=None,
                correctness_passed=correctness,
                execution_time_ms=exec_time,
                speedup=speedup,
                occupancy=metrics.get('occupancy', 0.0),
                memory_efficiency=metrics.get('memory_efficiency', 0.0),
                reasoning=reasoning,
            )
            self.llm_optimizer.add_attempt(attempt)
            if correctness and speedup > best_speedup: #update if improve
                best_speedup = speedup
                best_code = optimized_code
                if best_speedup >= target_speedup:
                    break
            current_metrics = metrics
        
        return best_code, best_speedup
    
def example_usage():
    #vector add :example
    baseline = """
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}"""
    def generate_inputs():
        n=1000000
        return{
            'a': np.random.randn(n).astype(np.float32),
            'b': np.random.randn(n).astype(np.float32),
            'n': n,
        }
    def validate(output, reference):
        return np.allclose(output, reference, atol=1e-5)
    
    optimizer=GPUOptimizer(baseline_code=baseline, kernel_name="vector_add", model_provider="lmstudio", model_name="local-model")
    best_code, _ = optimizer.optimize(input_generator=generate_inputs, validator=validate, max_iterations=5, target_speedup=4.0)

    return best_code


if __name__ == "__main__":
    example_usage()
