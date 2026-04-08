import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple
import openai
import re

load_dotenv()

@dataclass
class OptimizationAttempt:
    iteration:int
    code: str
    compilation_success: bool
    compilation_error: Optional[str]
    correctness_passed: bool
    execution_time_ms: float
    speedup: float
    occupancy: float
    memory_efficiency: float
    reasoning: Optional[str] = None

class LLMOptimizer:
    def __init__(self, model_provider:Optional[str]=None, model_name:Optional[str]=None, api_key:Optional[str]=None):
        self.model_provder = self.model_provder or os.getenv("Model_Proivder", "lmstudio")

        if model_name is None:
            if self.model_provder == "lmstudio":
                self.model_name = os.getenv("LM_STUDIO_MODEL", "Qwen2.5 Coder 14B")
            elif self.model_provider =="openai":
                self.model_name = os.getenv("OPENAI_MODEL", "gpt-4")
            # elif self.model_provider == "anthropic":
            #     self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-opus")
            else:
                self.model_name = "local-model"
        else:
            self.model_name = model_name
        
        if api_key is None:
            if self.model_provider =="openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            # elif self.model_provider == "anthropic":
            #     self.api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                self.api_key = None 
        else:
            self.api_key = api_key
        self.history: List[OptimizationAttempt] = []
    

    def init_llm(self):
        if self.model_provder =="lmstudio":
            base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/")
            self.client = openai.OpenAI(base_url=base_url, api_key="not-needed")
            print(f"using lm studio at {base_url}")
        
        elif self.model_provder == "openai":
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client=openai.OpenAI(api_key=self.api_key)
            print(f"using openai {self.model_name}")

        else:
            raise ValueError(f"{self.model_provder}")      


    def build_prompt(self)->str:
        return """You are an expert CUDA kernel optimization engineer.

Your goal: Iteratively improve GPU kernel performance through architectural optimizations.

Key optimization techniques:
1. **Shared Memory Tiling**: Load data into on-chip memory for reuse (100× faster than global memory)
2. **Memory Coalescing**: Ensure consecutive threads access consecutive memory addresses
3. **Warp Efficiency**: Avoid divergent branches within warps (32 threads)
4. **Occupancy**: Maximize GPU core utilization through proper block/thread configuration
5. **Vectorization**: Use float2/float4 for wider memory transactions
6. **Bank Conflicts**: Pad shared memory to avoid bank conflicts
7. **Register Blocking**: Keep intermediate results in registers when possible

CRITICAL RULES:
- OUTPUT ONLY CUDA CODE - NO EXPLANATIONS, NO REASONING TEXT OUTSIDE CODE
- Keep thinking to 1-2 brief comment lines at the top ONLY
- Always maintain correctness (validate against reference implementation)
- Consider GPU architecture limits (shared memory, registers, threads per block)

MANDATORY FORMAT - Return ONLY this, nothing else:
```cuda
// Brief strategy: [one line max]
[CUDA kernel code here]
```

DO NOT write explanations, analysis, or reasoning outside the code block. ONLY code with minimal inline comments."""
    
    def optimize_prompt(self, baseline_code:str, current_metrics:Optional[Dict]=None)->str:
        history_text=" "
        if self.history:
            history_text = "\n\n Optimization History:\n"
            for attempt in self.history[-5:]:
                history_text += f"\n Attempt {attempt.iteration}: {status}\n"
                if attempt.compilation_success:
                    history_text +=f"speedup {attempt.iteration}×\n"
                    history_text += f"  Occupancy: {attempt.occupancy:.1f}%\n"
                    history_text += f"  Memory Efficiency: {attempt.memory_efficiency:.1f}%\n"
                else:
                    history_text += f"  Compilation Error: {attempt.compilation_error}\n"
        
        # build current metrics
        metrics_text = ""
        if current_metrics:
            metrics_text = f"""Current Performance Metrics:
- Execution time: {current_metrics.get('execution_time_ms', 0):.3f}ms
- Occupancy: {current_metrics.get('occupancy', 0):.1f}% (target: >75%)
- Global memory efficiency: {current_metrics.get('memory_efficiency', 0):.1f}%
- Warp divergence: {current_metrics.get('warp_divergence', 0):.1f}%
- Shared memory used: {current_metrics.get('shared_memory_bytes', 0)} bytes
- Registers per thread: {current_metrics.get('registers_per_thread', 0)}"""
        
        #get best speedup so far
        best_speedup = max([a.speedup for a in self.history], default=1.0)
        
        prompt = f"""Baseline CUDA Kernel:```cuda {baseline_code}``` {history_text} {metrics_text}

Current best speedup: {best_speedup:.2f}×

Your task: Propose the next optimization to improve performance further.

IMPORTANT: Output ONLY the optimized code with 1 brief comment line. NO explanations outside code.
Follow this exact format:
```cuda
// Strategy: [one sentence max]
[optimized kernel code]
```
"""
        return prompt
    
    def call_llm(self, system_prompt:str, user_prompt:str)->str:
        if self.model_provder =="lmstudio":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7, #creative but not much
                max_tokens=2000,
            )
            return response.choices[0].message.content
        
        elif self.model_provider == "openai":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
            return response.content[0].text

        else:
            raise NotImplementedError()
        
    def extract_cuda_code(self, llm_response:str)->Tuple[str,Optional[str]]:
        cuda_pattern =  r"```(?:cuda|c\+\+|cpp)?\n(.*?)```"
        matches = re.findall(cuda_pattern, llm_response, re.DOTALL)

        if matches:
            code =matches[0].strip()
            reasoning_pattern = r"//\s*Optimization strategy:\s*(.+?)(?=\n//|\n\n|$)"
            reasoning_match = re.search(reasoning_pattern, code, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else None

            return code, reasoning
        