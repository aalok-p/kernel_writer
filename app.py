import os
from dotenv import load_dotenv
from typing import Iterator, Tuple
from env_server import KernelOptimization_env, TASKS
from openai import OpenAI
from models import Action
import gradio as gr
import traceback

load_dotenv()

def task_baseline_code(task_id: str) -> str:
    return TASKS[task_id]["baseline_code"]

def extract_code(text: str) -> str:
    if "```" not in text:
        return text
    start = text.find("```")
    end = text.rfind("```")
    chunk = text[start + 3 : end]
    if chunk.startswith("cuda") or chunk.startswith("cpp"):
        return chunk.split("\n", 1)[1]
    return chunk

def ui(task_id:str, kernel_code: str, max_steps:int, openai_api_key:str)-> Iterator[Tuple[str,str]]:
    log= []
    env=KernelOptimization_env()
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield "ERROR: Missing OPENAI_API_KEY", ""
        return

    model = os.getenv("MODEL_NAME", "llama-3.2-3b")
    client = OpenAI(api_key=api_key, base_url=os.getenv("API_BASE_URL", "https://api.oxlo.ai/v1"))
    obs = env.reset(task_id=task_id)["observation"]
    if kernel_code and kernel_code.strip():
        custom_code = kernel_code.strip()
        env.state.best_code = custom_code
        obs["current_best_code"] = custom_code
    best_code = obs["current_best_code"]
    log.append(f"Task: {obs['task_name']}")

    for _ in range(max_steps):
        try:
            prompt = f"Optimize CUDA code:\n{obs['current_best_code']}\nPending checks: {obs['pending_checks']}\nReturn code only."
            res = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Return only optimized CUDA code."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (res.choices[0].message.content or "").strip()
            code = extract_code(raw).strip() or obs["current_best_code"]
            step = env.step(Action(optimized_code=code, strategy="ui_proposed"))
            obs = step.observation.model_dump()
            best_code = obs["current_best_code"]
            log.append(f"step={obs['step_count']} reward={step.reward.value:.3f} speedup={obs['current_best_speedup']:.3f}x")
            yield "\n".join(log), best_code
            if step.done:
                break
        except Exception as e:
            yield f"{chr(10).join(log)}\nERROR: {e}\n{traceback.format_exc()}", best_code
            return

with gr.Blocks(title="CUDA Kernel Optimizer") as demo:
    gr.Markdown("CUDA Kernel Optimizer - OpenEnv-aligned workflow")
    task = gr.Dropdown(choices=list(TASKS.keys()), value="vector_add_easy", label="Task")
    kernel_input = gr.Code(
        label="Kernel Code (editable, used as optimization input)",
        language="cpp",
        lines=16,
        value=TASKS["vector_add_easy"]["baseline_code"],
    )
    steps = gr.Slider(minimum=1, maximum=12, value=6, step=1, label="Max Steps")
    key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...")
    run = gr.Button("Run Optimization", variant="primary")
    logs = gr.Textbox(label="Logs", lines=14)
    code = gr.Code(label="Best Code", language="cpp", lines=16)
    task.change(task_baseline_code, inputs=[task], outputs=[kernel_input])
    run.click(ui, inputs=[task, kernel_input, steps, key], outputs=[logs, code])


if __name__ == "__main__":
    demo.launch(
        show_error=True,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
