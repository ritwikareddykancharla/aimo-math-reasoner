# scripts/code_executor_server.py
"""
Simple FastAPI code executor for SGLang tool use.
Run this on BOTH nodes before launching training.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import tempfile
import os
import uvicorn

app = FastAPI()

class CodeRequest(BaseModel):
    code: str
    timeout: int = 30

@app.post("/execute")
def execute_code(req: CodeRequest):
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as f:
        f.write(req.code)
        fname = f.name

    try:
        result = subprocess.run(
            ["python3.12", fname],
            capture_output=True,
            text=True,
            timeout=req.timeout,
            # Sandboxed — no network, limited memory
            env={
                "PATH": "/usr/bin:/usr/local/bin",
                "PYTHONPATH": "",
            }
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr[:500]}"
        return {"output": output, "error": None}

    except subprocess.TimeoutExpired:
        return {"output": "", "error": "Execution timed out"}
    except Exception as e:
        return {"output": "", "error": str(e)}
    finally:
        os.unlink(fname)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
