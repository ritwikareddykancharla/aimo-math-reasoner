"""
Local sandbox server that matches SandboxFusion API format exactly.
VERL's call_sandbox_api sends POST requests expecting this response format.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import subprocess
import tempfile
import os
import uvicorn

app = FastAPI()

class SandboxRequest(BaseModel):
    code: str
    stdin: Optional[str] = None
    compile_timeout: int = 30
    run_timeout: int = 30
    memory_limit_MB: int = 1024
    language: str = "python"
    files: dict = {}
    fetch_files: list = []

@app.post("/execute")
def execute_code(req: SandboxRequest):
    if req.language != "python":
        return {
            "status": "Failed",
            "compile_result": None,
            "run_result": {
                "status": "Error",
                "stdout": "",
                "stderr": f"Unsupported language: {req.language}",
                "return_code": 1,
                "execution_time": 0,
            }
        }

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
            timeout=req.run_timeout,
            input=req.stdin,
            env={
                "PATH": "/usr/bin:/usr/local/bin:/home/ssm-user/.local/bin",
                "HOME": "/tmp",
                "PYTHONPATH": "/home/ssm-user/.local/lib/python3.12/site-packages",
            }
        )

        return {
            "status": "Success",
            "compile_result": {
                "status": "Finished",
                "stderr": "",
                "return_code": 0,
                "execution_time": 0,
            },
            "run_result": {
                "status": "Finished",
                "stdout": result.stdout,
                "stderr": result.stderr[:500] if result.stderr else "",
                "return_code": result.returncode,
                "execution_time": req.run_timeout,
            }
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "Failed",
            "compile_result": None,
            "run_result": {
                "status": "TimeLimitExceeded",
                "stdout": "",
                "stderr": "Execution timed out",
                "return_code": 1,
                "execution_time": req.run_timeout,
            }
        }
    except Exception as e:
        return {
            "status": "Failed",
            "compile_result": None,
            "run_result": {
                "status": "Error",
                "stdout": "",
                "stderr": str(e),
                "return_code": 1,
                "execution_time": 0,
            }
        }
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="warning")
