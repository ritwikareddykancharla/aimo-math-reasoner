import os
import sys
import subprocess

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

def setup_environment():
    # Only run this on Kaggle
    if not os.path.exists('/kaggle'):
        return

    print("Setting up environment...")
    
    # Uninstall conflicting packages
    packages_to_uninstall = ['keras', 'matplotlib', 'scikit-learn', 'tensorflow']
    for pkg in packages_to_uninstall:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', pkg], check=False)
    
    # Install wheels
    input_archive = '/kaggle/input/aimo-3-utils/wheels.tar.gz'
    temp_dir = '/kaggle/tmp/setup'
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        subprocess.run(['tar', '-xzf', input_archive, '-C', temp_dir], check=True)
    
    subprocess.run([
        sys.executable, 
        '-m', 
        'pip', 
        'install', 
        '--no-index', 
        '--find-links', 
        f'{temp_dir}/wheels', 
        'unsloth', 
        'trl', 
        'vllm', 
        'openai_harmony'
    ], check=True)
    
    # Set env vars
    os.environ['TRANSFORMERS_NO_TF'] = '1'
    os.environ['TRANSFORMERS_NO_FLAX'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
    os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/tmp/setup/tiktoken_encodings'

setup_environment()

import gc
import re
import math
import time
import queue
import threading
import contextlib
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
import polars as pl

from openai import OpenAI

# We assume these packages are installed in the environment or via the Kaggle dataset
try:
    from openai_harmony import (
        HarmonyEncodingName, 
        load_harmony_encoding, 
        SystemContent, 
        ReasoningEffort, 
        ToolNamespaceConfig, 
        Author, 
        Message, 
        Role, 
        TextContent, 
        Conversation
    )
except ImportError:
    # Fallback/Mock for local testing if openai_harmony isn't available
    pass

from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server

# ============================================================
# CONFIGURATION
# ============================================================

class CFG:
    # We reduce attempts to 2 because each attempt is now much deeper (Solve -> Critique -> Refine)
    attempts = 2
    
    # System prompt emphasizes the new workflow
    system_prompt = (
        'You are an elite mathematical problem solver. Your goal is to find the correct answer '
        'through rigorous reasoning and self-verification.\n\n'
        'You have access to a Python environment. Use it to verify every step.\n'
        'If you find an error in your logic or code, acknowledge it and correct it immediately.\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Output the final answer in \\boxed{}.'
    )
    
    tool_prompt = (
        'Use this tool to execute Python code. '
        'Always use print() to display results.'
    )
    
    # New prompts for the Critic workflow
    critic_prompt = (
        "Review your solution above. "
        "1. Did you verify the answer with Python code? "
        "2. Did you consider edge cases (e.g., n=0, n=1)? "
        "3. Is the logic sound? "
        "If the solution is correct, verify it again with a different method if possible. "
        "If there are errors, explain them clearly."
    )
    
    refine_prompt = (
        "Based on your review, fix any errors and provide the final correct answer. "
        "Ensure the answer is an integer between 0 and 99999. "
        "Put the final answer inside \\boxed{}."
    )
    
    # Model settings
    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'
    gpu_memory_utilization = 0.96
    
    # Time limits
    high_problem_timeout = 900
    base_problem_timeout = 300
    notebook_limit = 17400
    server_timeout = 180
    
    # Sandbox
    session_timeout = 960
    jupyter_timeout = 10  # Increased slightly for safety
    sandbox_timeout = 5
    
    # Inference
    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    
    workers = 16
    turns = 64  # Turns per phase
    seed = 42
    
    temperature = 0.7 # Lower temperature for more focused reasoning
    min_p = 0.02

# ============================================================
# TEMPLATE & SANDBOX
# ============================================================

class AIMO3Template:
    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(self, system_prompt: str, user_prompt: str, tool_config: ToolNamespaceConfig) -> list[Message]:
        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]

class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        
        ports = self._get_next_ports(5)
        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import numpy as np\n'
            'import sympy as sp\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                if content.get('name') == 'stdout':
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])
                stderr_parts.append(self._format_error(traceback_list))
            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')
                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')
            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)
        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr
        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy as np\n'
            'import sympy as sp\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):
        self.close()

class AIMO3Tool:
    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().split('\n')
        if not lines: return code
        last_line = lines[-1].strip()
        if 'print' in last_line or 'import' in last_line: return code
        if not last_line or last_line.startswith('#'): return code
        lines[-1] = 'print(' + last_line + ')'
        return '\n'.join(lines)

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(name='python', description=self._tool_prompt, tools=[])

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f'[ERROR] {exc}'
        return [self._make_response(output, channel=message.channel)]

# ============================================================
# SOLVER WITH CRITIC LOOP
# ============================================================

class AIMO3Solver:
    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
    
        self._preload_model_weights()
        self.server_process = self._start_server()
    
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)
        self._wait_for_server()
        self._initialize_kernels()
    
        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _preload_model_weights(self) -> None:
        print(f'Loading model weights from {self.cfg.model_path}...')
        start_time = time.time()
        files_to_load = []
        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
        
        def _read_file(path: str) -> None:
            with open(path, 'rb') as f:
                while f.read(1024 * 1024 * 1024): pass
        
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))
        print(f'Preloaded in {time.time() - start_time:.2f}s.')

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--seed', str(self.cfg.seed),
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--tensor-parallel-size', '1',
            '--max-num-seqs', str(self.cfg.batch_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--host', '0.0.0.0', '--port', str(self.port),
            '--dtype', self.cfg.dtype, '--kv-cache-dtype', self.cfg.kv_cache_dtype,
            '--max-model-len', str(self.cfg.context_tokens),
            '--stream-interval', str(self.cfg.stream_interval),
            '--async-scheduling', '--disable-log-stats', '--enable-prefix-caching'
        ]
        self.log_file = open('vllm_server.log', 'w')
        return subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)

    def _wait_for_server(self):
        print('Waiting for vLLM server...')
        start_time = time.time()
        for _ in range(self.cfg.server_timeout):
            if self.server_process.poll() is not None:
                raise RuntimeError('Server died.')
            try:
                self.client.models.list()
                print(f'Server ready ({time.time() - start_time:.2f}s).')
                return
            except:
                time.sleep(1)
        raise RuntimeError('Server timeout.')

    def _initialize_kernels(self) -> None:
        print(f'Initializing {self.cfg.workers} kernels...')
        self.sandbox_pool = queue.Queue()
        def _create_sandbox(): return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

    def _scan_for_answer(self, text: str) -> int | None:
        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)
        if matches:
            try:
                val = int(matches[-1].replace(',', ''))
                if 0 <= val <= 99999: return val
            except ValueError: pass
        return None

    def _run_conversation_step(
        self, 
        conversation: Conversation, 
        local_tool: AIMO3Tool, 
        attempt_seed: int, 
        stop_event: threading.Event, 
        deadline: float,
        stats: dict
    ) -> int | None:
        """Runs the generation loop for the current state of conversation."""
        final_answer = None
        
        for _ in range(self.cfg.turns):
            if stop_event.is_set() or time.time() > deadline:
                break

            prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            max_tokens = self.cfg.context_tokens - len(prompt_ids)
            if max_tokens < self.cfg.buffer_tokens:
                break

            stream = self.client.completions.create(
                model=self.cfg.served_model_name,
                temperature=self.cfg.temperature,
                logprobs=self.cfg.top_logprobs,
                max_tokens=max_tokens,
                prompt=prompt_ids,
                seed=attempt_seed,
                stream=True,
                extra_body={
                    'min_p': self.cfg.min_p,
                    'stop_token_ids': self.stop_token_ids,
                    'return_token_ids': True
                }
            )

            try:
                token_buffer = []
                text_chunks = []
                
                for chunk in stream:
                    if stop_event.is_set() or time.time() > deadline:
                        break
                    
                    new_tokens = chunk.choices[0].token_ids
                    new_text = chunk.choices[0].text
                    
                    if new_tokens:
                        token_buffer.extend(new_tokens)
                        stats['total_tokens'] += len(new_tokens)
                        text_chunks.append(new_text)
                        
                        if chunk.choices[0].logprobs and chunk.choices[0].logprobs.top_logprobs:
                            stats['logprobs_buffer'].extend(chunk.choices[0].logprobs.top_logprobs)

                    if '}' in new_text:
                        search_text = ''.join(text_chunks[-self.cfg.search_tokens:])
                        ans = self._scan_for_answer(search_text)
                        if ans is not None:
                            final_answer = ans
                            # Don't break yet, let it finish the thought/tool call if needed? 
                            # Actually, for speed, we usually break if we found a boxed answer in the final phase.
                            # But in intermediate phases, we might want to continue.
                            # For now, let's just break the stream, but not the loop if we need to run tools.
                            break 
                            
            finally:
                stream.close()

            if not token_buffer:
                break

            new_messages = self.encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
            conversation.messages.extend(new_messages)
            last_message = new_messages[-1]

            if last_message.channel == 'final':
                # Model signaled it's done
                ans_text = last_message.content[0].text
                final_answer = self._scan_for_answer(ans_text)
                break

            if last_message.recipient == 'python':
                stats['python_calls'] += 1
                tool_responses = local_tool.process_sync_plus(last_message)
                resp_text = tool_responses[0].content[0].text
                if resp_text.startswith('[ERROR]') or 'Traceback' in resp_text:
                    stats['python_errors'] += 1
                conversation.messages.extend(tool_responses)
            else:
                # Text output (thought)
                if final_answer is not None:
                    # If we found an answer in text and it wasn't a tool call, we are likely done with this phase
                    break
        
        return final_answer

    def _process_attempt(self, problem: str, system_prompt: str, attempt_index: int, stop_event: threading.Event, deadline: float) -> dict:
        sandbox = None
        stats = {
            'total_tokens': 0, 'python_calls': 0, 'python_errors': 0, 'logprobs_buffer': []
        }
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))
        final_answer = None

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox
            )

            # --- PHASE 1: INITIAL SOLVE ---
            messages = self.template.apply_chat_template(system_prompt, problem, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)
            
            # Run solve loop
            ans_1 = self._run_conversation_step(conversation, local_tool, attempt_seed, stop_event, deadline, stats)
            
            # --- PHASE 2: CRITIQUE ---
            # Append critic prompt
            conversation.messages.append(Message.from_role_and_content(Role.USER, self.cfg.critic_prompt))
            
            # Run critique loop
            self._run_conversation_step(conversation, local_tool, attempt_seed, stop_event, deadline, stats)
            
            # --- PHASE 3: REFINE ---
            # Append refine prompt
            conversation.messages.append(Message.from_role_and_content(Role.USER, self.cfg.refine_prompt))
            
            # Run refine loop
            ans_final = self._run_conversation_step(conversation, local_tool, attempt_seed, stop_event, deadline, stats)
            
            if ans_final is not None:
                final_answer = ans_final
            elif ans_1 is not None:
                final_answer = ans_1 # Fallback to initial answer

        except Exception as e:
            print(f"Attempt {attempt_index} failed: {e}")
            stats['python_errors'] += 1
        finally:
            if sandbox:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        # Compute entropy
        entropy = float('inf')
        if stats['logprobs_buffer']:
            total_ent = 0.0
            count = 0
            for top_logprobs in stats['logprobs_buffer']:
                if top_logprobs:
                    curr_ent = 0.0
                    for p_log in top_logprobs.values():
                        p = math.exp(p_log)
                        if p > 0: curr_ent -= p * math.log2(p)
                    total_ent += curr_ent
                    count += 1
            if count > 0: entropy = total_ent / count

        return {
            'Attempt': attempt_index + 1,
            'Answer': final_answer,
            'Response Length': stats['total_tokens'],
            'Python Calls': stats['python_calls'],
            'Python Errors': stats['python_errors'],
            'Entropy': entropy
        }

    def _select_answer(self, results: list) -> int:
        votes = defaultdict(float)
        counts = defaultdict(int)
        for r in results:
            ans = r['Answer']
            if ans is not None:
                weight = 1.0 / (r['Entropy'] + 1e-9)
                votes[ans] += weight
                counts[ans] += 1
        
        if not votes: return 0
        best_ans = max(votes.items(), key=lambda x: x[1])[0]
        
        print("\nVote Summary:")
        for ans, w in votes.items():
            print(f"Answer {ans}: Vote Score {w:.3f} ({counts[ans]} votes)")
        print(f"Selected: {best_ans}")
        return best_ans

    def solve_problem(self, problem: str) -> int:
        print(f'\nProblem: {problem[:100]}...')
        elapsed = time.time() - self.notebook_start_time
        budget = max(self.cfg.base_problem_timeout, min(self.cfg.high_problem_timeout, self.cfg.notebook_limit - elapsed - (self.problems_remaining * self.cfg.base_problem_timeout)))
        deadline = time.time() + budget
        print(f'Budget: {budget:.1f}s')

        stop_event = threading.Event()
        with ThreadPoolExecutor(max_workers=self.cfg.attempts) as executor:
            futures = [
                executor.submit(self._process_attempt, problem, self.cfg.system_prompt, i, stop_event, deadline)
                for i in range(self.cfg.attempts)
            ]
            results = []
            for f in as_completed(futures):
                try: results.append(f.result())
                except Exception as e: print(f"Future error: {e}")
        
        self.problems_remaining = max(0, self.problems_remaining - 1)
        return self._select_answer(results)

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()

# ============================================================
# MAIN
# ============================================================

solver = None

def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    global solver
    if solver is None:
        solver = AIMO3Solver(CFG)
    
    id_val = id_.item(0)
    q_text = question.item(0)
    
    try:
        ans = solver.solve_problem(q_text)
    except Exception as e:
        print(f"Solve failed: {e}")
        ans = 0
        
    return pl.DataFrame({'id': id_val, 'answer': ans})

if __name__ == "__main__":
    # Test locally
    if not os.path.exists('/kaggle'):
        print("Running local test...")
        # Create a dummy config if needed or just run
        solver = AIMO3Solver(CFG)
        print(solver.solve_problem("What is 10 + 10?"))
    else:
        inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            inference_server.serve()
        else:
            inference_server.run_local_gateway(('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',))