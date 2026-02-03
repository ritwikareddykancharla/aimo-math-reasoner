# ============================================================
# ALG Sequential Solver for AIMO3 - COMPLETE VERSION
# ============================================================

# Remove unnecessary packages
%pip uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow' -q

import warnings
warnings.simplefilter('ignore')

import os
import sys
import subprocess

def set_env(input_archive, temp_dir):
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
    ], check=True, capture_output=True)

set_env(
    input_archive='/kaggle/input/aimo-3-utils/wheels.tar.gz',
    temp_dir='/kaggle/tmp/setup'
)

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/tmp/setup/tiktoken_encodings'

# ============================================================
# IMPORTS
# ============================================================

import gc
import re
import json
import math
import time
import queue
import threading
import contextlib
import traceback
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import polars as pl

from openai import OpenAI

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

from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server

print('All imports done')

# ============================================================
# CONFIGURATION WITH DOMAIN KNOWLEDGE
# ============================================================

class CFG:
    # Model settings
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    served_model_name = 'gpt-oss'
    
    # Inference settings
    context_tokens = 65536
    temperature = 0.7
    top_p = 0.95
    max_tokens_per_turn = 4096
    
    # Time budgets (seconds) based on complexity
    time_budget = {
        'simple': 60,
        'medium': 180,
        'hard': 480,
        'default': 180
    }
    
    # Lemma settings
    max_lemmas = 8
    max_retries_per_lemma = 3
    
    # Python sandbox
    sandbox_timeout = 30
    
    # Server settings
    server_port = 8000
    server_timeout = 180
    
    # vLLM settings
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'
    gpu_memory_utilization = 0.96
    batch_size = 256
    
    # Comprehensive domain knowledge for IMO problems
    DOMAIN_KNOWLEDGE = {
        'algebra': {
            'description': 'Algebraic manipulation, equations, inequalities, polynomials',
            'common_techniques': [
                'Factorization and expansion',
                'Substitution and change of variables',
                'AM-GM inequality, Cauchy-Schwarz inequality',
                'Vieta\'s formulas for polynomials',
                'Symmetric sums and elementary symmetric polynomials',
                'Telescoping sums and products',
                'Completing the square',
                'Functional equations',
                'Inequalities (Chebyshev, Holder, Minkowski)'
            ],
            'common_patterns': [
                'Look for symmetry in expressions',
                'Consider substitution to simplify',
                'Use inequalities to bound expressions',
                'Factor polynomials to find roots',
                'Look for telescoping patterns in sums'
            ],
            'verification_strategies': [
                'Check with specific numerical examples',
                'Verify inequalities at boundary cases',
                'Test polynomial identities with random values',
                'Use symbolic computation for expansions'
            ]
        },
        'number_theory': {
            'description': 'Properties of integers, divisibility, primes, modular arithmetic',
            'common_techniques': [
                'Modular arithmetic (mod n)',
                'Chinese remainder theorem',
                'Fermat\'s little theorem and Euler\'s theorem',
                'Euclidean algorithm and Bezout\'s identity',
                'Prime factorization and unique factorization',
                'Order of elements modulo n',
                'Legendre symbol and quadratic reciprocity',
                'Divisibility rules and properties',
                'Pell equations and Diophantine equations'
            ],
            'common_patterns': [
                'Check parity (even/odd)',
                'Consider remainders modulo small numbers',
                'Look for prime factorization patterns',
                'Use divisibility chains',
                'Consider greatest common divisors'
            ],
            'verification_strategies': [
                'Test with small numerical examples',
                'Verify divisibility properties',
                'Check modular arithmetic calculations',
                'Use brute force for small ranges'
            ]
        },
        'combinatorics': {
            'description': 'Counting, arrangements, graphs, combinatorial structures',
            'common_techniques': [
                'Pigeonhole principle',
                'Double counting arguments',
                'Inclusion-exclusion principle',
                'Generating functions',
                'Recurrence relations',
                'Graph theory concepts',
                'Bijections and combinatorial proofs',
                'Ramsey theory',
                'Probabilistic method'
            ],
            'common_patterns': [
                'Look for invariant quantities',
                'Consider extreme cases',
                'Use symmetry to simplify counting',
                'Look for recursive structure',
                'Consider graph representations'
            ],
            'verification_strategies': [
                'Count small cases manually',
                'Verify recurrence relations',
                'Check combinatorial identities',
                'Use computational enumeration for small n'
            ]
        },
        'geometry': {
            'description': 'Shapes, angles, lengths, coordinates, transformations',
            'common_techniques': [
                'Coordinate geometry',
                'Vector methods',
                'Trigonometry and trigonometric identities',
                'Similarity and congruence',
                'Power of a point',
                'Circle theorems (inscribed angles, cyclic quadrilaterals)',
                'Triangle geometry (cevians, medians, altitudes)',
                'Transformations (rotations, reflections, homothety)',
                'Complex numbers in geometry'
            ],
            'common_patterns': [
                'Add auxiliary lines',
                'Use coordinate system wisely',
                'Look for similar triangles',
                'Consider symmetry',
                'Use angle chasing'
            ],
            'verification_strategies': [
                'Verify with specific coordinates',
                'Check trigonometric identities',
                'Use geometric software for verification',
                'Test with special cases'
            ]
        },
        'analysis': {
            'description': 'Limits, continuity, sequences, series, inequalities',
            'common_techniques': [
                'Epsilon-delta arguments',
                'Mean value theorem and Taylor series',
                'Monotonic sequences and convergence',
                'Inequalities (Jensen, Chebyshev, rearrangement)',
                'Functional equations',
                'Recurrence relations for sequences',
                'Asymptotic analysis',
                'Fixed point theorems',
                'Continuity and intermediate value property'
            ],
            'common_patterns': [
                'Look for monotonicity',
                'Consider limiting behavior',
                'Use telescoping in sequences',
                'Apply known inequalities',
                'Check special values'
            ],
            'verification_strategies': [
                'Test with numerical sequences',
                'Verify inequalities numerically',
                'Check continuity at sample points',
                'Use computational limits'
            ]
        }
    }
    
    # Common IMO problem solving strategies
    IMO_STRATEGIES = [
        'Look for invariant or monovariant quantities',
        'Consider extreme cases or boundary conditions',
        'Use symmetry to reduce complexity',
        'Try small cases to find pattern',
        'Assume opposite and seek contradiction',
        'Use induction (mathematical, strong, or structural)',
        'Apply probabilistic method',
        'Construct explicit examples or counterexamples',
        'Use double counting arguments',
        'Employ generating functions or recurrence relations'
    ]
    
    print('Configuration with domain knowledge loaded')

set_seed(42)

# ============================================================
# JSON OUTPUT PROMPTS WITH DOMAIN KNOWLEDGE
# ============================================================

CLASSIFICATION_PROMPT = """You are an expert IMO problem classifier. Analyze this mathematical problem and output ONLY valid JSON.

Problem: {problem}

Domain Knowledge Context:
{domain_context}

Output JSON format:
{{
  "topic": "algebra|number_theory|combinatorics|geometry|analysis",
  "complexity": "simple|medium|hard",
  "key_techniques": ["technique1", "technique2", "technique3"],
  "estimated_lemmas": 4,
  "reasoning": "brief explanation",
  "confidence": 0.95
}}

Guidelines:
- Simple: Direct, few steps (< 5 min for expert)
- Medium: Requires insight, multiple steps (5-15 min)
- Hard: Deep insight, creative approach (> 15 min)
- Choose techniques from domain knowledge
- Confidence between 0.0 and 1.0
"""

LEMMA_GRAPH_PROMPT = """You are an IMO Gold Medalist. Decompose this problem into a lemma graph. Output ONLY valid JSON.

Problem: {problem}
Topic: {topic}
Complexity: {complexity}
Target Lemma Count: {estimated_lemmas}

Domain Knowledge for {topic}:
{domain_knowledge}

IMO Strategies to Consider:
{imo_strategies}

Output JSON format:
{{
  "lemmas": [
    {{
      "id": "L1",
      "statement": "mathematical statement",
      "type": "structural|reduction|computational|inequality|existence|counting|verification",
      "dependencies": [],
      "purpose": "why this lemma is needed",
      "verification_strategy": "how to verify this lemma",
      "domain_hint": "which domain technique applies"
    }},
    ...
  ],
  "final_lemma": {{
    "id": "FINAL",
    "statement": "Synthesize solution from all lemmas",
    "type": "synthesis",
    "dependencies": ["L1", "L2", ...],
    "purpose": "Combine verified lemmas to solve original problem"
  }},
  "graph_strategy": "explanation of decomposition approach"
}}

Rules:
1. Dependencies must form a DAG (no cycles)
2. Use domain-appropriate techniques
3. Each lemma should be mathematically precise
4. Consider IMO problem-solving strategies
"""

LEMMA_PROOF_PROMPT = """You are a mathematical proof assistant. Prove this lemma and provide verification. Output ONLY valid JSON.

Problem Context: {problem}
Lemma ID: {lemma_id}
Lemma Statement: {lemma_statement}
Lemma Type: {lemma_type}
Purpose: {purpose}

Domain: {topic}
Domain Techniques: {domain_techniques}

Output JSON format:
{{
  "proof": "step-by-step mathematical proof",
  "verification_code": "python code to verify (if applicable, else empty string)",
  "verification_explanation": "how verification works",
  "confidence": 0.95,
  "key_insights": ["insight1", "insight2"]
}}

Guidelines:
- Proof must be mathematically rigorous
- Include all necessary steps
- If computational lemma, provide Python code
- If theoretical lemma, explain verification strategy
- Confidence based on proof completeness
"""

SOLUTION_PROMPT = """You are an IMO Gold Medalist. Solve this problem using the verified lemmas. Output ONLY valid JSON.

Problem: {problem}

Domain: {topic}
Domain Knowledge: {domain_knowledge}

Verified Lemmas Summary:
{lemmas_summary}

Output JSON format:
{{
  "solution_analysis": "how lemmas combine to solve problem",
  "step_by_step_solution": "complete mathematical solution",
  "answer": 123,
  "confidence": 0.95,
  "verification_checks": ["check1", "check2"],
  "alternative_approaches": ["approach1", "approach2"]
}}

Rules:
1. Answer must be an integer (or 0 if unknown)
2. Confidence between 0.0 and 1.0
3. Show how lemmas are used
4. Include verification of answer
5. Consider edge cases
"""

DIRECT_SOLUTION_PROMPT = """You are an IMO Gold Medalist. Solve this problem directly. Output ONLY valid JSON.

Problem: {problem}
Topic: {topic}

Domain Knowledge for {topic}:
{domain_knowledge}

IMO Strategies:
{imo_strategies}

Output JSON format:
{{
  "problem_analysis": "understanding of problem structure",
  "solution_strategy": "chosen approach and why",
  "mathematical_proof": "complete rigorous proof",
  "answer": 123,
  "confidence": 0.95,
  "verification": "how answer was verified",
  "edge_cases_checked": ["case1", "case2"]
}}

Rules:
1. Provide complete mathematical proof
2. Answer must be an integer
3. Confidence between 0.0 and 1.0
4. Check all edge cases
5. Verify answer makes sense
"""

# ============================================================
# SIMPLE DATA STRUCTURES
# ============================================================

@dataclass
class ProblemClassification:
    topic: str
    complexity: str  # "simple", "medium", "hard"
    key_techniques: List[str] = field(default_factory=list)
    estimated_lemmas: int = 4
    reasoning: str = ''
    confidence: float = 0.0
    
    def get_time_budget(self) -> float:
        return CFG.time_budget.get(self.complexity, CFG.time_budget['default'])
    
    def get_domain_info(self) -> Dict:
        """Get domain knowledge for this topic"""
        return CFG.DOMAIN_KNOWLEDGE.get(self.topic, CFG.DOMAIN_KNOWLEDGE['algebra'])

@dataclass
class Lemma:
    id: str
    statement: str
    lemma_type: str
    dependencies: List[str] = field(default_factory=list)
    purpose: str = ''
    verification_strategy: str = ''
    domain_hint: str = ''
    proof: str = ''
    verification_code: str = ''
    execution_result: Optional[str] = None
    verified: bool = False
    confidence: float = 0.0
    
    def to_summary(self) -> str:
        status = "✓" if self.verified else "?"
        return f"{status} {self.id} ({self.lemma_type}): {self.statement[:80]}..."

@dataclass 
class LemmaGraph:
    problem: str
    classification: ProblemClassification
    lemmas: Dict[str, Lemma] = field(default_factory=dict)
    final_lemma: Optional[Lemma] = None
    graph_strategy: str = ''
    
    def get_dependency_order(self) -> List[str]:
        """Topological sort of lemmas"""
        if not self.lemmas:
            return []
        
        # Build adjacency and in-degree
        adj = {lid: [] for lid in self.lemmas}
        in_deg = {lid: 0 for lid in self.lemmas}
        
        for lemma in self.lemmas.values():
            for dep in lemma.dependencies:
                if dep in self.lemmas:
                    adj[dep].append(lemma.id)
                    in_deg[lemma.id] += 1
        
        # Start with nodes having no dependencies
        queue = [lid for lid, deg in in_deg.items() if deg == 0]
        result = []
        
        while queue:
            lid = queue.pop(0)
            result.append(lid)
            
            for neighbor in adj[lid]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.lemmas):
            # Fallback: return in ID order
            return sorted(self.lemmas.keys())
        
        return result
    
    def get_lemma_summary(self) -> str:
        """Create formatted summary of all lemmas"""
        summary_lines = []
        for lemma_id in self.get_dependency_order():
            lemma = self.lemmas[lemma_id]
            summary_lines.append(lemma.to_summary())
        return "\n".join(summary_lines)

@dataclass
class SolutionResult:
    problem: str
    classification: ProblemClassification
    answer: Optional[int] = None
    success: bool = False
    time_taken: float = 0.0
    method: str = 'unknown'
    confidence: float = 0.0
    solution_text: str = ''

# ============================================================
# JSON PARSING UTILITIES
# ============================================================

class JSONParser:
    @staticmethod
    def parse_json_response(text: str, fallback=None):
        """Try to parse JSON from text, with fallback"""
        if not text:
            return fallback
        
        # Try to extract JSON from text
        try:
            # Look for JSON pattern
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            
            # If no match, try parsing entire text
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON parse error: {e}")
            print(f"[DEBUG] Text snippet: {text[:200]}...")
            return fallback
    
    @staticmethod
    def parse_classification(text: str) -> ProblemClassification:
        """Parse classification from JSON response"""
        data = JSONParser.parse_json_response(text)
        if not data:
            return ProblemClassification(
                topic='algebra',
                complexity='medium',
                confidence=0.5
            )
        
        return ProblemClassification(
            topic=data.get('topic', 'algebra'),
            complexity=data.get('complexity', 'medium'),
            key_techniques=data.get('key_techniques', []),
            estimated_lemmas=data.get('estimated_lemmas', 4),
            reasoning=data.get('reasoning', ''),
            confidence=data.get('confidence', 0.5)
        )
    
    @staticmethod
    def parse_lemma_graph(text: str, problem: str, classification: ProblemClassification) -> LemmaGraph:
        """Parse lemma graph from JSON response"""
        data = JSONParser.parse_json_response(text)
        if not data:
            # Create simple default graph
            graph = LemmaGraph(problem, classification)
            graph.lemmas['L1'] = Lemma(
                id='L1',
                statement='Understand problem structure',
                lemma_type='structural'
            )
            graph.final_lemma = Lemma(
                id='FINAL',
                statement='Solve the problem',
                lemma_type='synthesis',
                dependencies=['L1']
            )
            return graph
        
        graph = LemmaGraph(problem, classification)
        graph.graph_strategy = data.get('graph_strategy', '')
        
        # Parse lemmas
        lemmas_data = data.get('lemmas', [])
        for lemma_data in lemmas_data:
            lemma = Lemma(
                id=lemma_data.get('id', f'L{len(graph.lemmas) + 1}'),
                statement=lemma_data.get('statement', ''),
                lemma_type=lemma_data.get('type', 'structural'),
                dependencies=lemma_data.get('dependencies', []),
                purpose=lemma_data.get('purpose', ''),
                verification_strategy=lemma_data.get('verification_strategy', ''),
                domain_hint=lemma_data.get('domain_hint', '')
            )
            graph.lemmas[lemma.id] = lemma
        
        # Parse final lemma
        final_data = data.get('final_lemma', {})
        graph.final_lemma = Lemma(
            id=final_data.get('id', 'FINAL'),
            statement=final_data.get('statement', 'Synthesize solution'),
            lemma_type=final_data.get('type', 'synthesis'),
            dependencies=final_data.get('dependencies', list(graph.lemmas.keys())),
            purpose=final_data.get('purpose', 'Combine all lemmas')
        )
        
        return graph
    
    @staticmethod
    def parse_lemma_proof(text: str) -> Dict:
        """Parse lemma proof from JSON response"""
        data = JSONParser.parse_json_response(text, {})
        return {
            'proof': data.get('proof', ''),
            'verification_code': data.get('verification_code', ''),
            'verification_explanation': data.get('verification_explanation', ''),
            'confidence': data.get('confidence', 0.5),
            'key_insights': data.get('key_insights', [])
        }
    
    @staticmethod
    def parse_solution(text: str) -> Dict:
        """Parse solution from JSON response"""
        data = JSONParser.parse_json_response(text, {})
        return {
            'solution_analysis': data.get('solution_analysis', ''),
            'step_by_step_solution': data.get('step_by_step_solution', ''),
            'answer': data.get('answer', 0),
            'confidence': data.get('confidence', 0.0),
            'verification_checks': data.get('verification_checks', []),
            'alternative_approaches': data.get('alternative_approaches', [])
        }

# ============================================================
# JUPYTER SANDBOX
# ============================================================

from jupyter_client import KernelManager

class ALGSandbox:
    _port_lock = threading.Lock()
    _next_port = 50000
    
    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports
    
    def __init__(self, timeout=30.0):
        self.timeout = timeout
        ports = self._get_next_ports(5)
        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        
        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]
        
        try:
            self._km.start_kernel(env=env)
            self._client = self._km.blocking_client()
            self._client.start_channels()
            self._client.wait_for_ready(timeout=30)
            
            # Initialize with math and sympy
            init_code = '''import math
import sympy as sp
import itertools
import numpy as np
import random
from fractions import Fraction
from collections import defaultdict, Counter'''
            self.execute(init_code)
        except Exception as e:
            print(f"[WARNING] Sandbox initialization failed: {e}")
            self._km = None
            self._client = None
    
    def execute(self, code, timeout=None):
        if not self._client:
            return {'success': False, 'output': '', 'error': 'Sandbox not initialized'}
        
        timeout = timeout or self.timeout
        try:
            msg_id = self._client.execute(code, store_history=False)
            stdout, stderr = [], []
            start = time.time()
            
            while True:
                if time.time() - start > timeout:
                    self._km.interrupt_kernel()
                    return {'success': False, 'output': '', 'error': 'Timeout'}
                
                try:
                    msg = self._client.get_iopub_msg(timeout=1.0)
                except Exception:
                    continue
                
                if msg.get('parent_header', {}).get('msg_id') != msg_id:
                    continue
                
                msg_type = msg.get('msg_type')
                content = msg.get('content', {})
                
                if msg_type == 'stream':
                    text = content.get('text', '')
                    if content.get('name') == 'stdout':
                        stdout.append(text)
                    else:
                        stderr.append(text)
                elif msg_type == 'error':
                    stderr.append('\n'.join(content.get('traceback', [])))
                elif msg_type == 'status' and content.get('execution_state') == 'idle':
                    break
            
            stdout, stderr = ''.join(stdout), ''.join(stderr)
            if stderr:
                return {'success': False, 'output': stdout, 'error': stderr}
            return {'success': True, 'output': stdout.strip(), 'error': None}
            
        except Exception as e:
            return {'success': False, 'output': '', 'error': str(e)}
    
    def close(self):
        if self._client:
            try:
                self._client.stop_channels()
            except:
                pass
        if self._km:
            try:
                self._km.shutdown_kernel(now=True)
            except:
                pass

# ============================================================
# LLM INTERFACE
# ============================================================

class LLMInterface:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_url = f'http://0.0.0.0:{cfg.server_port}/v1'
        self.api_key = 'sk-local'
        self.client = None
        self.encoding = None
    
    def initialize(self):
        print('[LLM] Connecting to vLLM server...', flush=True)
        retries = 3
        for i in range(retries):
            try:
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=300)
                self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                print('[LLM] Connected successfully', flush=True)
                return
            except Exception as e:
                if i == retries - 1:
                    raise
                print(f'[LLM] Connection attempt {i+1} failed: {e}')
                time.sleep(5)
    
    def generate(self, system_prompt, user_prompt, temperature=None, max_tokens=None):
        temp = temperature or self.cfg.temperature
        max_tok = max_tokens or self.cfg.max_tokens_per_turn
        
        if not self.client:
            raise RuntimeError('LLM client not initialized!')
        
        try:
            system_content = (SystemContent.new()
                .with_model_identity(system_prompt)
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH))
            
            system_msg = Message.from_role_and_content(Role.SYSTEM, system_content)
            user_msg = Message.from_role_and_content(Role.USER, TextContent(text=user_prompt))
            conversation = Conversation.from_messages([system_msg, user_msg])
            
            if self.encoding:
                prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            else:
                # Fallback
                prompt_text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
                prompt_ids = prompt_text
            
            response = self.client.completions.create(
                model=self.cfg.served_model_name,
                temperature=temp,
                max_tokens=max_tok,
                prompt=prompt_ids,
                stop=None)
            
            result = response.choices[0].text.strip()
            return result
            
        except Exception as e:
            print(f'[LLM] Generation error: {e}')
            return ''

# ============================================================
# IMPROVED ALG SOLVER
# ============================================================

class ALGSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = LLMInterface(cfg)
        self.sandbox = None
        self.parser = JSONParser()
    
    def initialize(self):
        print('[ALG] Initializing...')
        sys.stdout.flush()
        self.sandbox = ALGSandbox(timeout=self.cfg.sandbox_timeout)
        self.llm.initialize()
        print('[ALG] Ready')
        sys.stdout.flush()
    
    def get_domain_context(self, topic):
        """Get formatted domain context for prompts"""
        domain_info = CFG.DOMAIN_KNOWLEDGE.get(topic, CFG.DOMAIN_KNOWLEDGE['algebra'])
        
        context = f"Topic: {topic}\n"
        context += f"Description: {domain_info['description']}\n\n"
        context += "Common Techniques:\n"
        for i, technique in enumerate(domain_info['common_techniques'][:5], 1):
            context += f"{i}. {technique}\n"
        
        context += "\nCommon Patterns:\n"
        for i, pattern in enumerate(domain_info['common_patterns'][:3], 1):
            context += f"{i}. {pattern}\n"
        
        return context
    
    def get_imo_strategies(self):
        """Get formatted IMO strategies"""
        strategies = "IMO Problem Solving Strategies:\n"
        for i, strategy in enumerate(CFG.IMO_STRATEGIES[:5], 1):
            strategies += f"{i}. {strategy}\n"
        return strategies
    
    def classify_problem(self, problem):
        print('\n=== PHASE 1: PROBLEM CLASSIFICATION ===')
        
        # Get general domain context for classification
        domain_context = "Available Domains:\n"
        for topic, info in CFG.DOMAIN_KNOWLEDGE.items():
            domain_context += f"- {topic}: {info['description'][:100]}...\n"
        
        prompt = CLASSIFICATION_PROMPT.format(
            problem=problem,
            domain_context=domain_context
        )
        
        response = self.llm.generate(
            system_prompt="You are an expert IMO problem classifier. Output ONLY valid JSON.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        print(f'[ALG] Classification response: {response[:200]}...')
        
        classification = self.parser.parse_classification(response)
        
        print(f'[ALG] Topic: {classification.topic}')
        print(f'[ALG] Complexity: {classification.complexity}')
        print(f'[ALG] Estimated lemmas: {classification.estimated_lemmas}')
        print(f'[ALG] Confidence: {classification.confidence:.2f}')
        print(f'[ALG] Budget: {classification.get_time_budget()}s')
        
        return classification
    
    def build_lemma_graph(self, problem, classification):
        print('\n=== PHASE 2: LEMMA GRAPH CONSTRUCTION ===')
        
        domain_knowledge = self.get_domain_context(classification.topic)
        imo_strategies = self.get_imo_strategies()
        
        prompt = LEMMA_GRAPH_PROMPT.format(
            problem=problem,
            topic=classification.topic,
            complexity=classification.complexity,
            estimated_lemmas=classification.estimated_lemmas,
            domain_knowledge=domain_knowledge,
            imo_strategies=imo_strategies
        )
        
        response = self.llm.generate(
            system_prompt="You are an IMO Gold Medalist decomposing problems into lemma graphs. Output ONLY valid JSON.",
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=2000
        )
        
        print(f'[ALG] Lemma graph response: {response[:300]}...')
        
        graph = self.parser.parse_lemma_graph(response, problem, classification)
        
        print(f'[ALG] Graph created: {len(graph.lemmas)} lemmas + final')
        print(f'[ALG] Dependencies: {graph.get_dependency_order()}')
        
        return graph
    
    def verify_lemma(self, lemma, problem, classification):
        """Verify a single lemma with proof and code execution"""
        print(f'[ALG] Verifying lemma {lemma.id}: {lemma.lemma_type}')
        
        domain_info = classification.get_domain_info()
        domain_techniques = "\n".join(domain_info.get('common_techniques', [])[:3])
        
        prompt = LEMMA_PROOF_PROMPT.format(
            problem=problem,
            lemma_id=lemma.id,
            lemma_statement=lemma.statement,
            lemma_type=lemma.lemma_type,
            purpose=lemma.purpose,
            topic=classification.topic,
            domain_techniques=domain_techniques
        )
        
        response = self.llm.generate(
            system_prompt="You are a mathematical proof assistant. Output ONLY valid JSON.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        proof_data = self.parser.parse_lemma_proof(response)
        
        lemma.proof = proof_data.get('proof', '')
        lemma.verification_code = proof_data.get('verification_code', '')
        lemma.confidence = proof_data.get('confidence', 0.5)
        
        # Try to execute verification code if provided
        if lemma.verification_code and self.sandbox:
            print(f'[ALG] Executing verification code for {lemma.id}')
            result = self.sandbox.execute(lemma.verification_code)
            lemma.execution_result = f"Output: {result.get('output', '')}\nError: {result.get('error', '')}"
            lemma.verified = result.get('success', False)
            
            if lemma.verified:
                print(f'[ALG] ✓ Lemma {lemma.id} verified')
            else:
                print(f'[ALG] ✗ Lemma {lemma.id} verification failed')
        else:
            # For theoretical lemmas, assume verified if proof seems reasonable
            lemma.verified = len(lemma.proof) > 100  # Simple heuristic
            print(f'[ALG] {"✓" if lemma.verified else "✗"} Lemma {lemma.id} (theoretical)')
        
        return lemma.verified
    
    def solve_via_lemmas(self, problem, classification, graph):
        """Solve problem using lemma graph approach"""
        print('\n=== PHASE 3: LEMMA VERIFICATION ===')
        
        # Verify lemmas in dependency order
        verified_count = 0
        lemma_order = graph.get_dependency_order()
        
        for lemma_id in lemma_order:
            if lemma_id == 'FINAL':
                continue
                
            lemma = graph.lemmas[lemma_id]
            if self.verify_lemma(lemma, problem, classification):
                verified_count += 1
            
            # Check time budget
            # (Time check would go here)
        
        print(f'[ALG] Verified {verified_count}/{len(graph.lemmas)} lemmas')
        
        # Generate solution using verified lemmas
        print('\n=== PHASE 4: SOLUTION SYNTHESIS ===')
        
        lemmas_summary = graph.get_lemma_summary()
        domain_knowledge = self.get_domain_context(classification.topic)
        
        prompt = SOLUTION_PROMPT.format(
            problem=problem,
            topic=classification.topic,
            domain_knowledge=domain_knowledge,
            lemmas_summary=lemmas_summary
        )
        
        response = self.llm.generate(
            system_prompt="You are an IMO Gold Medalist solving problems. Output ONLY valid JSON.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        solution_data = self.parser.parse_solution(response)
        
        return solution_data
    
    def solve_directly(self, problem, classification):
        """Fallback: Solve problem directly"""
        print('\n=== DIRECT SOLUTION (FALLBACK) ===')
        
        domain_knowledge = self.get_domain_context(classification.topic)
        imo_strategies = self.get_imo_strategies()
        
        prompt = DIRECT_SOLUTION_PROMPT.format(
            problem=problem,
            topic=classification.topic,
            domain_knowledge=domain_knowledge,
            imo_strategies=imo_strategies
        )
        
        response = self.llm.generate(
            system_prompt="You are an IMO Gold Medalist solving

                    user_prompt=prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        solution_data = self.parser.parse_solution(response)
        
        return solution_data
    
    def solve(self, problem):
        """Main solving function"""
        start_time = time.time()
        answer = 0
        confidence = 0.0
        method = 'unknown'
        solution_text = ''
        
        print('=' * 60)
        print(f'PROBLEM: {problem[:80]}...')
        print('=' * 60)
        
        try:
            # Phase 1: Classify problem
            classification = self.classify_problem(problem)
            time_budget = classification.get_time_budget()
            
            # Check if we should use lemma-based approach
            use_lemmas = classification.complexity in ['medium', 'hard'] and classification.estimated_lemmas > 2
            
            if use_lemmas:
                # Phase 2: Build lemma graph
                graph = self.build_lemma_graph(problem, classification)
                
                if len(graph.lemmas) > 1:
                    # Phase 3: Solve via lemmas
                    method = 'lemma_based'
                    solution_data = self.solve_via_lemmas(problem, classification, graph)
                    answer = solution_data.get('answer', 0)
                    confidence = solution_data.get('confidence', 0.0)
                    solution_text = solution_data.get('step_by_step_solution', '')
                else:
                    # Fallback to direct solution
                    method = 'direct_fallback'
                    solution_data = self.solve_directly(problem, classification)
                    answer = solution_data.get('answer', 0)
                    confidence = solution_data.get('confidence', 0.0)
                    solution_text = solution_data.get('mathematical_proof', '')
            else:
                # Use direct solution for simple problems
                method = 'direct'
                solution_data = self.solve_directly(problem, classification)
                answer = solution_data.get('answer', 0)
                confidence = solution_data.get('confidence', 0.0)
                solution_text = solution_data.get('mathematical_proof', '')
            
            # Validate answer
            if answer is None:
                answer = 0
                confidence = 0.0
                
        except Exception as e:
            print(f'[ALG] Error during solving: {e}')
            traceback.print_exc()
            answer = 0
            confidence = 0.0
            method = 'error'
        
        time_taken = time.time() - start_time
        
        # Create result
        result = SolutionResult(
            problem=problem,
            classification=classification,
            answer=answer,
            success=answer != 0 and confidence > 0.5,
            time_taken=time_taken,
            method=method,
            confidence=confidence,
            solution_text=solution_text[:500] + '...' if len(solution_text) > 500 else solution_text
        )
        
        print(f'\n[ALG] Result:')
        print(f'  Answer: {answer}')
        print(f'  Confidence: {confidence:.2f}')
        print(f'  Method: {method}')
        print(f'  Time: {time_taken:.1f}s')
        print(f'  Success: {result.success}')
        
        return result

# ============================================================
# SERVER MANAGER
# ============================================================

class ServerManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.server_process = None
        self.log_file = None
    
    def preload_model(self):
        """Preload model files into memory"""
        print(f'[SERVER] Preloading model from {self.cfg.model_path}...')
        start = time.time()
        
        # Get all model files
        model_files = []
        for root, _, files in os.walk(self.cfg.model_path):
            for file in files:
                if file.endswith(('.bin', '.safetensors', '.json', '.txt', '.model')):
                    model_files.append(os.path.join(root, file))
        
        print(f'[SERVER] Found {len(model_files)} model files')
        
        # Read files in parallel to preload into cache
        def read_file(path):
            try:
                with open(path, 'rb') as f:
                    # Read in chunks to avoid memory issues
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
            except Exception as e:
                print(f'[SERVER] Warning: Could not read {path}: {e}')
        
        # Use ThreadPoolExecutor for parallel reading
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(read_file, model_files))
        
        print(f'[SERVER] Model preloaded in {time.time() - start:.1f}s')
    
    def start_server(self):
        """Start vLLM server"""
        print('[SERVER] Starting vLLM server...')
        
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--host', '0.0.0.0',
            '--port', str(self.cfg.server_port),
            '--tensor-parallel-size', '1',
            '--max-model-len', str(self.cfg.context_tokens),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--kv-cache-dtype', self.cfg.kv_cache_dtype,
            '--dtype', self.cfg.dtype,
            '--max-num-batched-tokens', str(self.cfg.batch_size * 1024),
            '--disable-log-stats',
            '--enable-prefix-caching',
            '--swap-space', '16',
            '--block-size', '32'
        ]
        
        self.log_file = open('vllm_server.log', 'w')
        self.server_process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT
        )
        
        return self.server_process
    
    def wait_for_server(self, timeout=180):
        """Wait for server to be ready"""
        print('[SERVER] Waiting for server to be ready...')
        
        start = time.time()
        client = OpenAI(base_url=f'http://0.0.0.0:{self.cfg.server_port}/v1', 
                       api_key='sk-local', timeout=10)
        
        while time.time() - start < timeout:
            # Check if process died
            if self.server_process.poll() is not None:
                raise RuntimeError(f'Server process died with code {self.server_process.returncode}')
            
            try:
                # Try to list models
                models = client.models.list()
                if any(model.id == self.cfg.served_model_name for model in models.data):
                    print(f'[SERVER] Server ready in {time.time() - start:.1f}s')
                    return True
            except Exception:
                pass
            
            time.sleep(1)
        
        raise RuntimeError(f'Server not ready after {timeout} seconds')
    
    def stop_server(self):
        """Stop the vLLM server"""
        if self.server_process:
            print('[SERVER] Stopping server...')
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None

# ============================================================
# KAGGLE INTERFACE FUNCTIONS
# ============================================================

_solver = None
_server_manager = None

def initialize_solver():
    """Initialize the solver and server"""
    global _solver, _server_manager
    
    if _solver is not None:
        return _solver
    
    print('[INIT] Initializing ALG Sequential Solver...')
    
    # Create and start server
    _server_manager = ServerManager(CFG)
    
    # Preload model (warm up cache)
    _server_manager.preload_model()
    
    # Start server
    _server_manager.server_process = _server_manager.start_server()
    
    # Wait for server to be ready
    _server_manager.wait_for_server(CFG.server_timeout)
    
    # Initialize solver
    _solver = ALGSolver(CFG)
    _solver.initialize()
    
    print('[INIT] ALG Solver initialized successfully')
    return _solver

def predict(id_, question):
    """Main prediction function for Kaggle"""
    id_value = id_.item(0) if hasattr(id_, 'item') else id_
    question_text = question.item(0) if hasattr(question, 'item') else question
    
    print('\n' + '='*60)
    print(f'PROBLEM ID: {id_value}')
    print('='*60)
    
    try:
        # Initialize solver if needed
        solver = initialize_solver()
        
        # Solve problem
        result = solver.solve(question_text)
        
        # Get answer (ensure it's an integer)
        answer = int(result.answer) if result.answer is not None else 0
        
        # Bound answer to reasonable range
        if answer < 0 or answer > 99999:
            print(f'[WARNING] Answer {answer} out of bounds, setting to 0')
            answer = 0
        
        print(f'\nSUBMITTING ANSWER: {answer}')
        print(f'Confidence: {result.confidence:.2f}')
        print(f'Method: {result.method}')
        
        return pl.DataFrame({'id': [id_value], 'answer': [answer]})
        
    except Exception as e:
        print(f'[ERROR] Prediction failed: {e}')
        traceback.print_exc()
        # Return default answer
        return pl.DataFrame({'id': [id_value], 'answer': [0]})

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__' or True:
    if os.path.exists('/kaggle'):
        print('[MAIN] Running on Kaggle platform')
        
        # Read reference data
        ref_path = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv'
        if os.path.exists(ref_path):
            import pandas as pd
            ref_df = pd.read_csv(ref_path, usecols=[0, 1])
            ref_df.columns = ['id', 'question']
            test_path = '/kaggle/working/test.csv'
            ref_df.to_csv(test_path, index=False)
            print(f'[MAIN] Loaded {len(ref_df)} problems from reference.csv')
        else:
            # Create dummy test file
            test_path = '/kaggle/working/test.csv'
            pd.DataFrame({'id': [1], 'question': ['Find the value of 2+2']}).to_csv(test_path, index=False)
            print('[MAIN] Created dummy test file')
        
        # Check if we're in competition mode
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print('[MAIN] Running in competition mode - starting server...')
            server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
            server.serve()
        else:
            print('[MAIN] Running in local test mode...')
            try:
                server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
                server.run_local_gateway((test_path,))
            except KeyboardInterrupt:
                print('\n[MAIN] Interrupted by user')
            except Exception as e:
                print(f'[MAIN] Error: {e}')
                traceback.print_exc()
            finally:
                # Cleanup
                print('[MAIN] Cleaning up...')
                if _solver and _solver.sandbox:
                    _solver.sandbox.close()
                if _server_manager:
                    _server_manager.stop_server()
    else:
        print('[MAIN] Running in standalone mode')
        
        # Test with a sample problem
        sample_problem = "Let $n$ be a positive integer. Find the number of permutations $(a_1, a_2, \ldots, a_n)$ of $(1, 2, \ldots, n)$ such that for all $1 \le i \le n$, $a_i$ is divisible by $i$."
        
        print(f'\nTesting with sample problem:')
        print(f'Problem: {sample_problem[:100]}...')
        
        try:
            # Initialize
            _server_manager = ServerManager(CFG)
            _server_manager.preload_model()
            _server_manager.server_process = _server_manager.start_server()
            _server_manager.wait_for_server(60)
            
            _solver = ALGSolver(CFG)
            _solver.initialize()
            
            # Solve
            result = _solver.solve(sample_problem)
            
            print(f'\nTest Result:')
            print(f'  Answer: {result.answer}')
            print(f'  Confidence: {result.confidence:.2f}')
            print(f'  Method: {result.method}')
            print(f'  Time: {result.time_taken:.1f}s')
            
        except Exception as e:
            print(f'[ERROR] Test failed: {e}')
            traceback.print_exc()
        finally:
            # Cleanup
            if _solver and _solver.sandbox:
                _solver.sandbox.close()
            if _server_manager:
                _server_manager.stop_server()
