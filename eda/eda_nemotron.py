"""
Nemotron Math v2 EDA
====================
Run this on your server where the dataset lives:
    python3 eda_nemotron.py

Outputs:
    eda_results/summary.txt        -- key stats
    eda_results/samples.jsonl      -- representative samples per category
    eda_results/techniques.json    -- technique frequency counts
    eda_results/length_dist.json   -- token length distribution
"""

import os, re, json, math
from collections import Counter, defaultdict
from datasets import load_from_disk

os.makedirs("eda_results", exist_ok=True)

DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-math-v2")
print(f"Loading dataset from {DATA_DIR}...")
ds = load_from_disk(DATA_DIR)
print(f"Total examples: {len(ds)}")

# ============================================================
# BASIC STATS
# ============================================================

problems   = [ex['messages'][0]['content'] for ex in ds]
solutions  = [ex['messages'][1]['content'] for ex in ds]

prob_lens  = [len(p.split()) for p in problems]
sol_lens   = [len(s.split()) for s in solutions]

print(f"\nProblem lengths (words):")
print(f"  min={min(prob_lens)} max={max(prob_lens)} "
      f"mean={sum(prob_lens)/len(prob_lens):.0f} "
      f"median={sorted(prob_lens)[len(prob_lens)//2]}")

print(f"\nSolution lengths (words):")
print(f"  min={min(sol_lens)} max={max(sol_lens)} "
      f"mean={sum(sol_lens)/len(sol_lens):.0f} "
      f"median={sorted(sol_lens)[len(sol_lens)//2]}")

# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if not matches:
        return None
    raw = matches[-1].strip().replace(',','').replace(' ','')
    try:
        return int(float(raw))
    except:
        return raw  # return as string if not int

answers = [extract_answer(s) for s in solutions]
int_answers = [a for a in answers if isinstance(a, int)]
str_answers = [a for a in answers if isinstance(a, str)]
none_answers = [a for a in answers if a is None]

print(f"\nAnswer types:")
print(f"  Integer answers: {len(int_answers)} ({100*len(int_answers)/len(ds):.1f}%)")
print(f"  String answers:  {len(str_answers)} ({100*len(str_answers)/len(ds):.1f}%)")
print(f"  No boxed answer: {len(none_answers)} ({100*len(none_answers)/len(ds):.1f}%)")

if int_answers:
    valid_aimo = [a for a in int_answers if 0 <= a <= 99999]
    print(f"  Valid AIMO3 range [0,99999]: {len(valid_aimo)} ({100*len(valid_aimo)/len(ds):.1f}%)")
    print(f"  Answer distribution (top 20): {Counter(int_answers).most_common(20)}")

# ============================================================
# DOMAIN / CATEGORY DETECTION
# ============================================================

DOMAIN_KEYWORDS = {
    'number_theory': [
        'prime', 'divisib', 'modulo', 'gcd', 'lcm', 'congruent',
        'remainder', 'digit', 'integer', 'factor', 'coprime',
        'totient', 'primitive root', 'quadratic residue'
    ],
    'combinatorics': [
        'choose', 'combinat', 'permut', 'arrange', 'subset',
        'count', 'ways', 'select', 'committee', 'distribute',
        'graph', 'path', 'cycle', 'coloring', 'tournament'
    ],
    'algebra': [
        'polynomial', 'equation', 'function', 'sequence', 'series',
        'inequality', 'maximum', 'minimum', 'real number', 'complex',
        'functional equation', 'recurrence', 'sum of', 'product of'
    ],
    'geometry': [
        'triangle', 'circle', 'angle', 'polygon', 'area', 'length',
        'perpendicular', 'parallel', 'tangent', 'chord', 'radius',
        'inscribed', 'circumscribed', 'coordinate', 'point'
    ],
    'sequences': [
        'sequence', 'series', 'term', 'fibonacci', 'arithmetic',
        'geometric', 'recurrence', 'limit', 'converge'
    ]
}

def classify_domain(text):
    text_lower = text.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'other'

print("\nClassifying domains...")
domains = [classify_domain(p) for p in problems]
domain_counts = Counter(domains)
print(f"\nDomain distribution:")
for domain, count in domain_counts.most_common():
    print(f"  {domain:20s}: {count:6d} ({100*count/len(ds):.1f}%)")

# ============================================================
# TECHNIQUE DETECTION
# ============================================================

TECHNIQUES = {
    # Number theory
    'modular_arithmetic':    ['mod', 'modulo', 'congruent', 'remainder'],
    'tower_exponentiation':  ['tower', 'a^(b^c)', 'power tower', 'tetration'],
    'chinese_remainder':     ['chinese remainder', 'crt', 'coprime moduli'],
    'inclusion_exclusion':   ['inclusion-exclusion', 'inclusion exclusion', 'PIE'],
    'generating_functions':  ['generating function', 'power series', 'ogf', 'egf'],
    'functional_equations':  ['functional equation', 'f(x+y)', 'f(xy)', 'cauchy'],
    'linear_recurrences':    ['recurrence', 'a_n =', 'linear recurrence', 'characteristic'],
    'pigeonhole':            ['pigeonhole', 'pigeon hole', 'drawer principle'],
    'invariant':             ['invariant', 'monovariant', 'conservation'],
    'induction':             ['induction', 'base case', 'inductive step'],
    'extremal':              ['extremal', 'maximum', 'minimum', 'optimal'],
    'bijection':             ['bijection', 'bijective', 'one-to-one correspondence'],
    'double_counting':       ['double count', 'count two ways', 'overcounting'],
    'vieta_jumping':         ['vieta jumping', 'vieta', 'by symmetry swap'],
    'infinite_descent':      ['infinite descent', 'smallest', 'well-ordering'],
    'am_gm':                 ['am-gm', 'am gm', 'arithmetic mean', 'geometric mean'],
    'cauchy_schwarz':        ['cauchy-schwarz', 'cauchy schwarz', 'CBS'],
    'polynomial_roots':      ['polynomial', 'roots', 'vieta formulas', 'factor theorem'],
    'floor_ceiling':         ['floor', 'ceiling', '\\lfloor', '\\lceil', 'greatest integer'],
    'combinatorial_identity':['binomial', 'pascal', 'vandermonde', 'hockey stick'],
    'graph_theory':          ['graph', 'vertex', 'edge', 'spanning tree', 'chromatic'],
    'probability':           ['probability', 'expected value', 'random', 'expectation'],
    'geometry_coordinate':   ['coordinate', 'x-axis', 'y-axis', 'slope', 'distance formula'],
    'geometry_classical':    ['similar triangle', 'power of a point', 'ptolemy', 'radical axis'],
    'number_theory_advanced':['lifting the exponent', 'LTE', 'zsygmondy', 'primitive root'],
    'matrix':                ['matrix', 'determinant', 'eigenvalue', 'linear algebra'],
    'complex_numbers':       ['complex number', 'argument', 'modulus', 'roots of unity'],
    'python_computation':    ['```python', 'sympy', 'import math', 'brute force', 'for i in range'],
}

print("\nDetecting techniques in solutions...")
technique_counts = Counter()
technique_by_domain = defaultdict(Counter)
problem_techniques = []

for i, (sol, domain) in enumerate(zip(solutions, domains)):
    sol_lower = sol.lower()
    prob_lower = problems[i].lower()
    combined = sol_lower + ' ' + prob_lower
    
    found = []
    for technique, keywords in TECHNIQUES.items():
        if any(kw.lower() in combined for kw in keywords):
            found.append(technique)
            technique_counts[technique] += 1
            technique_by_domain[domain][technique] += 1
    
    problem_techniques.append(found)

print(f"\nTop techniques across all problems:")
for tech, count in technique_counts.most_common(25):
    print(f"  {tech:30s}: {count:6d} ({100*count/len(ds):.1f}%)")

print(f"\nTop techniques by domain:")
for domain in domain_counts.most_common():
    domain_name = domain[0]
    print(f"\n  {domain_name}:")
    for tech, count in technique_by_domain[domain_name].most_common(8):
        print(f"    {tech:30s}: {count:5d}")

# ============================================================
# PYTHON CODE USAGE IN SOLUTIONS
# ============================================================

has_python = sum(1 for s in solutions if '```python' in s.lower() or 'import' in s)
has_sympy  = sum(1 for s in solutions if 'sympy' in s.lower())
has_boxed  = sum(1 for s in solutions if '\\boxed' in s)

print(f"\nSolution characteristics:")
print(f"  Has Python code:   {has_python:6d} ({100*has_python/len(ds):.1f}%)")
print(f"  Uses sympy:        {has_sympy:6d} ({100*has_sympy/len(ds):.1f}%)")
print(f"  Has \\boxed answer: {has_boxed:6d} ({100*has_boxed/len(ds):.1f}%)")

# Solution length buckets
short  = sum(1 for l in sol_lens if l < 200)
medium = sum(1 for l in sol_lens if 200 <= l < 1000)
long_  = sum(1 for l in sol_lens if 1000 <= l < 3000)
vlong  = sum(1 for l in sol_lens if l >= 3000)
print(f"\nSolution length buckets (words):")
print(f"  Short  (<200):     {short:6d} ({100*short/len(ds):.1f}%)")
print(f"  Medium (200-1k):   {medium:6d} ({100*medium/len(ds):.1f}%)")
print(f"  Long   (1k-3k):    {long_:6d} ({100*long_/len(ds):.1f}%)")
print(f"  V.Long (>3k):      {vlong:6d} ({100*vlong/len(ds):.1f}%)")

# ============================================================
# FUNCTIONAL EQUATION DEEP DIVE (most relevant for RAG)
# ============================================================

fe_indices = [i for i, techs in enumerate(problem_techniques) 
              if 'functional_equations' in techs]
print(f"\nFunctional equation problems: {len(fe_indices)}")

# What techniques co-occur with functional equations?
fe_technique_cooccurrence = Counter()
for i in fe_indices:
    for tech in problem_techniques[i]:
        if tech != 'functional_equations':
            fe_technique_cooccurrence[tech] += 1

print(f"Techniques co-occurring with functional equations:")
for tech, count in fe_technique_cooccurrence.most_common(10):
    print(f"  {tech:30s}: {count}")

# ============================================================
# SAMPLE PROBLEMS PER CATEGORY FOR RAG QUALITY CHECK
# ============================================================

print("\nSaving sample problems...")
samples = {}
for domain in domain_counts.keys():
    domain_indices = [i for i, d in enumerate(domains) if d == domain]
    # Take 5 representative samples (mix of lengths)
    selected = domain_indices[:5] if len(domain_indices) >= 5 else domain_indices
    samples[domain] = []
    for i in selected:
        samples[domain].append({
            'problem': problems[i][:500],
            'solution_preview': solutions[i][:500],
            'techniques': problem_techniques[i],
            'answer': str(answers[i]),
            'sol_len_words': sol_lens[i],
        })

with open('eda_results/samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

# ============================================================
# RAG STRATEGY RECOMMENDATIONS
# ============================================================

print("\nGenerating RAG recommendations...")

rag_recs = {
    'dataset_size': len(ds),
    'domain_distribution': dict(domain_counts),
    'top_techniques': dict(technique_counts.most_common(20)),
    'technique_by_domain': {d: dict(t.most_common(10)) 
                             for d, t in technique_by_domain.items()},
    'has_python_solutions_pct': round(100*has_python/len(ds), 1),
    'median_solution_words': sorted(sol_lens)[len(sol_lens)//2],
    'functional_equation_count': len(fe_indices),
    'recommendations': {
        'index_strategy': (
            'Build separate FAISS indices per domain for better retrieval precision. '
            'Problem text embedding is better than solution embedding for similarity search.'
        ),
        'context_budget': (
            f'Median solution is {sorted(sol_lens)[len(sol_lens)//2]} words. '
            'For RAG context, truncate retrieved solutions to 300-500 words '
            'to avoid context overflow while preserving key techniques.'
        ),
        'technique_cards': (
            f'Top 5 techniques to build strategy cards for: '
            + str([t for t, _ in technique_counts.most_common(5)])
        ),
        'functional_eq_special': (
            f'{len(fe_indices)} functional equation problems available. '
            'Build specialized retrieval for these — they need different strategies '
            'than computational problems. Key: residue class analysis.'
        ),
        'python_solutions': (
            f'{100*has_python/len(ds):.1f}% of solutions contain Python code. '
            'These are highest value for RAG — model can directly reuse code patterns.'
        )
    }
}

with open('eda_results/rag_recommendations.json', 'w') as f:
    json.dump(rag_recs, f, indent=2)

# ============================================================
# SUMMARY OUTPUT
# ============================================================

summary = f"""
NEMOTRON MATH V2 EDA SUMMARY
=============================
Total problems: {len(ds)}

DOMAIN DISTRIBUTION:
{chr(10).join(f'  {d}: {c} ({100*c/len(ds):.1f}%)' for d,c in domain_counts.most_common())}

ANSWER TYPES:
  Integer (AIMO-compatible): {len(int_answers)} ({100*len(int_answers)/len(ds):.1f}%)
  Valid AIMO range [0-99999]: {len([a for a in int_answers if 0<=a<=99999])}
  String/expression:         {len(str_answers)}
  No boxed answer:           {len(none_answers)}

TOP 15 TECHNIQUES:
{chr(10).join(f'  {t}: {c} ({100*c/len(ds):.1f}%)' for t,c in technique_counts.most_common(15))}

SOLUTION STATS:
  Has Python code: {has_python} ({100*has_python/len(ds):.1f}%)
  Median length:   {sorted(sol_lens)[len(sol_lens)//2]} words
  
FUNCTIONAL EQUATIONS:
  Total FE problems: {len(fe_indices)}
  
RAG RECOMMENDATIONS:
  1. Build 5 domain-specific FAISS indices
  2. Truncate retrieved solutions to 400 words
  3. Special index for functional equations
  4. Prioritize problems with Python code solutions for retrieval
  5. Build technique strategy cards for top 10 techniques
"""

print(summary)
with open('eda_results/summary.txt', 'w') as f:
    f.write(summary)

print("\nAll outputs saved to eda_results/")
print("Next steps:")
print("  1. Review eda_results/summary.txt")  
print("  2. Check eda_results/samples.json for quality")
print("  3. Use eda_results/rag_recommendations.json to build RAG")
