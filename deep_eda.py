"""
Deep EDA on filtered Nemotron Math v2 dataset
==============================================
Verifies:
1. All answers are valid integers [0, 99999]
2. Solutions have actual reasoning (not just "answer = N")
3. Problems are competition-math style
4. No data quality issues

Run: DATA_DIR=./data/nemotron-math-v2 python3 deep_eda.py
"""

import re, json, os, random
from collections import Counter, defaultdict
from datasets import load_from_disk

DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-math-v2")
os.makedirs("eda_results2", exist_ok=True)

print(f"Loading from {DATA_DIR}...")
ds = load_from_disk(DATA_DIR)
print(f"Total: {len(ds)}\n")

problems  = [ex['messages'][0]['content'] for ex in ds]
solutions = [ex['messages'][1]['content'] for ex in ds]

# ============================================================
# 1. VERIFY ALL ANSWERS ARE VALID INTEGERS [0, 99999]
# ============================================================
print("="*60)
print("1. ANSWER VERIFICATION")
print("="*60)

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if not matches:
        return None
    raw = matches[-1].strip().replace(',','').replace(' ','')
    try:
        return int(float(raw))
    except:
        return raw

answers = [extract_boxed(s) for s in solutions]
int_answers   = [(i,a) for i,a in enumerate(answers) if isinstance(a,int)]
str_answers   = [(i,a) for i,a in enumerate(answers) if isinstance(a,str)]
none_answers  = [(i,a) for i,a in enumerate(answers) if a is None]
valid_range   = [(i,a) for i,a in int_answers if 0 <= a <= 99999]
invalid_range = [(i,a) for i,a in int_answers if not (0 <= a <= 99999)]

print(f"Integer answers:           {len(int_answers):6d} ({100*len(int_answers)/len(ds):.1f}%)")
print(f"  Valid [0,99999]:         {len(valid_range):6d} ({100*len(valid_range)/len(ds):.1f}%)")
print(f"  Outside range:           {len(invalid_range):6d}")
print(f"String/expression answers: {len(str_answers):6d} ({100*len(str_answers)/len(ds):.1f}%)")
print(f"No boxed answer:           {len(none_answers):6d} ({100*len(none_answers)/len(ds):.1f}%)")

if invalid_range:
    print(f"\nSamples OUTSIDE [0,99999]:")
    for i,a in invalid_range[:5]:
        print(f"  answer={a} | problem: {problems[i][:80]}...")

if str_answers:
    print(f"\nSamples with STRING answers (should be 0 if fully filtered):")
    for i,a in str_answers[:5]:
        print(f"  answer='{a}' | problem: {problems[i][:80]}...")

if none_answers:
    print(f"\nSamples with NO boxed answer (should be 0 if fully filtered):")
    for i,_ in none_answers[:5]:
        print(f"  solution: {solutions[i][:150]}...")

# ============================================================
# 2. ANSWER DISTRIBUTION
# ============================================================
print(f"\n{'='*60}")
print("2. ANSWER DISTRIBUTION")
print("="*60)

all_valid_answers = [a for _,a in valid_range]
ans_counter = Counter(all_valid_answers)

print(f"Unique answer values:    {len(ans_counter)}")
print(f"Most common answers:     {ans_counter.most_common(20)}")
print(f"Answers = 0:             {ans_counter[0]}")
print(f"Answers = 1:             {ans_counter[1]}")
print(f"Answers > 10000:         {sum(1 for a in all_valid_answers if a > 10000)}")
print(f"Answers > 50000:         {sum(1 for a in all_valid_answers if a > 50000)}")

# Range buckets
buckets = [(0,0), (1,9), (10,99), (100,999), (1000,9999), (10000,99999)]
print(f"\nAnswer range distribution:")
for lo, hi in buckets:
    count = sum(1 for a in all_valid_answers if lo <= a <= hi)
    print(f"  [{lo:6d}, {hi:6d}]: {count:6d} ({100*count/len(all_valid_answers):.1f}%)")

# ============================================================
# 3. SOLUTION QUALITY CHECK
# ============================================================
print(f"\n{'='*60}")
print("3. SOLUTION QUALITY")
print("="*60)

sol_word_counts = [len(s.split()) for s in solutions]
sol_char_counts = [len(s) for s in solutions]

print(f"Solution lengths (words):")
sorted_wc = sorted(sol_word_counts)
n = len(sorted_wc)
print(f"  min={sorted_wc[0]} p10={sorted_wc[n//10]} p25={sorted_wc[n//4]} "
      f"median={sorted_wc[n//2]} p75={sorted_wc[3*n//4]} "
      f"p90={sorted_wc[9*n//10]} max={sorted_wc[-1]}")

# Quality flags
too_short      = sum(1 for w in sol_word_counts if w < 20)
just_answer    = sum(1 for s in solutions if len(s.split()) < 5)
has_steps      = sum(1 for s in solutions if any(
    kw in s.lower() for kw in ['step', 'first', 'note that', 'therefore', 
                                 'since', 'because', 'we have', 'let']))
has_math       = sum(1 for s in solutions if any(
    sym in s for sym in ['\\frac', '\\sum', '\\prod', '=', '\\leq', '\\geq']))
has_proof      = sum(1 for s in solutions if any(
    kw in s.lower() for kw in ['proof', 'qed', 'thus', 'hence', 'therefore']))

print(f"\nQuality indicators:")
print(f"  Too short (<20 words):   {too_short:6d} ({100*too_short/len(ds):.1f}%)")
print(f"  Just answer (<5 words):  {just_answer:6d} ({100*just_answer/len(ds):.1f}%)")
print(f"  Has reasoning steps:     {has_steps:6d} ({100*has_steps/len(ds):.1f}%)")
print(f"  Has math notation:       {has_math:6d} ({100*has_math/len(ds):.1f}%)")
print(f"  Has proof language:      {has_proof:6d} ({100*has_proof/len(ds):.1f}%)")

# ============================================================
# 4. PROBLEM QUALITY CHECK
# ============================================================
print(f"\n{'='*60}")
print("4. PROBLEM QUALITY")
print("="*60)

prob_word_counts = [len(p.split()) for p in problems]
sorted_pw = sorted(prob_word_counts)
n = len(sorted_pw)
print(f"Problem lengths (words):")
print(f"  min={sorted_pw[0]} p10={sorted_pw[n//10]} p25={sorted_pw[n//4]} "
      f"median={sorted_pw[n//2]} p75={sorted_pw[3*n//4]} "
      f"p90={sorted_pw[9*n//10]} max={sorted_pw[-1]}")

# Competition math indicators
has_find       = sum(1 for p in problems if 'find' in p.lower())
has_compute    = sum(1 for p in problems if 'compute' in p.lower())
has_determine  = sum(1 for p in problems if 'determine' in p.lower())
has_prove      = sum(1 for p in problems if 'prove' in p.lower())
has_latex      = sum(1 for p in problems if '$' in p or '\\' in p)
has_boxed_req  = sum(1 for p in problems if 'boxed' in p.lower())

print(f"\nProblem type indicators:")
print(f"  'Find ...' problems:     {has_find:6d} ({100*has_find/len(ds):.1f}%)")
print(f"  'Compute ...' problems:  {has_compute:6d} ({100*has_compute/len(ds):.1f}%)")
print(f"  'Determine ...' probs:   {has_determine:6d} ({100*has_determine/len(ds):.1f}%)")
print(f"  'Prove ...' problems:    {has_prove:6d} ({100*has_prove/len(ds):.1f}%)")
print(f"  Has LaTeX:               {has_latex:6d} ({100*has_latex/len(ds):.1f}%)")
print(f"  Asks for \\boxed:         {has_boxed_req:6d} ({100*has_boxed_req/len(ds):.1f}%)")

# ============================================================
# 5. DOMAIN DISTRIBUTION
# ============================================================
print(f"\n{'='*60}")
print("5. DOMAIN DISTRIBUTION")
print("="*60)

DOMAIN_KEYWORDS = {
    'number_theory': ['prime', 'divisib', 'modulo', 'gcd', 'lcm', 'congruent',
                      'remainder', 'digit', 'coprime', 'totient', 'factorial'],
    'combinatorics': ['choose', 'combinat', 'permut', 'arrange', 'subset',
                      'count', 'ways', 'select', 'committee', 'distribute',
                      'graph', 'coloring', 'tournament', 'sequence of'],
    'algebra':       ['polynomial', 'equation', 'function', 'inequality',
                      'maximum', 'minimum', 'real number', 'complex number',
                      'functional equation', 'recurrence'],
    'geometry':      ['triangle', 'circle', 'angle', 'polygon', 'area',
                      'perpendicular', 'tangent', 'chord', 'radius',
                      'inscribed', 'circumscribed', 'coordinate'],
    'number_theory_hard': ['primitive root', 'quadratic residue', 'p-adic',
                            'lifting the exponent', 'zsygmondy'],
}

def classify(text):
    text_lower = text.lower()
    scores = {d: sum(1 for kw in kws if kw in text_lower) 
              for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'other'

print("Classifying domains (this takes ~30s)...")
domains = [classify(p) for p in problems]
dom_counts = Counter(domains)
print(f"\nDomain distribution:")
for d, c in dom_counts.most_common():
    print(f"  {d:25s}: {c:6d} ({100*c/len(ds):.1f}%)")

# ============================================================
# 6. SPOT CHECK RANDOM SAMPLES
# ============================================================
print(f"\n{'='*60}")
print("6. RANDOM SAMPLE SPOT CHECK")
print("="*60)

random.seed(42)
sample_indices = random.sample(range(len(ds)), 10)
samples = []
for i in sample_indices:
    ans = answers[i]
    samples.append({
        'index': i,
        'domain': domains[i],
        'answer': ans,
        'problem_words': prob_word_counts[i],
        'solution_words': sol_word_counts[i],
        'problem': problems[i][:200],
        'solution': solutions[i][:400],
    })

print("10 random samples (check for quality):")
for s in samples:
    print(f"\n  [{s['index']}] domain={s['domain']} answer={s['answer']}")
    print(f"  prob_words={s['problem_words']} sol_words={s['solution_words']}")
    print(f"  PROBLEM: {s['problem'][:120]}...")
    print(f"  SOLUTION: {s['solution'][:200]}...")

with open('eda_results2/samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

# ============================================================
# 7. NEMOTRON "HIGH" FILTER CHECK
# ============================================================
print(f"\n{'='*60}")
print("7. CHECKING NEMOTRON 'HIGH' FILTER INDICATORS")
print("="*60)

# The original Nemotron-Math-v2 has quality ratings
# Check if there's metadata about the 'high' filter
sample = ds[0]
print(f"Example record keys: {list(sample.keys())}")
print(f"Messages structure:")
for msg in sample['messages']:
    print(f"  role={msg['role']} content_len={len(msg['content'])} preview={msg['content'][:100]}...")

# Check for any extra fields
extra_fields = {k: v for k, v in sample.items() if k != 'messages'}
if extra_fields:
    print(f"\nExtra fields found: {extra_fields}")
else:
    print(f"\nOnly 'messages' field — dataset is already pre-filtered")

# ============================================================
# 8. SUMMARY + RAG RECOMMENDATION
# ============================================================
print(f"\n{'='*60}")
print("8. SUMMARY & RAG RECOMMENDATIONS")
print("="*60)

total_clean = len(valid_range)
total_with_reasoning = sum(1 for i,_ in valid_range if sol_word_counts[i] >= 30)

print(f"""
DATASET HEALTH CHECK:
  Total examples:              {len(ds):6d}
  Valid integer answers:       {len(valid_range):6d} ({100*len(valid_range)/len(ds):.1f}%)
  Outside [0,99999]:           {len(invalid_range):6d}
  String answers (non-int):    {len(str_answers):6d}
  No boxed answer:             {len(none_answers):6d}

SOLUTION QUALITY:
  Has actual reasoning (≥30w): {total_with_reasoning:6d} ({100*total_with_reasoning/len(ds):.1f}%)
  Too short (<20 words):       {too_short:6d} ({100*too_short/len(ds):.1f}%)
  Has math notation:           {has_math:6d} ({100*has_math/len(ds):.1f}%)

FOR RAG - USE:
  High quality (valid int + ≥30w solution): {total_with_reasoning}
  This is your core RAG corpus.

RAG BUILD PLAN:
  1. Filter to {total_with_reasoning} high-quality examples
  2. Embed problem texts with BAAI/bge-large-en-v1.5
  3. Build single FAISS index (start simple)
  4. At retrieval: fetch top-5 similar problems + solutions
  5. Truncate solutions to 300 words for context budget
  6. Total context added per problem: ~1500 words = ~2000 tokens
  7. Fits comfortably in 65536 token context window
""")

# Save final counts
with open('eda_results2/final_counts.json', 'w') as f:
    json.dump({
        'total': len(ds),
        'valid_integer_answers': len(valid_range),
        'invalid_range': len(invalid_range),
        'string_answers': len(str_answers),
        'no_boxed_answer': len(none_answers),
        'high_quality_for_rag': total_with_reasoning,
        'domain_distribution': dict(dom_counts),
        'answer_distribution_top20': dict(ans_counter.most_common(20)),
        'median_solution_words': sorted_wc[n//2],
        'median_problem_words': sorted_pw[n//2],
    }, f, indent=2)

print("Saved to eda_results2/")
