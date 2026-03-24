"""
Deep EDA using actual metadata fields
======================================
Now we know the real structure:
- expected_answer: the ground truth answer
- metadata: accuracy scores at different reasoning levels
- data_source: where the problem came from
- messages: problem + solution

Run: DATA_DIR=./data/nemotron-math-v2 python3 deep_eda2.py
"""

import re, json, os, random
from collections import Counter, defaultdict
from datasets import load_from_disk

DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-math-v2")
os.makedirs("eda_results3", exist_ok=True)

print(f"Loading from {DATA_DIR}...")
ds = load_from_disk(DATA_DIR)
print(f"Total: {len(ds)}\n")

# ============================================================
# 1. USE expected_answer FIELD (not boxed extraction)
# ============================================================
print("="*60)
print("1. EXPECTED_ANSWER FIELD ANALYSIS")
print("="*60)

def parse_expected_answer(val):
    """Parse expected_answer field to integer if possible."""
    if val is None:
        return None
    s = str(val).strip().replace(',','').replace(' ','')
    try:
        v = int(float(s))
        if 0 <= v <= 99999:
            return v
        return None
    except:
        return None

expected_answers = [ex.get('expected_answer') for ex in ds]
parsed_answers   = [parse_expected_answer(a) for a in expected_answers]

valid_int   = [(i,a) for i,a in enumerate(parsed_answers) if a is not None]
invalid     = [(i,a) for i,a in enumerate(expected_answers) 
               if parse_expected_answer(a) is None]

print(f"Valid integer answers [0,99999]: {len(valid_int):6d} ({100*len(valid_int)/len(ds):.1f}%)")
print(f"Non-integer / out of range:      {len(invalid):6d} ({100*len(invalid)/len(ds):.1f}%)")
print(f"\nSample non-integer expected_answers:")
for i,a in invalid[:10]:
    print(f"  '{a}'")

# ============================================================
# 2. METADATA ANALYSIS - what does 'high' mean?
# ============================================================
print(f"\n{'='*60}")
print("2. METADATA QUALITY SCORES")
print("="*60)

# Extract accuracy scores
high_with_tool_accs  = []
high_no_tool_accs    = []
medium_no_tool_accs  = []
low_no_tool_accs     = []

for ex in ds:
    meta = ex.get('metadata', {})
    if not meta:
        continue
    if 'reason_high_with_tool' in meta:
        high_with_tool_accs.append(meta['reason_high_with_tool'].get('accuracy', 0))
    if 'reason_high_no_tool' in meta:
        high_no_tool_accs.append(meta['reason_high_no_tool'].get('accuracy', 0))
    if 'reason_medium_no_tool' in meta:
        medium_no_tool_accs.append(meta['reason_medium_no_tool'].get('accuracy', 0))
    if 'reason_low_no_tool' in meta:
        low_no_tool_accs.append(meta['reason_low_no_tool'].get('accuracy', 0))

def stats(arr):
    if not arr:
        return "N/A"
    arr = sorted(arr)
    n = len(arr)
    return (f"min={arr[0]:.2f} median={arr[n//2]:.2f} "
            f"mean={sum(arr)/n:.2f} max={arr[-1]:.2f}")

print(f"Accuracy distributions:")
print(f"  high_with_tool:  {stats(high_with_tool_accs)}")
print(f"  high_no_tool:    {stats(high_no_tool_accs)}")
print(f"  medium_no_tool:  {stats(medium_no_tool_accs)}")
print(f"  low_no_tool:     {stats(low_no_tool_accs)}")

# What does 'high' filter mean?
# Count problems by their high accuracy threshold
if high_no_tool_accs:
    acc_dist = Counter(round(a, 2) for a in high_no_tool_accs)
    print(f"\nhigh_no_tool accuracy distribution:")
    for acc in sorted(acc_dist.keys()):
        print(f"  {acc:.2f}: {acc_dist[acc]:5d} problems")

# ============================================================
# 3. DATA SOURCE DISTRIBUTION
# ============================================================
print(f"\n{'='*60}")
print("3. DATA SOURCES")
print("="*60)

sources = [ex.get('data_source', 'unknown') for ex in ds]
source_counts = Counter(sources)
print(f"Data sources ({len(source_counts)} unique):")
for src, count in source_counts.most_common(20):
    print(f"  {src:30s}: {count:6d} ({100*count/len(ds):.1f}%)")

# ============================================================
# 4. SOLUTION QUALITY - EMPTY vs REAL
# ============================================================
print(f"\n{'='*60}")
print("4. SOLUTION QUALITY BREAKDOWN")
print("="*60)

solutions = [ex['messages'][1]['content'] for ex in ds]
problems  = [ex['messages'][0]['content'] for ex in ds]

sol_lens = [len(s.split()) for s in solutions]

empty_solutions  = sum(1 for l in sol_lens if l == 0)
tiny_solutions   = sum(1 for l in sol_lens if 0 < l < 20)
short_solutions  = sum(1 for l in sol_lens if 20 <= l < 100)
medium_solutions = sum(1 for l in sol_lens if 100 <= l < 500)
long_solutions   = sum(1 for l in sol_lens if l >= 500)

print(f"Solution length breakdown:")
print(f"  Empty (0 words):         {empty_solutions:6d} ({100*empty_solutions/len(ds):.1f}%)")
print(f"  Tiny (1-19 words):       {tiny_solutions:6d} ({100*tiny_solutions/len(ds):.1f}%)")
print(f"  Short (20-99 words):     {short_solutions:6d} ({100*short_solutions/len(ds):.1f}%)")
print(f"  Medium (100-499 words):  {medium_solutions:6d} ({100*medium_solutions/len(ds):.1f}%)")
print(f"  Long (500+ words):       {long_solutions:6d} ({100*long_solutions/len(ds):.1f}%)")

# What are the empty solutions?
empty_indices = [i for i,l in enumerate(sol_lens) if l == 0]
print(f"\nSample empty solution problems:")
for i in empty_indices[:5]:
    ex = ds[i]
    print(f"  source={ex.get('data_source')} "
          f"answer={ex.get('expected_answer')} "
          f"problem: {problems[i][:100]}...")

# ============================================================
# 5. THE REAL USABLE DATASET
# ============================================================
print(f"\n{'='*60}")
print("5. WHAT IS ACTUALLY USABLE FOR RAG?")
print("="*60)

usable = []
reasons_excluded = Counter()

for i, ex in enumerate(ds):
    # Check 1: has valid integer answer
    ans = parse_expected_answer(ex.get('expected_answer'))
    if ans is None:
        reasons_excluded['no_valid_integer_answer'] += 1
        continue
    
    # Check 2: has non-empty solution
    sol = ex['messages'][1]['content']
    if len(sol.split()) < 20:
        reasons_excluded['empty_or_tiny_solution'] += 1
        continue
    
    # Check 3: solution has math content
    has_math = any(sym in sol for sym in ['\\', '=', '+', '-', '*'])
    if not has_math:
        reasons_excluded['no_math_content'] += 1
        continue
    
    usable.append({
        'index': i,
        'problem': ex['messages'][0]['content'],
        'solution': sol,
        'answer': ans,
        'source': ex.get('data_source', 'unknown'),
        'metadata': ex.get('metadata', {}),
    })

print(f"Exclusion reasons:")
for reason, count in reasons_excluded.most_common():
    print(f"  {reason:35s}: {count:6d}")
print(f"\nUSABLE FOR RAG: {len(usable):6d} / {len(ds)} ({100*len(usable)/len(ds):.1f}%)")

# ============================================================
# 6. DIFFICULTY DISTRIBUTION IN USABLE SET
# ============================================================
print(f"\n{'='*60}")
print("6. DIFFICULTY OF USABLE PROBLEMS")
print("="*60)

# Use high_no_tool accuracy as difficulty proxy
# Low accuracy = hard problem, high accuracy = easy problem
difficulty_buckets = {
    'very_easy (>0.875)':   0,
    'easy (0.625-0.875)':   0,
    'medium (0.375-0.625)': 0,
    'hard (0.125-0.375)':   0,
    'very_hard (<0.125)':   0,
    'no_metadata':          0,
}

for ex_data in usable:
    meta = ex_data['metadata']
    if not meta or 'reason_high_no_tool' not in meta:
        difficulty_buckets['no_metadata'] += 1
        continue
    acc = meta['reason_high_no_tool'].get('accuracy', 0)
    if acc > 0.875:
        difficulty_buckets['very_easy (>0.875)'] += 1
    elif acc > 0.625:
        difficulty_buckets['easy (0.625-0.875)'] += 1
    elif acc > 0.375:
        difficulty_buckets['medium (0.375-0.625)'] += 1
    elif acc > 0.125:
        difficulty_buckets['hard (0.125-0.375)'] += 1
    else:
        difficulty_buckets['very_hard (<0.125)'] += 1

print(f"Difficulty distribution of usable problems:")
for diff, count in difficulty_buckets.items():
    pct = 100*count/len(usable) if usable else 0
    print(f"  {diff:35s}: {count:6d} ({pct:.1f}%)")

print(f"\nNote: 'high' filter = problems where model got ≥7/8 correct")
print(f"So most of these are EASY problems the model already solves!")
print(f"For RAG: easy problems are useful as EXAMPLES of technique,")
print(f"not as a difficulty benchmark.")

# ============================================================
# 7. SOURCE QUALITY FOR RAG
# ============================================================
print(f"\n{'='*60}")
print("7. BEST SOURCES FOR RAG")
print("="*60)

source_stats = defaultdict(lambda: {'count': 0, 'avg_sol_len': 0, 'sol_lens': []})
for ex_data in usable:
    src = ex_data['source']
    sol_len = len(ex_data['solution'].split())
    source_stats[src]['count'] += 1
    source_stats[src]['sol_lens'].append(sol_len)

print(f"Sources in usable set:")
for src, stats_d in sorted(source_stats.items(), 
                            key=lambda x: -x[1]['count'])[:15]:
    lens = stats_d['sol_lens']
    avg = sum(lens)/len(lens)
    med = sorted(lens)[len(lens)//2]
    print(f"  {src:25s}: {stats_d['count']:5d} problems | "
          f"median_sol={med:4d}w avg_sol={avg:.0f}w")

# ============================================================
# 8. DOMAIN DETECTION ON USABLE SET
# ============================================================
print(f"\n{'='*60}")
print("8. DOMAIN DISTRIBUTION OF USABLE SET")
print("="*60)

DOMAIN_KEYWORDS = {
    'number_theory':  ['prime', 'divisib', 'modulo', 'gcd', 'lcm',
                       'remainder', 'digit', 'coprime', 'factorial'],
    'combinatorics':  ['choose', 'combinat', 'permut', 'arrange',
                       'count', 'ways', 'select', 'committee'],
    'algebra':        ['polynomial', 'equation', 'function', 'inequality',
                       'maximum', 'minimum', 'real number', 'recurrence'],
    'geometry':       ['triangle', 'circle', 'angle', 'polygon', 'area',
                       'perpendicular', 'tangent', 'chord', 'radius'],
    'functional_eq':  ['functional equation', 'f(x+y)', 'f(xy)',
                       'g(x+y)', 'cauchy', 'f:', 'g:'],
}

def classify(text):
    t = text.lower()
    scores = {d: sum(1 for kw in kws if kw in t) 
              for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'other'

domains_usable = [classify(ex['problem']) for ex in usable]
dom_counts = Counter(domains_usable)
print(f"Domain distribution (usable set):")
for d, c in dom_counts.most_common():
    print(f"  {d:20s}: {c:6d} ({100*c/len(usable):.1f}%)")

# ============================================================
# 9. SAVE CLEAN DATASET SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("9. FINAL SUMMARY & NEXT STEPS")
print("="*60)

summary = {
    'total_raw': len(ds),
    'usable_for_rag': len(usable),
    'excluded': dict(reasons_excluded),
    'domain_distribution': dict(dom_counts),
    'source_distribution': {
        src: stats_d['count'] 
        for src, stats_d in source_stats.items()
    },
    'difficulty_distribution': difficulty_buckets,
    'key_insight': (
        "'High' filter = model solved ≥7/8 with high reasoning effort. "
        "These are problems the model CAN solve, making them perfect "
        "as RAG examples showing correct approach."
    ),
    'rag_recommendation': {
        'corpus_size': len(usable),
        'embed_field': 'problem text (not solution)',
        'return_at_query': 'problem + solution (truncate to 400 words)',
        'index_type': 'single FAISS IndexFlatIP with BGE embeddings',
        'filter_before_index': 'solution >= 20 words AND valid integer answer',
    }
}

with open('eda_results3/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Save a sample of usable data for inspection
random.seed(42)
sample = random.sample(usable, min(20, len(usable)))
with open('eda_results3/usable_samples.json', 'w') as f:
    json.dump([{
        'problem': ex['problem'][:300],
        'solution': ex['solution'][:400],
        'answer': ex['answer'],
        'source': ex['source'],
    } for ex in sample], f, indent=2)

print(f"""
FINAL NUMBERS:
  Raw dataset:              {len(ds):6d}
  Empty/bad solutions:      {reasons_excluded['empty_or_tiny_solution']:6d}
  Non-integer answers:      {reasons_excluded['no_valid_integer_answer']:6d}
  No math content:          {reasons_excluded.get('no_math_content',0):6d}
  ─────────────────────────────────
  USABLE FOR RAG:           {len(usable):6d}  ← build your index from these

WHAT 'HIGH' FILTER MEANS:
  These are problems where GPT-OSS (or similar) got ≥7/8 correct
  at high reasoning effort. NOT hard problems — easy-to-medium problems
  the model reliably solves.
  
  FOR RAG THIS IS PERFECT:
  - Model sees similar problem → recognizes technique
  - Gold solution shows the correct approach clearly
  - High accuracy means solutions are reliable

NEXT STEPS:
  1. Run build_rag_index.py to embed {len(usable)} problems
  2. Upload FAISS index to Kaggle (~200-400MB)
  3. Integrate retrieval into inference notebook
  4. Test on IMO benchmark → compare scores

Saved to eda_results3/
""")
