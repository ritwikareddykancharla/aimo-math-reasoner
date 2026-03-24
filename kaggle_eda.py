"""
EDA for nemotron-math-v2-filtered-high (440K Kaggle dataset)
=============================================================
Downloads from Kaggle via kagglehub then runs full EDA.

Usage:
    pip install kagglehub datasets
    python3 kaggle_eda.py

Requires Kaggle credentials:
    ~/.kaggle/kaggle.json  OR  KAGGLE_USERNAME + KAGGLE_KEY env vars
"""

import os, re, json, random
from collections import Counter, defaultdict

# ============================================================
# STEP 1: DOWNLOAD FROM KAGGLE
# ============================================================

import kagglehub

print("Downloading from Kaggle...")
DATA_DIR = kagglehub.dataset_download(
    "ritwikakancharla/nemotron-math-v2-filtered-high"
)
print(f"Downloaded to: {DATA_DIR}")
print(f"\nFiles in dataset:")
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        fpath = os.path.join(root, f)
        print(f"  {fpath}  ({os.path.getsize(fpath)/1e6:.1f}MB)")

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

from datasets import load_from_disk
import glob

print("\nLoading dataset...")

# Try direct load first
try:
    ds = load_from_disk(DATA_DIR)
    print(f"Loaded directly: {len(ds)} examples")
except:
    # Find subdirectory with dataset
    subdirs = [d for d in glob.glob(os.path.join(DATA_DIR, "*")) 
               if os.path.isdir(d)]
    for subdir in subdirs:
        try:
            ds = load_from_disk(subdir)
            print(f"Loaded from {subdir}: {len(ds)} examples")
            break
        except:
            continue

print(f"\nTotal examples: {len(ds)}")

# ============================================================
# STEP 3: INSPECT STRUCTURE
# ============================================================

print("\n" + "="*60)
print("1. DATASET STRUCTURE")
print("="*60)

sample = ds[0]
print(f"All keys: {list(sample.keys())}")
print(f"\nFull sample record:")
for k, v in sample.items():
    print(f"\n  [{k}]:")
    print(f"    {str(v)[:400]}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_problem(ex):
    if 'messages' in ex:
        for m in ex['messages']:
            if m.get('role') == 'user':
                return m['content']
        return ex['messages'][0]['content']
    return ex.get('problem', '')

def get_solution(ex):
    if 'messages' in ex:
        for m in ex['messages']:
            if m.get('role') == 'assistant':
                return m['content']
        return ex['messages'][-1]['content'] if len(ex['messages']) > 1 else ''
    return ex.get('solution', '')

def parse_int(val):
    try:
        v = int(float(str(val).strip().replace(',', '')))
        return v if 0 <= v <= 99999 else None
    except:
        return None

def get_answer(ex):
    # Priority 1: expected_answer field
    if 'expected_answer' in ex:
        v = parse_int(ex['expected_answer'])
        if v is not None:
            return v, 'expected_answer'
    # Priority 2: boxed in solution
    sol = get_solution(ex)
    matches = re.findall(r'\\boxed\{([^}]+)\}', sol)
    if matches:
        v = parse_int(matches[-1])
        if v is not None:
            return v, 'boxed'
    return None, None

# ============================================================
# 4. ANSWER ANALYSIS
# ============================================================

print("\n" + "="*60)
print("2. ANSWER ANALYSIS")
print("="*60)

print("Extracting answers (may take 1-2 min for 440K)...")
answers, answer_sources = [], []
for ex in ds:
    v, src = get_answer(ex)
    answers.append(v)
    answer_sources.append(src)

valid   = [(i, a) for i, a in enumerate(answers) if a is not None]
invalid = [(i, a) for i, a in enumerate(answers) if a is None]

print(f"Valid integer [0,99999]:  {len(valid):7d} ({100*len(valid)/len(ds):.1f}%)")
print(f"No valid integer:         {len(invalid):7d} ({100*len(invalid)/len(ds):.1f}%)")

src_counts = Counter(answer_sources)
print(f"\nAnswer extracted from:")
for src, count in src_counts.most_common():
    print(f"  {str(src):25s}: {count}")

all_valid_answers = [a for _, a in valid]
ans_counter = Counter(all_valid_answers)
print(f"\nUnique answer values:    {len(ans_counter)}")
print(f"Top 20 most common:      {ans_counter.most_common(20)}")

print(f"\nAnswer range distribution:")
for lo, hi in [(0,0),(1,9),(10,99),(100,999),(1000,9999),(10000,99999)]:
    c = sum(1 for a in all_valid_answers if lo <= a <= hi)
    print(f"  [{lo:6d},{hi:6d}]: {c:7d} ({100*c/max(len(all_valid_answers),1):.1f}%)")

print(f"\nSample invalid expected_answers:")
for i, _ in invalid[:8]:
    ea = ds[i].get('expected_answer', 'N/A')
    print(f"  '{ea}' | {get_problem(ds[i])[:80]}...")

# ============================================================
# 5. SOLUTION QUALITY
# ============================================================

print("\n" + "="*60)
print("3. SOLUTION QUALITY")
print("="*60)

print("Computing solution lengths...")
sol_lens = [len(get_solution(ex).split()) for ex in ds]
ss = sorted(sol_lens)
n  = len(ss)

print(f"Solution length (words):")
print(f"  min={ss[0]}  p10={ss[n//10]}  p25={ss[n//4]}  "
      f"median={ss[n//2]}  p75={ss[3*n//4]}  p90={ss[9*n//10]}  max={ss[-1]}")

empty  = sum(1 for l in sol_lens if l == 0)
tiny   = sum(1 for l in sol_lens if 0 < l < 20)
short  = sum(1 for l in sol_lens if 20 <= l < 100)
medium = sum(1 for l in sol_lens if 100 <= l < 500)
long_  = sum(1 for l in sol_lens if l >= 500)

print(f"\nLength buckets:")
print(f"  Empty (0 words):        {empty:7d} ({100*empty/n:.1f}%)")
print(f"  Tiny (1-19 words):      {tiny:7d} ({100*tiny/n:.1f}%)")
print(f"  Short (20-99 words):    {short:7d} ({100*short/n:.1f}%)")
print(f"  Medium (100-499 words): {medium:7d} ({100*medium/n:.1f}%)")
print(f"  Long (500+ words):      {long_:7d} ({100*long_/n:.1f}%)")

# ============================================================
# 6. METADATA ANALYSIS
# ============================================================

print("\n" + "="*60)
print("4. METADATA / QUALITY SCORES")
print("="*60)

has_meta = sum(1 for ex in ds if ex.get('metadata'))
print(f"Has metadata: {has_meta} ({100*has_meta/len(ds):.1f}%)")

if has_meta:
    meta_sample = next(ex['metadata'] for ex in ds if ex.get('metadata'))
    print(f"Metadata keys: {list(meta_sample.keys())}")

    for key in meta_sample.keys():
        accs = []
        for ex in ds:
            m = ex.get('metadata') or {}
            if key in m and isinstance(m[key], dict):
                acc = m[key].get('accuracy')
                if acc is not None:
                    accs.append(acc)
        if accs:
            accs_s = sorted(accs)
            m2 = len(accs_s)
            dist = Counter(round(a, 2) for a in accs)
            print(f"\n  {key}:")
            print(f"    n={m2}  min={accs_s[0]:.2f}  "
                  f"median={accs_s[m2//2]:.2f}  "
                  f"mean={sum(accs)/m2:.2f}  max={accs_s[-1]:.2f}")
            print(f"    dist: {dict(sorted(dist.items()))}")

# ============================================================
# 7. DATA SOURCE
# ============================================================

print("\n" + "="*60)
print("5. DATA SOURCES")
print("="*60)

sources = [ex.get('data_source', 'unknown') for ex in ds]
sc = Counter(sources)
print(f"Unique sources: {len(sc)}")
for src, count in sc.most_common(20):
    print(f"  {str(src):35s}: {count:7d} ({100*count/len(ds):.1f}%)")

# ============================================================
# 8. DOMAIN DISTRIBUTION
# ============================================================

print("\n" + "="*60)
print("6. DOMAIN DISTRIBUTION")
print("="*60)

DOMAINS = {
    'number_theory':     ['prime', 'divisib', 'modulo', 'gcd', 'lcm',
                          'remainder', 'digit', 'coprime', 'factorial'],
    'combinatorics':     ['choose', 'combinat', 'permut', 'arrange',
                          'count', 'ways', 'select', 'committee'],
    'algebra':           ['polynomial', 'equation', 'function', 'inequality',
                          'maximum', 'minimum', 'recurrence', 'sequence'],
    'geometry':          ['triangle', 'circle', 'angle', 'polygon', 'area',
                          'perpendicular', 'tangent', 'chord', 'radius'],
    'functional_eq':     ['functional equation', 'f(x+y)', 'g(x+y)', 'cauchy'],
}

def classify(text):
    t = text.lower()
    scores = {d: sum(1 for kw in kws if kw in t) for d, kws in DOMAINS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'other'

print("Classifying domains...")
domains = [classify(get_problem(ex)) for ex in ds]
dom_counts = Counter(domains)
for d, c in dom_counts.most_common():
    print(f"  {d:20s}: {c:7d} ({100*c/len(ds):.1f}%)")

# ============================================================
# 9. USABLE FOR RAG
# ============================================================

print("\n" + "="*60)
print("7. USABLE FOR RAG")
print("="*60)

usable_idx = []
excluded = Counter()
for i, ex in enumerate(ds):
    v, _ = get_answer(ex)
    if v is None:
        excluded['no_valid_answer'] += 1
        continue
    if sol_lens[i] < 20:
        excluded['empty_or_tiny_solution'] += 1
        continue
    usable_idx.append(i)

print(f"Exclusion reasons:")
for r, c in excluded.most_common():
    print(f"  {r:35s}: {c:7d}")
print(f"\nUSABLE FOR RAG: {len(usable_idx):7d} / {len(ds)} ({100*len(usable_idx)/len(ds):.1f}%)")

# ============================================================
# 10. SPOT CHECK
# ============================================================

print("\n" + "="*60)
print("8. RANDOM SAMPLE SPOT CHECK")
print("="*60)

random.seed(42)
sample_idxs = random.sample(usable_idx, min(10, len(usable_idx)))
samples_out = []
for i in sample_idxs:
    ex = ds[i]
    v, src = get_answer(ex)
    prob = get_problem(ex)
    sol  = get_solution(ex)
    print(f"\n[{i}] answer={v} source={src} sol_words={sol_lens[i]}")
    print(f"  PROBLEM:  {prob[:150]}...")
    print(f"  SOLUTION: {sol[:200]}...")
    samples_out.append({
        'index': i, 'answer': v,
        'data_source': ex.get('data_source'),
        'problem': prob[:300],
        'solution': sol[:400],
    })

# ============================================================
# 11. SAVE RESULTS
# ============================================================

os.makedirs("eda_kaggle", exist_ok=True)
summary = {
    'total': len(ds),
    'valid_answers': len(valid),
    'no_valid_answer': len(invalid),
    'usable_for_rag': len(usable_idx),
    'empty_solutions': empty,
    'domain_distribution': dict(dom_counts),
    'source_distribution': dict(sc.most_common(20)),
    'answer_top30': dict(ans_counter.most_common(30)),
    'median_solution_words': ss[n//2],
}
with open('eda_kaggle/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
with open('eda_kaggle/samples.json', 'w') as f:
    json.dump(samples_out, f, indent=2)

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Total:                {len(ds):7d}")
print(f"Valid integer answer: {len(valid):7d} ({100*len(valid)/len(ds):.1f}%)")
print(f"Usable for RAG:       {len(usable_idx):7d} ({100*len(usable_idx)/len(ds):.1f}%)")
print(f"Empty solutions:      {empty:7d} ({100*empty/len(ds):.1f}%)")
print(f"Median sol length:    {ss[n//2]} words")
print(f"\nSaved to eda_kaggle/")
