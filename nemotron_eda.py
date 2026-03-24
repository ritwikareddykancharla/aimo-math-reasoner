"""
EDA for:
  1. nvidia/Nemotron-Math-v2 (HuggingFace - full dataset)
  2. ritwikakancharla/nemotron-math-v2-filtered-high (Kaggle - 440K filtered)

Usage:
    # Set your tokens first
    export HF_TOKEN=your_huggingface_token
    export KAGGLE_USERNAME=your_kaggle_username
    export KAGGLE_KEY=your_kaggle_api_key

    python3 nemotron_eda.py
"""

import os, re, json, random
from collections import Counter, defaultdict

os.makedirs("eda_output", exist_ok=True)

# ============================================================
# LOAD DATASETS
# ============================================================

print("="*60)
print("LOADING DATASETS")
print("="*60)

# ── 1. HuggingFace: nvidia/Nemotron-Math-v2 ──────────────────
print("\n[1] Loading nvidia/Nemotron-Math-v2 from HuggingFace...")
hf_ds = None
try:
    from datasets import load_dataset
    hf_ds = load_dataset(
        "nvidia/Nemotron-Math-v2",
        token=os.environ.get("HF_TOKEN"),
        split="train",
    )
    print(f"    Loaded: {len(hf_ds)} examples")
    print(f"    Columns: {hf_ds.column_names}")
except Exception as e:
    print(f"    FAILED: {e}")
    print("    Set HF_TOKEN env var if dataset is gated.")

# ── 2. Kaggle: ritwikakancharla/nemotron-math-v2-filtered-high ─
print("\n[2] Loading ritwikakancharla/nemotron-math-v2-filtered-high from Kaggle...")
kg_ds = None
try:
    import kagglehub
    path = kagglehub.dataset_download(
        "ritwikakancharla/nemotron-math-v2-filtered-high"
    )
    print(f"    Downloaded to: {path}")
    print(f"    Files: {os.listdir(path)}")

    # Auto-detect format
    import glob
    from datasets import load_from_disk

    # Try HuggingFace arrow
    try:
        kg_ds = load_from_disk(path)
        print(f"    Loaded as HF dataset: {len(kg_ds)} examples")
    except:
        pass

    # Try parquet
    if kg_ds is None:
        parquet = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
        if parquet:
            import pandas as pd
            df = pd.concat([pd.read_parquet(f) for f in parquet])
            kg_ds = df.to_dict('records')
            print(f"    Loaded as parquet: {len(kg_ds)} examples")
            print(f"    Columns: {list(df.columns)}")

    # Try jsonl
    if kg_ds is None:
        jsonl = glob.glob(os.path.join(path, "**/*.jsonl"), recursive=True)
        if jsonl:
            kg_ds = []
            for f in jsonl:
                with open(f) as fp:
                    kg_ds.extend(json.loads(l) for l in fp)
            print(f"    Loaded as JSONL: {len(kg_ds)} examples")

except Exception as e:
    print(f"    FAILED: {e}")
    print("    Set KAGGLE_USERNAME and KAGGLE_KEY env vars.")

# Normalize to list
def to_list(ds):
    if ds is None:
        return None
    if hasattr(ds, '__len__') and not isinstance(ds, list):
        return [ds[i] for i in range(len(ds))]
    return ds

hf_list = to_list(hf_ds)
kg_list = to_list(kg_ds)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_problem(ex):
    if 'messages' in ex:
        for m in ex['messages']:
            if isinstance(m, dict) and m.get('role') == 'user':
                return m['content']
        return ex['messages'][0]['content'] if ex['messages'] else ""
    return ex.get('problem', ex.get('question', ''))

def get_solution(ex):
    if 'messages' in ex:
        for m in ex['messages']:
            if isinstance(m, dict) and m.get('role') == 'assistant':
                return m['content']
        return ex['messages'][-1]['content'] if len(ex['messages']) > 1 else ""
    return ex.get('solution', ex.get('answer', ''))

def extract_boxed_int(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    for m in reversed(matches):
        try:
            v = int(float(m.strip().replace(',', '').replace(' ', '')))
            if 0 <= v <= 99999:
                return v
        except:
            pass
    return None

def get_answer(ex):
    # expected_answer field (most reliable)
    for field in ['expected_answer', 'answer', 'label']:
        raw = ex.get(field)
        if raw is not None:
            try:
                v = int(float(str(raw).strip().replace(',', '')))
                if 0 <= v <= 99999:
                    return v, field
            except:
                pass
    # fallback: boxed in solution
    sol = get_solution(ex)
    v = extract_boxed_int(sol)
    if v is not None:
        return v, 'boxed_in_solution'
    return None, None

def classify_domain(text):
    t = text.lower()
    domains = {
        'number_theory':  ['prime', 'divisib', 'modulo', 'gcd', 'lcm', 'remainder',
                           'digit', 'coprime', 'factorial', 'congruent'],
        'combinatorics':  ['choose', 'combinat', 'permut', 'arrange', 'count the',
                           'ways', 'committee', 'distribute', 'coloring', 'tournament'],
        'algebra':        ['polynomial', 'functional equation', 'inequality', 'maximum',
                           'minimum', 'real number', 'recurrence', 'sequence'],
        'geometry':       ['triangle', 'circle', 'angle', 'polygon', 'area',
                           'perpendicular', 'tangent', 'radius', 'inscribed'],
    }
    scores = {d: sum(1 for kw in kws if kw in t) for d, kws in domains.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'other'

def detect_techniques(problem, solution):
    text = (problem + ' ' + solution).lower()
    techniques = {
        'modular_arithmetic':   ['mod', 'modulo', 'congruent', 'remainder'],
        'inclusion_exclusion':  ['inclusion-exclusion', 'inclusion exclusion'],
        'generating_functions': ['generating function', 'power series'],
        'functional_equations': ['functional equation', 'f(x+y)', 'g(x+y)'],
        'linear_recurrences':   ['recurrence', 'a_n =', 'linear recurrence'],
        'pigeonhole':           ['pigeonhole', 'drawer'],
        'invariant':            ['invariant', 'monovariant'],
        'induction':            ['induction', 'base case', 'inductive'],
        'extremal':             ['maximum', 'minimum', 'extremal', 'optimal'],
        'bijection':            ['bijection', 'one-to-one'],
        'vieta_jumping':        ['vieta jumping', 'vieta'],
        'am_gm':                ['am-gm', 'arithmetic mean', 'geometric mean'],
        'cauchy_schwarz':       ['cauchy-schwarz', 'cauchy schwarz'],
        'infinite_descent':     ['infinite descent', 'well-ordering'],
        'graph_theory':         ['graph', 'vertex', 'edge', 'spanning tree'],
        'floor_ceiling':        ['floor', 'ceiling', '\\lfloor', '\\lceil'],
        'complex_numbers':      ['complex number', 'roots of unity', 'argument'],
        'number_theory_adv':    ['lifting the exponent', 'lte', 'zsygmondy',
                                 'primitive root'],
        'polynomial_roots':     ['vieta formulas', 'factor theorem', 'polynomial root'],
    }
    found = []
    for tech, keywords in techniques.items():
        if any(kw in text for kw in keywords):
            found.append(tech)
    return found

# ============================================================
# RUN EDA ON ONE DATASET
# ============================================================

def run_eda(ds_list, name):
    if ds_list is None:
        print(f"\n{'='*60}")
        print(f"SKIPPING {name} — not loaded")
        print("="*60)
        return {}

    print(f"\n{'='*60}")
    print(f"EDA: {name}  ({len(ds_list)} examples)")
    print("="*60)

    # Sample structure
    print(f"\nKeys: {list(ds_list[0].keys())}")

    # Extract all fields
    print("\nExtracting fields...")
    problems    = [get_problem(ex) for ex in ds_list]
    solutions   = [get_solution(ex) for ex in ds_list]
    ans_results = [get_answer(ex) for ex in ds_list]
    answers     = [a for a,_ in ans_results]
    ans_sources = [s for _,s in ans_results]

    prob_lens = [len(p.split()) for p in problems]
    sol_lens  = [len(s.split()) for s in solutions]

    # ── Answers ──────────────────────────────────────────────
    print(f"\n── ANSWERS ──")
    valid_int    = [(i,a) for i,(a,_) in enumerate(ans_results) if a is not None]
    invalid      = [(i,ex) for i,ex in enumerate(ds_list)
                    if answers[i] is None]

    print(f"Valid integer [0,99999]: {len(valid_int):7d} ({100*len(valid_int)/len(ds_list):.1f}%)")
    print(f"No valid integer:        {len(invalid):7d} ({100*len(invalid)/len(ds_list):.1f}%)")
    print(f"Answer sources: {Counter(ans_sources).most_common()}")

    valid_answers = [a for a in answers if a is not None]
    if valid_answers:
        ans_counter = Counter(valid_answers)
        print(f"Unique answer values:    {len(ans_counter)}")
        print(f"Most common:             {ans_counter.most_common(15)}")
        buckets = [(0,0),(1,9),(10,99),(100,999),(1000,9999),(10000,99999)]
        print(f"Answer range distribution:")
        for lo,hi in buckets:
            c = sum(1 for a in valid_answers if lo<=a<=hi)
            print(f"  [{lo:6d},{hi:6d}]: {c:6d} ({100*c/len(valid_answers):.1f}%)")

    # ── Solutions ─────────────────────────────────────────────
    print(f"\n── SOLUTIONS ──")
    empty = sum(1 for l in sol_lens if l == 0)
    tiny  = sum(1 for l in sol_lens if 0 < l < 20)
    short = sum(1 for l in sol_lens if 20 <= l < 100)
    med   = sum(1 for l in sol_lens if 100 <= l < 500)
    long_ = sum(1 for l in sol_lens if l >= 500)

    sl = sorted(sol_lens)
    n  = len(sl)
    print(f"Empty (0w):        {empty:7d} ({100*empty/n:.1f}%)")
    print(f"Tiny  (1-19w):     {tiny:7d} ({100*tiny/n:.1f}%)")
    print(f"Short (20-99w):    {short:7d} ({100*short/n:.1f}%)")
    print(f"Medium(100-499w):  {med:7d} ({100*med/n:.1f}%)")
    print(f"Long  (500+w):     {long_:7d} ({100*long_/n:.1f}%)")
    print(f"p10={sl[n//10]} p25={sl[n//4]} median={sl[n//2]} "
          f"p75={sl[3*n//4]} p90={sl[9*n//10]} max={sl[-1]}")

    # ── Problems ──────────────────────────────────────────────
    print(f"\n── PROBLEMS ──")
    pl = sorted(prob_lens)
    print(f"p10={pl[n//10]} p25={pl[n//4]} median={pl[n//2]} "
          f"p75={pl[3*n//4]} p90={pl[9*n//10]} max={pl[-1]}")
    has_latex = sum(1 for p in problems if '$' in p or '\\' in p)
    print(f"Has LaTeX: {has_latex} ({100*has_latex/n:.1f}%)")

    # ── Domains ───────────────────────────────────────────────
    print(f"\n── DOMAINS ──")
    print("Classifying...")
    domains = [classify_domain(p) for p in problems]
    dom_counts = Counter(domains)
    for d,c in dom_counts.most_common():
        print(f"  {d:20s}: {c:6d} ({100*c/n:.1f}%)")

    # ── Techniques ────────────────────────────────────────────
    print(f"\n── TECHNIQUES ──")
    print("Detecting techniques (slow)...")
    tech_counts = Counter()
    for p, s in zip(problems, solutions):
        for t in detect_techniques(p, s):
            tech_counts[t] += 1
    print("Top techniques:")
    for tech, count in tech_counts.most_common(20):
        print(f"  {tech:30s}: {count:6d} ({100*count/n:.1f}%)")

    # ── Data Sources ──────────────────────────────────────────
    print(f"\n── DATA SOURCES ──")
    sources = [ex.get('data_source', ex.get('source', 'unknown'))
               for ex in ds_list]
    src_counts = Counter(sources)
    print("Top sources:")
    for src, count in src_counts.most_common(20):
        print(f"  {src:35s}: {count:6d} ({100*count/n:.1f}%)")

    # ── Metadata quality scores ───────────────────────────────
    print(f"\n── METADATA / QUALITY SCORES ──")
    has_meta = sum(1 for ex in ds_list if ex.get('metadata'))
    print(f"Has metadata field: {has_meta} ({100*has_meta/n:.1f}%)")
    if has_meta > 0:
        sample_meta = next(ex['metadata'] for ex in ds_list if ex.get('metadata'))
        print(f"Metadata keys: {list(sample_meta.keys()) if isinstance(sample_meta, dict) else type(sample_meta)}")
        # Extract accuracy scores
        acc_keys = ['reason_high_no_tool', 'reason_high_with_tool',
                    'reason_medium_no_tool', 'reason_low_no_tool']
        for key in acc_keys:
            accs = []
            for ex in ds_list:
                meta = ex.get('metadata', {})
                if isinstance(meta, dict) and key in meta:
                    accs.append(meta[key].get('accuracy', 0))
            if accs:
                sa = sorted(accs)
                print(f"  {key}: n={len(accs)} "
                      f"min={sa[0]:.2f} median={sa[len(sa)//2]:.2f} "
                      f"mean={sum(accs)/len(accs):.2f} max={sa[-1]:.2f}")

    # ── Usable for RAG ────────────────────────────────────────
    print(f"\n── USABLE FOR RAG ──")
    usable = sum(1 for i,_ in valid_int if sol_lens[i] >= 20)
    print(f"Valid answer + solution≥20w: {usable:7d} ({100*usable/n:.1f}%)")
    print(f"This is your RAG corpus size.")

    # ── 10 Random samples ─────────────────────────────────────
    print(f"\n── 10 RANDOM SAMPLES ──")
    indices = random.sample(range(n), min(10, n))
    samples = []
    for i in indices:
        print(f"\n  [{i}] answer={answers[i]} domain={domains[i]} "
              f"prob_w={prob_lens[i]} sol_w={sol_lens[i]}")
        print(f"  P: {problems[i][:120]}...")
        print(f"  S: {solutions[i][:150]}...")
        samples.append({
            'index': i,
            'answer': answers[i],
            'domain': domains[i],
            'problem': problems[i][:300],
            'solution': solutions[i][:400],
            'source': sources[i] if sources else 'unknown',
        })

    # ── Save ──────────────────────────────────────────────────
    result = {
        'name': name,
        'total': n,
        'valid_integer_answers': len(valid_int),
        'empty_solutions': empty,
        'usable_for_rag': usable,
        'domain_distribution': dict(dom_counts),
        'technique_counts': dict(tech_counts.most_common(20)),
        'source_distribution': dict(src_counts.most_common(20)),
        'answer_range': {
            f'{lo}-{hi}': sum(1 for a in valid_answers if lo<=a<=hi)
            for lo,hi in [(0,0),(1,9),(10,99),(100,999),(1000,9999),(10000,99999)]
        },
        'solution_length_median': sl[n//2],
        'problem_length_median': pl[n//2],
    }
    safe_name = name.replace('/', '_').replace(' ', '_')
    with open(f"eda_output/{safe_name}_summary.json", 'w') as f:
        json.dump(result, f, indent=2)
    with open(f"eda_output/{safe_name}_samples.json", 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"\nSaved to eda_output/{safe_name}_*")
    return result

# ============================================================
# RUN ON BOTH DATASETS
# ============================================================

random.seed(42)

hf_result = run_eda(hf_list,  "nvidia_Nemotron_Math_v2_HuggingFace")
kg_result = run_eda(kg_list,  "ritwikakancharla_filtered_high_Kaggle")

# ============================================================
# COMPARISON
# ============================================================

if hf_result and kg_result:
    print(f"\n{'='*60}")
    print("COMPARISON: HuggingFace vs Kaggle Filtered")
    print("="*60)
    print(f"{'Metric':35s} {'HF':>10} {'Kaggle':>10}")
    print("-"*57)
    metrics = [
        ('Total examples',         'total'),
        ('Valid integer answers',   'valid_integer_answers'),
        ('Empty solutions',         'empty_solutions'),
        ('Usable for RAG',         'usable_for_rag'),
        ('Median solution words',  'solution_length_median'),
        ('Median problem words',   'problem_length_median'),
    ]
    for label, key in metrics:
        hv = hf_result.get(key, 'N/A')
        kv = kg_result.get(key, 'N/A')
        print(f"  {label:33s} {str(hv):>10} {str(kv):>10}")

    print(f"\nDomain comparison:")
    all_domains = set(list(hf_result.get('domain_distribution',{}).keys()) +
                      list(kg_result.get('domain_distribution',{}).keys()))
    for d in sorted(all_domains):
        hv = hf_result.get('domain_distribution',{}).get(d, 0)
        kv = kg_result.get('domain_distribution',{}).get(d, 0)
        hn = hf_result.get('total', 1)
        kn = kg_result.get('total', 1)
        print(f"  {d:20s}: HF={hv:6d}({100*hv/hn:.1f}%)  "
              f"Kaggle={kv:6d}({100*kv/kn:.1f}%)")

    print(f"\nKey insight for RAG:")
    print(f"  HF total:            {hf_result.get('total', 'N/A')}")
    print(f"  Kaggle total:        {kg_result.get('total', 'N/A')}")
    print(f"  HF usable for RAG:   {hf_result.get('usable_for_rag', 'N/A')}")
    print(f"  Kaggle usable RAG:   {kg_result.get('usable_for_rag', 'N/A')}")
    print(f"  → Use whichever gives more usable examples for RAG index")

print(f"\nAll outputs saved to eda_output/")
print("Done!")
