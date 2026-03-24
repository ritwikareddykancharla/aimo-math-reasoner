"""
Check what's actually in the 'no boxed answer' solutions
and find ALL valid integer answers for RAG
"""

import re, json
from collections import Counter
from datasets import load_from_disk
import os

DATA_DIR = os.environ.get("DATA_DIR", "./data/nemotron-math-v2")
ds = load_from_disk(DATA_DIR)
print(f"Total: {len(ds)}")

def extract_integer_answer(text):
    """Try multiple patterns to extract integer answer 0-99999."""
    
    # Pattern 1: \boxed{N}
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        for m in reversed(matches):  # take last boxed
            try:
                v = int(float(m.strip().replace(',','')))
                if 0 <= v <= 99999:
                    return v, 'boxed'
            except:
                pass
    
    # Pattern 2: "the answer is N" or "answer: N"
    for pat in [
        r'(?:the\s+)?answer\s+is\s+(\d+)',
        r'answer:\s*(\d+)',
        r'= (\d+)\s*$',
        r'equals?\s+(\d+)',
        r'is\s+(\d+)\s*\.',
    ]:
        matches = re.findall(pat, text.lower())
        if matches:
            try:
                v = int(matches[-1])
                if 0 <= v <= 99999:
                    return v, 'text_pattern'
            except:
                pass
    
    # Pattern 3: last standalone integer in solution
    # Look for integers at end of solution
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):  # check last 5 lines
        nums = re.findall(r'\b(\d{1,5})\b', line)
        if nums:
            try:
                v = int(nums[-1])
                if 0 <= v <= 99999:
                    return v, 'last_int'
            except:
                pass
    
    return None, None

# ============================================================
# Analyze all solutions
# ============================================================

results = {
    'boxed_integer': 0,
    'text_pattern': 0,
    'last_int': 0,
    'no_integer': 0,
}

valid_for_rag = []
source_counts = Counter()

print("Scanning all solutions for integer answers...")
for i, ex in enumerate(ds):
    if i % 10000 == 0:
        print(f"  {i}/{len(ds)}...")
    
    sol = ex['messages'][1]['content']
    prob = ex['messages'][0]['content']
    
    val, source = extract_integer_answer(sol)
    
    if val is not None:
        source_counts[source] += 1
        valid_for_rag.append({
            'problem': prob,
            'solution': sol,
            'answer': val,
            'answer_source': source,
        })
    else:
        results['no_integer'] += 1

print(f"\nResults:")
print(f"  Total valid integer answers found: {len(valid_for_rag)}")
print(f"  By source:")
for src, count in source_counts.most_common():
    print(f"    {src}: {count}")
print(f"  No integer found: {results['no_integer']}")

# ============================================================
# Check answer distribution for non-boxed sources
# ============================================================

text_pattern_answers = [ex['answer'] for ex in valid_for_rag 
                        if ex['answer_source'] == 'text_pattern']
last_int_answers = [ex['answer'] for ex in valid_for_rag 
                    if ex['answer_source'] == 'last_int']

print(f"\nText pattern answers distribution (top 20):")
print(Counter(text_pattern_answers).most_common(20))

print(f"\nLast int answers distribution (top 20):")
print(Counter(last_int_answers).most_common(20))

# Last int is noisy - check quality
print(f"\nSample 'last_int' sources (likely noisy):")
last_int_examples = [ex for ex in valid_for_rag if ex['answer_source'] == 'last_int']
for ex in last_int_examples[:3]:
    print(f"\n  Problem: {ex['problem'][:100]}...")
    print(f"  Solution end: ...{ex['solution'][-200:]}")
    print(f"  Answer: {ex['answer']}")

# ============================================================
# Recommendation: what to use for RAG
# ============================================================

boxed_only = [ex for ex in valid_for_rag if ex['answer_source'] == 'boxed_integer']
boxed_and_text = [ex for ex in valid_for_rag 
                  if ex['answer_source'] in ('boxed_integer', 'text_pattern')]

print(f"\n{'='*50}")
print(f"RAG DATASET OPTIONS:")
print(f"  Conservative (boxed only):           {len(boxed_only):6d}")
print(f"  Medium (boxed + text pattern):       {len(boxed_and_text):6d}")
print(f"  Aggressive (all with any integer):   {len(valid_for_rag):6d}")
print(f"  Total dataset:                       {len(ds):6d}")
print(f"\nRecommendation: use 'boxed only' ({len(boxed_only)}) for high quality RAG")
print(f"The 'last_int' source is likely too noisy (picks up random numbers)")

# Save valid dataset info
with open('eda_results/valid_for_rag.json', 'w') as f:
    json.dump({
        'total': len(ds),
        'boxed_integer': len(boxed_only),
        'text_pattern': len([ex for ex in valid_for_rag if ex['answer_source']=='text_pattern']),
        'last_int': len(last_int_examples),
        'no_integer': results['no_integer'],
    }, f, indent=2)

print(f"\nSaved to eda_results/valid_for_rag.json")
