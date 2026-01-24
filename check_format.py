
from datasets import load_from_disk
import re, sys

path = "./local_luogu_dataset"
try:
    ds = load_from_disk(path)
except Exception as e:
    print(f"Failed to load dataset at {path}:", e)
    sys.exit(1)

# use train split if available
if hasattr(ds, 'keys') and 'train' in ds:
    ds = ds['train']

print(f"Dataset: {path}, examples: {len(ds)}, columns: {ds.column_names}")

pat = re.compile(r'```(?:\s*(?:cpp|c\+\+|cxx))?\s*\n', re.I)
for i in range(min(10, len(ds))):
    row = ds[i]
    text = row.get('completion', '')
    valid = row.get('valid', None)
    has_fence = bool(pat.search(text))
    has_feat = any(k in text for k in ('#include', 'int main', 'using namespace std'))
    print(f"{i}: valid={valid} fence={has_fence} feat={has_feat}")
    print(text[:2000].replace("\n","\\n"))
    print("---")
