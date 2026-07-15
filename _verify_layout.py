"""Quick verification of keyboard layout conversion examples."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.query_normalization import _convert_word, _looks_like_wrong_layout, fix_keyboard_layout

examples = ['rjq', 'vjq', 'afqk', 'rjhgec', 'hello', 'rrr', 'lms', 'LMS', 'e']
for w in examples:
    print(f"{w!r:12} -> {_convert_word(w)!r:12} | detect={_looks_like_wrong_layout(w)}")

print()
for q in ['rjq afqk', 'Где rjhgec?', 'Где rjhgec', 'Как зайти в LMS?']:
    print(f"{q!r:30} -> {fix_keyboard_layout(q)!r}")
