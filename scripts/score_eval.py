# -*- coding: utf-8 -*-
"""Score multihop eval results against expected fields in multihop_test.json."""
"""Score eval_multihop results: PASS/FAIL + per-type aggregates.

Usage:
    python scripts/eval_multihop.py --config baseline --llm lmstudio
    python scripts/score_eval.py [results.json]
"""
import json
import os
import re
import sys

import os, sys
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "mirea_data", "eval", "multihop_baseline_lmstudio.json")
ts_path = os.path.join(PROJ, "mirea_data", "eval", "multihop_test.json")
if not os.path.exists(ts_path):
    ts_path = os.path.join(PROJ, "eval", "multihop_test.json")
mt = json.load(open(ts_path, encoding="utf-8"))
res = json.load(open(res_path, encoding="utf-8"))

answers = {r["id"]: r for r in res}


def norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    # унифицируем неразрывные пробелы и пробелы в числах: "1 000" == "1000"
    s = re.sub(r"(\d)\s+(\d{3})", r"\1\2", s)
    return s


REFUSAL = ["нет информации", "не уверен", "не удалось сформировать", "уточните вопрос", "нет данных"]

rows, by_type = [], {}
for q in mt["questions"]:
    qid = q["id"]
    r = answers.get(qid)
    if r is None:
        rows.append((qid, q.get("type", "?"), "MISSING", "нет результата прогона", q["question"][:60]))
        continue
    ans = norm(r.get("answer", ""))
    if q.get("expected_refusal"):
        ok = any(norm(m) in ans for m in REFUSAL)
        detail = "отказ" if ok else f'ОТВЕТ: {r["answer"][:120]}'
    else:
        missing_all = [e for e in q.get("expected_all", []) if norm(str(e)) not in ans]
        ok = not missing_all
        if ok and q.get("expected_any"):
            ok = any(norm(str(e)) in ans for e in q["expected_any"])
        detail = "ok" if ok else f'нет: {missing_all or q.get("expected_any")}'
    rows.append((qid, q.get("type", "?"), "PASS" if ok else "FAIL", detail, q["question"][:60]))
    t = q.get("type", "?")
    p, tot = by_type.get(t, (0, 0))
    by_type[t] = (p + (1 if ok else 0), tot + 1)

n_pass = sum(1 for r in rows if r[2] == "PASS")
hard = [r for r in rows if r[0].startswith("hd")]
hard_pass = sum(1 for r in hard if r[2] == "PASS")
old = [r for r in rows if not r[0].startswith("hd")]
old_pass = sum(1 for r in old if r[2] == "PASS")

for r in rows:
    mark = "✅" if r[2] == "PASS" else "❌"
    print(f'{mark} {r[0]:<10} [{r[1]:<18}] {r[3][:90]}')
    if r[2] != "PASS":
        print(f'      Q: {r[4]}')

print("\n=== ИТОГ ===")
print(f"Всего: {n_pass}/{len(rows)} ({100*n_pass/len(rows):.0f}%)")
print(f"Старый набор: {old_pass}/{len(old)}")
print(f"Hard set (hd###): {hard_pass}/{len(hard)}")
print("\nПо типам:")
for t, (p, tot) in sorted(by_type.items()):
    print(f"  {t:<20} {p}/{tot}")
sys.exit(0)
