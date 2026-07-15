"""Prompt-engineering A/B test (task t_b2d436d5).

Compares 3 system-prompt variants for the RAG answer generator against the
50-question golden set, judged by glm-5.2-chat via the existing LLM-as-judge
harness (scripts/eval_llm_judge.py).

Variants:
  A — baseline (current production prompt: "Отвечай кратко, по существу")
  B — completeness + anti-refusal (extract ALL facts; don't refuse on partial info)
  C — B + anti-vagueness (cite specific values; no ranges/site-referrals)
  D — B + calibrated formatting (lists for list-questions; prefer exact values;
      preserve ranges as-is instead of inventing precise figures). Addresses the
      correctness regression that rejected C: C forbade ranges outright, which
      pushed the model to hallucinate exact numbers. D keeps the list-format
      directive (which helped q008: A=2.0 → C=3.25) but lets ranges stand.

Each variant is generated fresh through the full RAG pipeline (E5 → Qdrant →
context → LM Studio), then judged on the same 4 dimensions.  Results are saved
per-variant for resumability, then aggregated into a comparison report.

Outputs:
  mirea_data/eval/ab_answers_{variant}.json   — cached answers per variant
  mirea_data/eval/ab_report.json              — comparison + per-query scores

Usage:
  python scripts/eval_prompt_ab.py                    # run all variants
  python scripts/eval_prompt_ab.py --variants B C     # run only B and C
  python scripts/eval_prompt_ab.py --variants A --regenerate   # re-run A
  python scripts/eval_prompt_ab.py --skip-gen          # judge from cache only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import config  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_ab")

GOLDEN_PATH = PROJ / "mirea_data/eval/golden_qa.json"
EVAL_DIR = PROJ / "mirea_data/eval"
REPORT_PATH = EVAL_DIR / "ab_report.json"

GEN_MODEL = "google/gemma-4-26b-a4b-qat"

# ── Prompt variants ──────────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {
    "A": (
        "Ты — справочный бот университета РТУ МИРЭА. "
        "Отвечай на вопросы абитуриентов и студентов на русском языке. "
        "Используй ТОЛЬКО информацию из предоставленного контекста. "
        "Если в контексте нет ответа на вопрос, честно скажи: "
        "\"У меня нет информации по этому вопросу.\" "
        "Не выдумывай факты, цифры, даты или названия. "
        "Отвечай кратко, по существу, дружелюбным тоном."
    ),
    "B": (
        "Ты — справочный бот университета РТУ МИРЭА. "
        "Отвечай на вопросы абитуриентов и студентов на русском языке.\n\n"
        "Используй ТОЛЬКО информацию из предоставленного контекста. "
        "Не выдумывай факты, цифры, даты или названия.\n\n"
        "Внимательно изучи контекст и извлеки из него ВСЕ факты, относящиеся "
        "к вопросу: конкретные цифры, даты, суммы, названия, коды направлений. "
        "Если в контексте есть релевантная информация — обязательно используй "
        "её в ответе.\n\n"
        "Говори «У меня нет информации по этому вопросу» ТОЛЬКО если в "
        "контексте действительно нет ничего, относящегося к вопросу. "
        "Не отказывайся от ответа, если в контексте есть хотя бы частичная "
        "информация.\n\n"
        "Отвечай по существу, дружелюбным тоном."
    ),
    "C": (
        "Ты — справочный бот университета РТУ МИРЭА. "
        "Отвечай на вопросы абитуриентов и студентов на русском языке.\n\n"
        "Используй ТОЛЬКО информацию из предоставленного контекста. "
        "Не выдумывай факты, цифры, даты или названия.\n\n"
        "Внимательно изучи контекст и извлеки из него ВСЕ факты, относящиеся к вопросу. "
        "Приводи конкретные значения из контекста: точные цифры, даты, суммы, "
        "коды направлений. Не используй общие фразы, приблизительные диапазоны "
        "или отсылки на сайт, если в контексте есть конкретное значение.\n\n"
        "Если вопрос предполагает список (экзамены, направления, контакты), "
        "перечисли все пункты из контекста.\n\n"
        "Говори «У меня нет информации по этому вопросу» ТОЛЬКО если в "
        "контексте действительно нет ничего, относящегося к вопросу.\n\n"
        "Отвечай по существу, дружелюбным тоном."
    ),
    # D — calibrated formatting (task t_8dd0b704). Keeps B's completeness/anti-
    # refusal base, adds C's list-format directive (helped q008), but replaces
    # C's "no ranges" rule — which caused hallucinated exact values — with an
    # explicit "preserve ranges as-is" guard. Prefer specifics when present, but
    # never invent a precise figure to replace a range that the context gives.
    "D": (
        "Ты – справочный бот университета РТУ МИРЭА. "
        "Отвечай на вопросы абитуриентов и студентов на русском языке.\n\n"
        "Используй ТОЛЬКО информацию из предоставленного контекста. "
        "Не выдумывай факты, цифры, даты или названия.\n\n"
        "Внимательно изучи контекст и извлеки из него ВСЕ факты, относящиеся "
        "к вопросу: конкретные цифры, даты, суммы, названия, коды направлений. "
        "Если в контексте есть релевантная информация – обязательно используй "
        "её в ответе.\n\n"
        "Если вопрос предполагает перечисление нескольких пунктов (экзамены, "
        "направления подготовки, требования, контакты, документы), оформляй "
        "ответ структурированным списком – по одному пункту на элемент. "
        "Это делает ответ удобным для чтения.\n\n"
        "Приводи конкретные значения из контекста (цифры, даты, суммы, коды), "
        "когда они там есть. Если в контексте указан диапазон – приводи его "
        "как есть, не заменяя диапазон выдуманной точной цифрой.\n\n"
        "Говори «У меня нет информации по этому вопросу» ТОЛЬКО если в "
        "контексте действительно нет ничего, относящегося к вопросу. "
        "Не отказывайся от ответа, если в контексте есть хотя бы частичная "
        "информация.\n\n"
        "Отвечай по существу, дружелюбным тоном."
    ),
}

# ── Judge integration (reuse eval_llm_judge infrastructure) ───────────────────

# We import the judge lazily so the E5 model isn't loaded until we actually
# generate.  The judge functions are pure HTTP — no model loading.

DIMENSIONS = ["correctness", "completeness", "relevance", "fluency"]


def _generate_variant(variant: str, items: list[dict], cache_path: Path,
                      regenerate: bool) -> list[dict]:
    """Generate answers for one prompt variant through the full RAG pipeline."""
    import src.rag.answer as answer_mod

    # Monkeypatch the system prompt
    answer_mod.SYSTEM_PROMPT = PROMPTS[variant]

    # Load cache (keyed by question id within this variant)
    cache: dict[str, dict] = {}
    if cache_path.exists() and not regenerate:
        try:
            cache = {r["id"]: r for r in json.loads(cache_path.read_text("utf-8"))["answers"]}
        except Exception:
            log.warning("cache load failed for variant %s, starting fresh", variant)

    config.LM_STUDIO_CONFIG["model"] = GEN_MODEL
    from src.rag.answer import answer_question  # loads E5 on first call

    results: list[dict] = []
    n_cached = 0
    t0 = time.time()
    for i, item in enumerate(items):
        qid = item["id"]
        if qid in cache:
            results.append(cache[qid])
            n_cached += 1
            continue
        print(f"    [{i+1}/{len(items)}] {variant} generating {qid} ...", flush=True)
        try:
            res = answer_question(item["question"])
            gen = res.get("answer", "")
        except Exception as e:
            log.error("gen failed %s/%s: %s", variant, qid, e)
            gen = f"GENERATION_ERROR: {e!r}"
        row = {
            "id": qid,
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "generated_answer": gen,
            "sources": res.get("sources", []) if isinstance(res, dict) else [],
            "context_chunks": res.get("context_chunks", 0) if isinstance(res, dict) else 0,
            "retrieval_scores": res.get("retrieval_scores", []) if isinstance(res, dict) else [],
            "generation_time": res.get("generation_time", 0) if isinstance(res, dict) else 0,
            "category": item.get("category", ""),
            "difficulty": item.get("difficulty", ""),
        }
        results.append(row)
        cache[qid] = row
        _write_cache(cache_path, variant, results)
    elapsed = time.time() - t0
    print(f"    {variant} gen: {len(results) - n_cached} new, {n_cached} cached, {elapsed:.0f}s")
    return results


def _write_cache(path: Path, variant: str, answers: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"variant": variant, "gen_model": GEN_MODEL,
             "system_prompt": PROMPTS[variant],
             "generated_at": datetime.now(timezone.utc).isoformat(),
             "answers": answers},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def _judge_variant(answers: list[dict], variant: str) -> list[dict]:
    """Judge all answers for a variant using glm-5.2."""
    from scripts.eval_llm_judge import judge_answer

    rows: list[dict] = []
    t0 = time.time()
    for i, ans in enumerate(answers):
        print(f"    [{i+1}/{len(answers)}] {variant} judging {ans['id']} ...", flush=True)
        scores = judge_answer(
            ans["question"], ans["expected_answer"], ans["generated_answer"]
        )
        rows.append({
            "id": ans["id"],
            "variant": variant,
            "question": ans["question"],
            "category": ans.get("category", ""),
            "difficulty": ans.get("difficulty", ""),
            "expected_answer": ans["expected_answer"],
            "generated_answer": ans["generated_answer"],
            "context_chunks": ans.get("context_chunks", 0),
            **scores,
        })
        if (i + 1) % 10 == 0:
            print(f"      ({i+1}/{len(answers)} done, {time.time()-t0:.0f}s)")
    print(f"    {variant} judged in {time.time()-t0:.0f}s")
    return rows


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    from scripts.eval_llm_judge import aggregate_scores, group_aggregate, score_distribution
    overall = aggregate_scores(rows)
    by_cat = group_aggregate(rows, "category")
    by_diff = group_aggregate(rows, "difficulty")
    dist = score_distribution(rows)
    return {"overall": overall, "by_category": by_cat,
            "by_difficulty": by_diff, "score_distribution": dist}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"],
                    choices=list(PROMPTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--skip-gen", action="store_true",
                    help="judge from cached answers only")
    ap.add_argument("--report", default=None,
                    help="output report path (default: ab_report.json). Use a "
                         "custom path to avoid clobbering an existing report.")
    args = ap.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text("utf-8"))
    items = golden["items"]
    if args.limit:
        items = items[: args.limit]
    n = len(items)
    print(f"golden: {n} items | variants: {args.variants} | model: {GEN_MODEL}")
    print(f"skip-gen: {args.skip_gen} | regenerate: {args.regenerate}")

    all_judged: dict[str, list[dict]] = {}
    for v in args.variants:
        print(f"\n{'='*60}")
        print(f"VARIANT {v}")
        print(f"{'='*60}")
        print(f"prompt ({len(PROMPTS[v])} chars):")
        print(f"  {PROMPTS[v][:120]}...")

        cache_path = EVAL_DIR / f"ab_answers_{v}.json"

        if args.skip_gen:
            if not cache_path.exists():
                print(f"  ERROR: --skip-gen but no cache for variant {v}")
                return 2
            answers = json.loads(cache_path.read_text("utf-8"))["answers"]
            print(f"  loaded {len(answers)} cached answers ({v})")
        else:
            print(f"\n  --- Generating (variant {v}) ---")
            answers = _generate_variant(v, items, cache_path, args.regenerate)

        print(f"\n  --- Judging (variant {v}) ---")
        all_judged[v] = _judge_variant(answers, v)

    # ── Comparison report ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    comparison: dict = {}
    for v in args.variants:
        agg = aggregate(all_judged[v])
        comparison[v] = agg
        o = agg["overall"]
        print(f"\n  Variant {v}: overall={o.get('mean_overall',0):.2f}/5 "
              f"({o.get('mean_overall_100',0):.1f}/100)  n={o.get('n',0)}")
        for d in DIMENSIONS:
            print(f"    {d:<13}: {o.get(f'mean_{d}',0):.2f}")

    # Per-question deltas (variant vs variant A if present, else first variant)
    base = "A" if "A" in all_judged else args.variants[0]
    per_q_deltas: list[dict] = []
    if len(args.variants) > 1 and base in all_judged:
        base_map = {r["id"]: r for r in all_judged[base]}
        for v in args.variants:
            if v == base:
                continue
            v_map = {r["id"]: r for r in all_judged[v]}
            for qid, br in base_map.items():
                vr = v_map.get(qid)
                if not vr:
                    continue
                bo = sum(br.get(d, 0) for d in DIMENSIONS) / 4
                vo = sum(vr.get(d, 0) for d in DIMENSIONS) / 4
                if abs(vo - bo) >= 0.5:  # only meaningful deltas
                    per_q_deltas.append({
                        "id": qid, "question": br["question"],
                        "category": br.get("category", ""),
                        "base": base, "base_overall": round(bo, 2),
                        "variant": v, "variant_overall": round(vo, 2),
                        "delta": round(vo - bo, 2),
                    })
        per_q_deltas.sort(key=lambda x: x["delta"])

    # Pick winner
    scores = {v: comparison[v]["overall"].get("mean_overall_100", 0)
              for v in args.variants}
    winner = max(scores, key=scores.get)

    report = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "gen_model": GEN_MODEL,
            "judge_model": "glm-5.2-chat",
            "n_items": n,
            "variants_run": args.variants,
        },
        "prompts": {v: PROMPTS[v] for v in args.variants},
        "comparison": comparison,
        "winner": winner,
        "winner_score": scores[winner],
        "per_query_deltas": per_q_deltas[:20],
        "per_query": {v: all_judged[v] for v in args.variants},
    }
    report_path = Path(args.report) if args.report else REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n  report -> {report_path}")
    print(f"\n  WINNER: variant {winner} ({scores[winner]:.1f}/100)")

    if per_q_deltas:
        print(f"\n  Biggest regressions ({base} → other):")
        for d in per_q_deltas[:5]:
            print(f"    {d['id']} [{d['category']}] {d['base']}={d['base_overall']} "
                  f"→ {d['variant']}={d['variant_overall']} ({d['delta']:+.1f})")
        print(f"\n  Biggest improvements ({base} → other):")
        for d in per_q_deltas[-5:]:
            print(f"    {d['id']} [{d['category']}] {d['base']}={d['base_overall']} "
                  f"→ {d['variant']}={d['variant_overall']} ({d['delta']:+.1f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
