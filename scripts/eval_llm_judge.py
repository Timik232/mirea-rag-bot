"""LLM-as-judge answer-quality evaluation (task t_7a079b41).

Evaluates RAG-generated answers against golden expected answers using
glm-5.2-chat as the judge via the LiteLLM proxy.

Pipeline:
  1. Generate: run golden questions through answer_question() (production RAG:
     E5 retrieval -> Qdrant -> context build -> LM Studio generation).
  2. Judge:     glm-5.2-chat scores each answer on 4 dimensions (0-5):
                  correctness   - factual accuracy, no hallucinations
                  completeness  - key facts from expected answer present
                  relevance     - directly addresses the question
                  fluency       - Russian language quality
  3. Report:    aggregate means, breakdowns by category/difficulty,
                per-question detail with judge notes.

Outputs:
  mirea_data/eval/judge_answers.json  - cached generated answers (resumable)
  mirea_data/eval/judge_report.json   - full machine-readable report
  stdout                              - human-readable summary

Usage:
  python scripts/eval_llm_judge.py
  python scripts/eval_llm_judge.py --limit 5        # quick smoke test
  python scripts/eval_llm_judge.py --gen-model X     # override LM Studio model
  python scripts/eval_llm_judge.py --regenerate      # ignore cached answers
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import config  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_judge")

GOLDEN_PATH = PROJ / "mirea_data/eval/golden_qa.json"
ANSWERS_CACHE = PROJ / "mirea_data/eval/judge_answers.json"
REPORT_PATH = PROJ / "mirea_data/eval/judge_report.json"

DIMENSIONS = ["correctness", "completeness", "relevance", "fluency"]

# ── Judge endpoint (glm-5.2 via LiteLLM proxy) ──────────────────────────────

def _get_litellm_key() -> str:
    """Read the LiteLLM proxy API key from the Hermes config."""
    for p in [
        Path.home() / "AppData/Local/hermes/config.yaml",
        Path.home() / ".hermes/config.yaml",
    ]:
        if not p.exists():
            continue
        m = re.search(r"^  api_key:\s*(\S+)", p.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            return m.group(1)
    raise RuntimeError("LiteLLM API key not found in Hermes config")


JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", os.environ.get("LITELLM_API_BASE", "http://localhost:1234/v1"))
JUDGE_MODEL = "glm-5.2-chat"
_judge_api_key: str | None = None


def judge_api_key() -> str:
    global _judge_api_key
    if _judge_api_key is None:
        _judge_api_key = _get_litellm_key()
    return _judge_api_key


# ── Prompt ───────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "Ты — беспристрастный эксперт-оценщик ответов справочного чат-бота "
    "приёмной комиссии РТУ МИРЭА."
)

JUDGE_TEMPLATE = """Оцени ответ чат-бота на вопрос абитуриента.

ВОПРОС:
{question}

ЭТАЛОННЫЙ ОТВЕТ (правильный, написан экспертом):
{expected}

ОТВЕТ БОТА (сгенерирован автоматически):
{generated}

Оцени ответ бота по четырём критериям. Каждый критерий — целое число от 0 до 5:
  correctness  — фактическая точность (5 = все факты верны, 0 = грубые ошибки/выдумки).
  completeness — полнота (5 = переданы все ключевые факты эталона, 0 = ничего существенного).
  relevance    — релевантность (5 = точно отвечает на вопрос, 0 = не по теме).
  fluency      — качество русского языка (5 = идеально, 0 = нечитаемо/бессвязно).

Шкала: 0=отвратительно, 1=очень плохо, 2=плохо, 3=приемлемо, 4=хорошо, 5=отлично.
Если ответ бота — это отказ («У меня нет информации…»), оценивай ниже по
correctness/completeness даже если текст грамотный.

Ответь СТРОГО одной строкой JSON (без markdown, без лишнего текста):
{{"correctness": N, "completeness": N, "relevance": N, "fluency": N, "note": "краткий комментарий по-русски"}}"""


# ── LLM calls ─────────────────────────────────────────────────────────────────

def _chat_completion(base_url: str, api_key: str, model: str,
                     messages: list[dict], max_tokens: int = 2000,
                     temperature: float = 0.0, timeout: int = 90) -> str:
    """Call an OpenAI-compatible chat endpoint. Returns the content string."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from e
    return data["choices"][0]["message"]["content"].strip()


def judge_answer(question: str, expected: str, generated: str) -> dict:
    """Ask glm-5.2 to judge a single answer. Returns parsed scores + raw note.

    Retries up to 3 times on network errors / unparseable JSON.
    """
    user_msg = JUDGE_TEMPLATE.format(
        question=question, expected=expected, generated=generated
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    last_err = None
    for attempt in range(1, 4):
        try:
            raw = _chat_completion(
                JUDGE_BASE_URL, judge_api_key(), JUDGE_MODEL,
                messages, max_tokens=2500, temperature=0.0, timeout=90,
            )
        except Exception as e:
            last_err = f"net: {e!r}"
            log.warning("judge attempt %d failed: %s", attempt, last_err)
            time.sleep(2 * attempt)
            continue
        scores = _parse_judge_json(raw)
        if scores is not None:
            return scores
        last_err = f"unparseable: {raw[:120]!r}"
        log.warning("judge attempt %d unparseable: %s", attempt, last_err)
        time.sleep(1)
    log.error("judge failed after 3 attempts: %s", last_err)
    return {d: -1 for d in DIMENSIONS} | {"note": f"PARSE_FAIL: {last_err}", "raw": ""}


def _parse_judge_json(raw: str) -> dict | None:
    """Extract the JSON object from the judge response.

    glm-5.2 is a reasoning model — the output may contain chain-of-thought
    before the JSON. We find the last {...} block (reasoning may contain
    earlier braces) and parse it.
    """
    if not raw:
        return None
    # Try direct parse first
    try:
        return _validate_scores(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        pass
    # Find the last JSON object in the text
    starts = [m.start() for m in re.finditer(r"\{", raw)]
    for s in reversed(starts):
        # Find the matching closing brace
        depth = 0
        for i in range(s, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return _validate_scores(json.loads(raw[s : i + 1]))
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


def _validate_scores(obj: dict) -> dict:
    """Validate and normalise the parsed judge scores."""
    out = {}
    for d in DIMENSIONS:
        v = obj.get(d)
        if v is None:
            raise ValueError(f"missing dimension {d}")
        v = int(v)
        if not 0 <= v <= 5:
            raise ValueError(f"{d}={v} out of range [0,5]")
        out[d] = v
    out["note"] = str(obj.get("note", ""))[:300]
    return out


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answers(items: list[dict], gen_model: str,
                     cache_path: Path, regenerate: bool) -> list[dict]:
    """Run each golden question through answer_question(). Returns list of
    {id, question, expected_answer, generated_answer, sources, ...}.

    Caches incrementally to cache_path so interrupted runs can resume.
    """
    # Load cache
    cache: dict[str, dict] = {}
    if cache_path.exists() and not regenerate:
        try:
            cache = {r["id"]: r for r in json.loads(cache_path.read_text("utf-8"))["answers"]}
        except Exception:
            log.warning("cache load failed, starting fresh")

    # Override the production model name
    config.LM_STUDIO_CONFIG["model"] = gen_model

    from src.rag.answer import answer_question  # delayed import (loads E5)

    results = []
    n_cached = 0
    t0 = time.time()
    for i, item in enumerate(items):
        qid = item["id"]
        if qid in cache:
            results.append(cache[qid])
            n_cached += 1
            continue
        print(f"  [{i+1}/{len(items)}] generating {qid} ...", flush=True)
        try:
            res = answer_question(item["question"])
            gen = res.get("answer", "")
        except Exception as e:
            log.error("generation failed for %s: %s", qid, e)
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
        # Write cache after each generation (resumable)
        _write_cache(cache_path, results)
    elapsed = time.time() - t0
    print(f"  generation done: {len(results) - n_cached} new, "
          f"{n_cached} cached, {elapsed:.1f}s")
    return results


def _write_cache(path: Path, answers: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
             "gen_model": config.LM_STUDIO_CONFIG["model"], "answers": answers},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_scores(rows: list[dict]) -> dict:
    """Compute mean scores over a list of judge result rows."""
    valid = [r for r in rows if all(r.get(d, -1) >= 0 for d in DIMENSIONS)]
    n = len(valid)
    if n == 0:
        return {"n": 0, "n_failed": len(rows)}
    out = {"n": n, "n_total": len(rows), "n_failed": len(rows) - n}
    for d in DIMENSIONS:
        vals = [r[d] for r in valid]
        out[f"mean_{d}"] = round(sum(vals) / n, 3)
    # Overall = mean of 4 dimensions, scaled to 0-100
    overalls = [sum(r[d] for d in DIMENSIONS) / len(DIMENSIONS) for r in valid]
    out["mean_overall"] = round(sum(overalls) / n, 3)
    out["mean_overall_100"] = round(sum(overalls) / n / 5 * 100, 1)
    return out


def group_aggregate(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get(key, "?")].append(r)
    return {g: aggregate_scores(rs) for g, rs in sorted(groups.items())}


def score_distribution(rows: list[dict]) -> dict:
    """Distribution of overall scores (0-5 buckets)."""
    valid = [r for r in rows if all(r.get(d, -1) >= 0 for d in DIMENSIONS)]
    dist = {f"{i}-{i+1}": 0 for i in range(0, 5)}
    for r in valid:
        overall = sum(r[d] for d in DIMENSIONS) / len(DIMENSIONS)
        bucket = min(int(overall), 4)
        dist[f"{bucket}-{bucket+1}"] += 1
    return dist


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="only run first N questions (0 = all)")
    ap.add_argument("--gen-model", default="google/gemma-4-26b-a4b-qat",
                    help="LM Studio model for answer generation")
    ap.add_argument("--regenerate", action="store_true",
                    help="ignore cached answers, regenerate all")
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip generation, judge from existing cache only")
    args = ap.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text("utf-8"))
    items = golden["items"]
    if args.limit:
        items = items[: args.limit]
    n_items = len(items)
    print(f"golden items: {n_items}")
    print(f"gen model: {args.gen_model} | judge: {JUDGE_MODEL}")

    # ── Phase 1: Generate answers ──────────────────────────────────────────
    if args.skip_gen:
        if not ANSWERS_CACHE.exists():
            print("ERROR: --skip-gen but no cache at", ANSWERS_CACHE)
            return 2
        answers = json.loads(ANSWERS_CACHE.read_text("utf-8"))["answers"]
        print(f"loaded {len(answers)} cached answers (skip-gen)")
    else:
        print("\n=== Phase 1: Generate answers ===")
        answers = generate_answers(items, args.gen_model, ANSWERS_CACHE, args.regenerate)

    # ── Phase 2: Judge answers ─────────────────────────────────────────────
    print("\n=== Phase 2: LLM-as-judge (glm-5.2) ===")
    judge_rows: list[dict] = []
    t0 = time.time()
    for i, ans in enumerate(answers):
        qid = ans["id"]
        print(f"  [{i+1}/{len(answers)}] judging {qid} ...", flush=True)
        scores = judge_answer(
            ans["question"], ans["expected_answer"], ans["generated_answer"]
        )
        row = {
            "id": qid,
            "question": ans["question"],
            "category": ans.get("category", ""),
            "difficulty": ans.get("difficulty", ""),
            "expected_answer": ans["expected_answer"],
            "generated_answer": ans["generated_answer"],
            "context_chunks": ans.get("context_chunks", 0),
            "generation_time": ans.get("generation_time", 0),
            **scores,
        }
        judge_rows.append(row)
        # Progress heartbeat every 10 questions
        if (i + 1) % 10 == 0:
            print(f"    ({i+1}/{len(answers)} done, {time.time()-t0:.0f}s)")
    judge_elapsed = time.time() - t0
    print(f"  judging done in {judge_elapsed:.1f}s")

    # ── Phase 3: Aggregate + report ────────────────────────────────────────
    overall = aggregate_scores(judge_rows)
    by_cat = group_aggregate(judge_rows, "category")
    by_diff = group_aggregate(judge_rows, "difficulty")
    dist = score_distribution(judge_rows)

    # Worst-scoring questions (for the report)
    worst = sorted(
        [r for r in judge_rows if all(r.get(d, -1) >= 0 for d in DIMENSIONS)],
        key=lambda r: sum(r[d] for d in DIMENSIONS),
    )[:5]

    report = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "gen_model": args.gen_model,
            "judge_model": JUDGE_MODEL,
            "judge_endpoint": JUDGE_BASE_URL,
            "n_items": n_items,
            "dimensions": DIMENSIONS,
            "scale": "0-5 per dimension, overall = mean(dims) scaled to 0-100",
        },
        "overall": overall,
        "by_category": by_cat,
        "by_difficulty": by_diff,
        "score_distribution": dist,
        "worst_5": [
            {
                "id": r["id"],
                "question": r["question"],
                "category": r["category"],
                "scores": {d: r[d] for d in DIMENSIONS},
                "overall": round(sum(r[d] for d in DIMENSIONS) / 4, 2),
                "note": r.get("note", ""),
                "generated_answer": r["generated_answer"][:200],
            }
            for r in worst
        ],
        "per_query": judge_rows,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nreport -> {REPORT_PATH}")

    # Guard: no valid results
    if overall.get("n", 0) == 0:
        print("\n" + "=" * 64)
        print(f"LLM-AS-JUDGE: 0/{n_items} valid (all judge calls failed)")
        print("=" * 64)
        return 1

    # ── Human-readable summary ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"LLM-AS-JUDGE RESULTS  ({overall['n']}/{n_items} valid)")
    print("=" * 64)
    print(f"  Judge model : {JUDGE_MODEL}")
    print(f"  Gen model   : {args.gen_model}")
    print(f"  Overall     : {overall['mean_overall']:.2f} / 5.0  "
          f"({overall['mean_overall_100']:.1f}/100)")
    for d in DIMENSIONS:
        print(f"  {d:<13}: {overall[f'mean_{d}']:.2f}")
    if overall.get("n_failed"):
        print(f"  FAILED      : {overall['n_failed']} (judge parse error)")
    print(f"\n  Score distribution (overall 0-5 buckets):")
    for bucket, cnt in dist.items():
        print(f"    {bucket}: {'#' * cnt} ({cnt})")
    print(f"\n  By category:")
    print(f"    {'category':<16} {'n':>3} {'overall':>7} {'corr':>5} {'compl':>5} {'rel':>5} {'fluen':>5}")
    for cat, agg in by_cat.items():
        print(f"    {cat:<16} {agg['n']:>3} {agg['mean_overall']:>7.2f} "
              f"{agg['mean_correctness']:>5.2f} {agg['mean_completeness']:>5.2f} "
              f"{agg['mean_relevance']:>5.2f} {agg['mean_fluency']:>5.2f}")
    print(f"\n  By difficulty:")
    for diff, agg in by_diff.items():
        print(f"    {diff:<10} {agg['n']:>3} overall={agg['mean_overall']:.2f} "
              f"corr={agg['mean_correctness']:.2f} compl={agg['mean_completeness']:.2f}")
    print(f"\n  Worst 5:")
    for r in worst:
        s = {d: r[d] for d in DIMENSIONS}
        print(f"    {r['id']} [{r['category']}] overall={sum(r[d] for d in DIMENSIONS)/4:.1f} "
              f"{s} — {r['question'][:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
