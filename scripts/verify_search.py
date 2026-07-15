#!/usr/bin/env python3
"""Interactive verification script for semantic search over MIREA data.

Encodes each query with the E5 model, runs vector search via Qdrant,
and prints PASS/FAIL for each query based on keyword overlap + score.
"""

import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.embeddings.indexer import search_qdrant


QUERIES = [
    {
        "query": "Какие факультеты есть в МИРЭА?",
        "keywords": ["факультет", "институт", "подразделени"],
        "label": "faculties",
    },
    {
        "query": "проходной балл",
        "keywords": ["балл", "проходн", "конкурс"],
        "label": "admission scores",
    },
    {
        "query": "адрес главного корпуса",
        "keywords": ["москва", "ул.", "проспект", "адрес", "корпус"],
        "label": "main address",
    },
    {
        "query": "направление кибербезопасность",
        "keywords": ["кибер", "безопасност", "криптограф"],
        "label": "cybersecurity",
    },
    {
        "query": "стипендия",
        "keywords": ["стипенди", "выплат", "социальн"],
        "label": "scholarship",
    },
    {
        "query": "общежитие",
        "keywords": ["общежит", "проживан", "комнат", "расселен"],
        "label": "dormitory",
    },
    {
        "query": "трудоустройство выпускников",
        "keywords": ["трудоустройств", "выпускник", "работ", "карьер"],
        "label": "employment",
    },
    {
        "query": "контакты приемной комиссии",
        "keywords": ["телефон", "495", "приёмн", "приемн", "контакт"],
        "label": "admission contacts",
    },
    {
        "query": "институт информационных технологий",
        "keywords": ["информационных технолог", "ИИТ", "институт"],
        "label": "IT institute",
    },
    {
        "query": "внеучебная деятельность",
        "keywords": ["внеучебн", "деятельност", "клуб", "спорт", "мероприят"],
        "label": "extracurricular",
    },
    {
        "query": "программа обучения",
        "keywords": ["программ", "обучен", "дисциплин", "курс"],
        "label": "curriculum",
    },
    {
        "query": "документы для поступления",
        "keywords": ["документ", "поступлен", "заявлен", "паспорт", "егэ"],
        "label": "admission docs",
    },
    {
        "query": "Где находится главный корпус МИРЭА?",
        "keywords": ["корпус", "стромынк", "москва"],
        "label": "QA: main building",
    },
    {
        "query": "кафедры университета",
        "keywords": ["кафедр", "подготовк", "преподавател", "институт", "образован", "направлен", "подразделен", "технолог"],
        "label": "departments",
    },
    # --- Student-specific queries (required by plan) ---
    {
        "query": "какие стипендии доступны",
        "keywords": ["стипенди", "выплат", "социальн", "академическ", "именных"],
        "label": "student: stipendii",
    },
    {
        "query": "как заселиться в общежитие",
        "keywords": ["общежит", "заселен", "проживан", "заявлен", "документ"],
        "label": "student: hostel",
    },
    {
        "query": "расписание занятий",
        "keywords": ["расписан", "занят", "пары", "семестр", "учебн"],
        "label": "student: education schedule",
    },
    {
        "query": "студенческие кружки и объединения",
        "keywords": ["объединен", "студенческ", "клуб", "секц", "спорт", "отряд", "профсоюз", "совет"],
        "label": "student: extracurricular",
    },
    {
        "query": "контакты деканата",
        "keywords": ["деканат", "декан", "телефон", "контакт", "кабинет"],
        "label": "student: contacts",
    },
]

EDGE_CASES = [
    {"query": "", "label": "empty query"},
    {"query": "МИРЭА", "label": "very short query"},
    {"query": "факултет", "label": "typo query"},
    {"query": "рецепт борща", "label": "out-of-domain query"},
]


def encode_query(model, text):
    prefixed = f"query: {text}"
    vec = model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=False)
    return vec.astype(np.float32)


def keyword_hit(results, keywords):
    for kw in keywords:
        for r in results:
            if kw.lower() in r["payload"]["text"].lower():
                return True
    return False


def main():
    print("=" * 70)
    print("MIREA Semantic Search Verification (Qdrant Backend)")
    print("=" * 70)
    print()

    print(f"Loading model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    from src.embeddings.qdrant_indexer import QdrantIndexer
    qi = QdrantIndexer()
    info = qi.get_collection_info(config.QDRANT_PATH, config.QDRANT_COLLECTION)
    print(f"Qdrant collection '{config.QDRANT_COLLECTION}': {info['points_count']} vectors, status={info['status']}")
    print()

    low_score_count = 0
    pass_count = 0
    fail_count = 0

    print("-" * 70)
    print("MAIN QUERIES")
    print("-" * 70)

    for q in QUERIES:
        query = q["query"]
        keywords = q["keywords"]
        label = q["label"]

        vec = encode_query(model, query)
        results = search_qdrant(vec, top_k=config.TOP_K_RETRIEVAL)

        hit = keyword_hit(results, keywords)
        top_score = results[0]["score"] if results else 0.0
        scores = [r["score"] for r in results]

        if top_score < 0.3:
            low_score_count += 1

        status = "PASS" if hit and top_score > 0.3 else "FAIL"
        if status == "PASS":
            pass_count += 1
        else:
            fail_count += 1

        print(f"\n[{status}] {label}: \"{query}\"")
        print(f"  Top score: {top_score:.4f} | Scores: {[f'{s:.4f}' for s in scores]}")
        if top_score < 0.3:
            print(f"  WARNING: top score < 0.3")
        for j, r in enumerate(results):
            preview = r["payload"]["text"][:100].replace("\n", " ")
            section = r["payload"].get("section", "?")
            print(f"  [{j+1}] (score={r['score']:.4f}, section={section}) {preview}...")

    print()
    print("-" * 70)
    print("EDGE CASES (must not crash)")
    print("-" * 70)

    for ec in EDGE_CASES:
        query = ec["query"]
        label = ec["label"]
        try:
            vec = encode_query(model, query)
            results = search_qdrant(vec, top_k=config.TOP_K_RETRIEVAL)
            top_score = results[0]["score"] if results else 0.0
            preview = results[0]["payload"]["text"][:80].replace("\n", " ") if results else "(none)"
            print(f"\n[PASS] {label}: \"{query}\"")
            print(f"  Top score: {top_score:.4f} | {preview}...")
            pass_count += 1
        except Exception as e:
            print(f"\n[FAIL] {label}: \"{query}\"")
            print(f"  Error: {e}")
            fail_count += 1

    print()
    print("=" * 70)
    total = pass_count + fail_count
    print(f"RESULTS: {pass_count}/{total} passed, {fail_count} failed")
    if low_score_count > 0:
        print(f"WARNING: {low_score_count} queries had top score < 0.3")
    print("=" * 70)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
