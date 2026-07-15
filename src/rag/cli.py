#!/usr/bin/env python3
"""Micro-CLI for testing the RAG answer layer.

Interactive REPL: type a question, get an answer + sources.
Exit with: quit, exit, or Ctrl+C.

Usage:
    python -m src.rag.cli
"""

import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rag.answer import answer_question


BANNER = """
╔══════════════════════════════════════════════╗
║  МИРЭА RAG Бот — тестовый CLI                ║
║  Задайте вопрос о МИРЭА (или 'quit' для выхода)  ║
╚══════════════════════════════════════════════╝
""".strip()


def main():
    print(BANNER)
    print()

    # Warm up the model on first run
    print("Загрузка моделей (при первом запуске может занять время)...")
    try:
        from src.rag.answer import _get_embedding_model
        _get_embedding_model()
        print("Готово!\n")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}\n")

    while True:
        try:
            query = input("❓ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nДо свидания!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "выход", "q"):
            print("До свидания!")
            break

        print()
        try:
            result = answer_question(query)

            # Print answer
            print(f"💬 {result['answer']}")
            print()

            # Print sources
            if result["sources"]:
                print("📎 Источники:")
                for src in result["sources"]:
                    score = f"{src['score']:.1%}" if src.get("score") else ""
                    section = src.get("section", "")
                    url = src.get("url", "")
                    print(f"   • [{section}] {score} {url}")

            # Print stats
            print(f"\n⏱ {result['generation_time']:.1f}с | "
                  f"чанков: {result['context_chunks']} | "
                  f"scores: {result.get('retrieval_scores', [])}")

        except ConnectionError as e:
            print(f"\n❌ {e}")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
