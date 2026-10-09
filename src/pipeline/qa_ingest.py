"""qa_ingest.py — Load and normalize Q&A pairs into the project's standard schema.

Usage:
    python -m src.pipeline.qa_ingest

Output:
    mirea_data/raw/qa_pairs.json
"""

import csv
import json
import logging
import os
import re
import sys

sys.path.insert(0, ".")
import config

logger = logging.getLogger(__name__)

MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 20


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    return text


def clean_text(text: str) -> str:
    text = strip_html(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def to_standard_schema(question: str, answer: str) -> dict[str, str]:
    return {
        "title": question,
        "text": answer,
        "url": "",
        "section": "qa_pairs",
        "subsection": "curated",
    }


def load_json(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data: list[dict[str, str]] = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be an array of objects")
    pairs: list[tuple[str, str]] = []
    for item in data:
        q = item.get("question") or item.get("title") or ""
        a = item.get("answer") or item.get("text") or ""
        pairs.append((q, a))
    return pairs


def load_csv(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row.get("question", ""), row.get("answer", "")))
    return pairs


def load_tsv(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        first_row = next(reader, None)
        if first_row is None:
            return pairs
        header_lower = [c.lower().strip() for c in first_row]
        if "question" in header_lower and "answer" in header_lower:
            pass
        else:
            if len(first_row) >= 2 and (first_row[0].strip() or first_row[1].strip()):
                pairs.append((first_row[0].strip(), "\t".join(first_row[1:]).strip()))
        for row in reader:
            if len(row) < 2:
                continue
            question = row[0].strip()
            answer = "\t".join(row[1:]).strip()
            if not question and not answer:
                continue
            pairs.append((question, answer))
    return pairs


def load_qa_pairs(path: str) -> list[tuple[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        return load_tsv(path)
    try:
        return load_json(path)
    except (json.JSONDecodeError, ValueError):
        return load_csv(path)


def validate_and_transform(pairs: list[tuple[str, str]]) -> tuple[list[dict[str, str]], int]:
    valid: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    skipped = 0
    for question, answer in pairs:
        q = clean_text(question)
        a = clean_text(answer)
        if not q or not a:
            skipped += 1
            continue
        if len(q) < MIN_QUESTION_LENGTH:
            logger.warning("Skipping short question (%d chars, min %d): %s",
                           len(q), MIN_QUESTION_LENGTH, q[:50])
            skipped += 1
            continue
        if len(a) < MIN_ANSWER_LENGTH:
            logger.warning("Skipping short answer (%d chars, min %d) for: %s",
                           len(a), MIN_ANSWER_LENGTH, q[:50])
            skipped += 1
            continue
        normalized = q.lower().strip()
        if normalized in seen_questions:
            logger.warning("Duplicate question skipped: %s", q[:50])
            skipped += 1
            continue
        seen_questions.add(normalized)
        valid.append(to_standard_schema(q, a))
    return valid, skipped


DEMO_QA_PAIRS: list[dict[str, str]] = [
    {
        "question": "Какие институты есть в МИРЭА?",
        "answer": "В состав МИРЭА входят следующие институты: Институт информационных технологий (ИИТ), Институт радиотехники и систем связи (ИРСС), Институт кибербезопасности и криптографии (ИКБ), Институт интеллектуальных робототехнических систем (ИИРС), Институт оптико-электронных систем и технологий (ИОЭСТ), Институт экологической безопасности и природопользования (ИЭБП), Институт экономики и управления (ИЭиУ), Институт высшего образования (ИВО), а также Институт дизайна и визуальных искусств (ИДВИ).",
    },
    {
        "question": "Какой проходной балл на программную инженерию?",
        "answer": "Проходной балл на направление «Программная инженерия» в МИРЭА варьируется по годам. В среднем за последние годы он составляет 220–260 баллов в зависимости от формы обучения (бюджет/контракт), кампуса и уровня конкуренции. Точную информацию по текущему году рекомендуется уточнять на официальном сайте университета или в приёмной комиссии.",
    },
    {
        "question": "Где находится главный корпус МИРЭА?",
        "answer": "Главный корпус МИРЭА (Московского технологического университета) расположен по адресу: г. Москва, ул. Стромынка, д. 20. Это историческое здание, в котором размещается ректорат и основные административные подразделения университета.",
    },
    {
        "question": "Как связаться с приёмной комиссией?",
        "answer": "Связаться с приёмной комиссией МИРЭА можно несколькими способами: по телефону +7 (495) 734-72-49, по электронной почте priem@mirea.ru, а также лично по адресу: г. Москва, ул. Стромынка, д. 20, кабинет приёмной комиссии. Актуальные часы работы указаны на официальном сайте mirea.ru в разделе «Абитуриентам».",
    },
    {
        "question": "Какие документы нужны для поступления?",
        "answer": "Для поступления в МИРЭА необходимы следующие документы: паспорт гражданина РФ, аттестат о полном среднем образовании (оригинал или копия), результаты ЕГЭ по профильным предметам, 6 фотографий 3×4, медицинская справка (форма 086/у), СНИЛС. Для льготных категорий также потребуются документы, подтверждающие право на льготы. Подробный перечень всегда доступен на сайте mirea.ru.",
    },
    {
        "question": "Есть ли общежитие для иногородних студентов?",
        "answer": "Да, МИРЭА предоставляет общежития для иногородних студентов. Университет располагает несколькими общежитиями, расположенными в пределах Московского метро. Заселение происходит на основании заявления и при наличии путёвки. Иногородние студенты, поступившие на бюджет, обеспечиваются местом в общежитии в приоритетном порядке.",
    },
    {
        "question": "Какие платформы дистанционного обучения используются?",
        "answer": "МИРЭА использует собственную образовательную платформу LMS (Learning Management System), а также Microsoft Teams для проведения онлайн-лекций и семинаров. Доступ к материалам курсов осуществляется через личный кабинет студента на портале mirea.ru.",
    },
    {
        "question": "Можно ли перевестись из другого вуза в МИРЭА?",
        "answer": "Да, перевод из другого вуза в МИРЭА возможен. Для этого необходимо предоставить справку об обучении из текущего вуза, заполнить заявление на перевод и пройти аттестацию по уже изученным дисциплинам. Перевод осуществляется в соответствии с Порядком перевода студентов, утверждённым Министерством науки и высшего образования РФ.",
    },
    {
        "question": "Какие направления подготовки самые популярные?",
        "answer": "Среди самых популярных направлений подготовки в МИРЭА: программная инженерия, информационная безопасность, радиотехника, искусственный интеллект, робототехника, дизайн и компьютерная графика. Эти направления традиционно имеют наибольший конкурс среди абитуриентов.",
    },
    {
        "question": "Есть ли военная кафедра?",
        "answer": "В МИРЭА действует военная учебный центр (ВУЦ), который готовит офицеров запаса по программе военной подготовки. Обучение на военной кафедре доступно для студентов очной формы обучения. По окончании обучения выпускникам присваивается офицерское звание.",
    },
    {
        "question": "Какие языки программирования изучают на ИИТ?",
        "answer": "Студенты Института информационных технологий МИРЭА изучают широкий спектр языков программирования. На первых курсах изучаются Python, C и C++, Java. На старших курсах добавляются JavaScript, TypeScript, Go, SQL, а также изучаются специализированные технологии для Data Science, веб-разработки и системного программирования.",
    },
    {
        "question": "Какие scholarships and grants are available for students?",
        "answer": "В МИРЭА студентам доступны различные формы материальной поддержки: академическая стипендия (выплачивается при отсутствии задолженностей), повышенная стипендия за отличную учёбу, стипендия Правительства Москвы для отличников, президентская стипендия, гранты на научно-исследовательские работы. Также существует помощь нуждающимся студентам через социальную стипендию.",
    },
    {
        "question": "Как пройти практику во время обучения?",
        "answer": "Практика является обязательной частью учебного процесса в МИРЭА. Виды практики включают учебную (2-3 курс), производственную (3-4 курс) и преддипломную (выпускной курс). Студенты проходят практику на базе профильных предприятий-партнёров университета, среди которых крупные IT-компании, научные центры и промышленные предприятия.",
    },
    {
        "question": "Сколько стоит контрактное обучение?",
        "answer": "Стоимость контрактного обучения в МИРЭА зависит от направления подготовки и формы обучения. В среднем она составляет от 180 000 до 320 000 рублей в год для бакалавриата. Для магистратуры стоимость немного выше. Точные цены по каждому направлению публикуются ежегодно на официальном сайте mirea.ru.",
    },
    {
        "question": "Какие extracurricular activities есть в университете?",
        "answer": "В МИРЭА активно развита внеучебная деятельность: студенческие спортивные секции (волейбол, баскетбол, шахматы, настольный теннис), творческие коллективы (КВН, театр, вокал), научные кружки и лаборатории, клубы по интересам (робототехника, геймдев, хакинг). Также регулярно проводуются конференции, хакатоны и фестивали.",
    },
]


def create_demo_data() -> list[tuple[str, str]]:
    return [(item["question"], item["answer"]) for item in DEMO_QA_PAIRS]


def main():
    input_path = config.QA_DATA_PATH
    output_dir = config.DATA_RAW_DIR
    output_path = os.path.join(output_dir, "qa_pairs.json")

    os.makedirs(output_dir, exist_ok=True)

    # Load Q&A pairs
    if os.path.exists(input_path):
        raw_pairs = load_qa_pairs(input_path)
        print(f"Loaded data from {input_path}")
    else:
        print(f"Input file '{input_path}' not found — FAILING (no silent demo fallback)")
        sys.exit(1)

    total = len(raw_pairs)

    # Validate and transform
    items, skipped = validate_and_transform(raw_pairs)

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Loaded {total} items, skipped {skipped} invalid, saved {len(items)} items")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
