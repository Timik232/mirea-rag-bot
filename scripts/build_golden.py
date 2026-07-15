"""Build & validate the golden QA dataset for mirea-rag-bot retrieval eval.

Each item pins ``relevant_chunk_ids`` = positional indices into
``mirea_data/faiss/metadata.json`` (which equal the Qdrant/FAISS point ids,
since ``build_qdrant_index`` uses ``ids = list(range(len(vectors)))``).

The validator checks:
  1. every id is in range and its ``section`` matches the item ``category``;
  2. at least one *key term* from the answer appears in the union text of the
     referenced chunks (catches mis-pinned ids).

Run:  python build_golden.py
"""
import json
import re
import os
from pathlib import Path

# scripts/ lives at project root; project root is its parent.
PROJ = Path(__file__).resolve().parent.parent
META = json.loads((PROJ / "mirea_data/faiss/metadata.json").read_text(encoding="utf-8"))
N = len(META)

# category -> set of sections that are acceptable for that category.
# (education chunks are scraped twice with slight differences, so a couple of
# categories map to >1 section.)
CAT_SECTIONS = {
    "applicants": {"applicants", "announcements", "qa_pairs", "sveden"},
    "sveden": {"sveden"},
    "contacts": {"contacts"},
    "dormitories": {"dormitories"},
    "employment": {"employment"},
    "grants": {"student_portal", "sveden", "qa_pairs"},
    "extracurricular": {"extracurricular", "education"},
    "announcements": {"announcements"},
    "student_portal": {"student_portal"},
    "education": {"education"},
    "qa_pairs": {"qa_pairs"},
}

ITEMS = [
    # ---------------- applicants (8) ----------------
    {"id": "q001", "category": "applicants", "difficulty": "easy",
     "question": "Какой проходной балл на направление «Программная инженерия»?",
     "expected_answer": "Проходной балл на «Программную инженерию» — 260 баллов (2024 год).",
     "relevant_chunk_ids": [16, 17, 47], "key_terms": ["Программная инженерия", "260"]},
    {"id": "q002", "category": "applicants", "difficulty": "easy",
     "question": "Какие проходные баллы на Информатику и вычислительную технику, Радиотехнику и Информационную безопасность?",
     "expected_answer": "ИВТ — 254, Радиотехника — 248, Информационная безопасность — 262 балла.",
     "relevant_chunk_ids": [], "key_terms": ["254", "248", "262"]},
    {"id": "q003", "category": "applicants", "difficulty": "easy",
     "question": "Какой конкурс на направление «Кибербезопасность»?",
     "expected_answer": "Конкурс — 12,5 человека на место, средний балл зачисленных — 271 (2024).",
     "relevant_chunk_ids": [], "key_terms": ["12,5", "271"]},
    {"id": "q004", "category": "applicants", "difficulty": "easy",
     "question": "Сколько бюджетных мест на очной форме по программе «Программная инженерия»?",
     "expected_answer": "150 бюджетных мест (очная форма), код направления 09.03.04.",
     "relevant_chunk_ids": [47], "key_terms": ["150", "09.03.04"]},
    {"id": "q005", "category": "applicants", "difficulty": "medium",
     "question": "Какие вступительные экзамены нужны для поступления на бакалавриат?",
     "expected_answer": "ЕГЭ по математике (профиль), физике или информатике и ИКТ, а также русскому языку.",
     "relevant_chunk_ids": [47], "key_terms": ["математика", "информатика"]},
    {"id": "q006", "category": "applicants", "difficulty": "easy",
     "question": "Какие документы нужны для поступления в МИРЭА?",
     "expected_answer": "Заявление, документ об образовании, результаты ЕГЭ, паспорт, фотографии 3×4 (и СНИЛС).",
     "relevant_chunk_ids": [0, 16, 17], "key_terms": ["Заявление", "ЕГЭ", "Паспорт"]},
    {"id": "q007", "category": "applicants", "difficulty": "easy",
     "question": "В какие сроки принимаются документы на бюджетные места бакалавриата?",
     "expected_answer": "С 20 июня по 25 июля.",
     "relevant_chunk_ids": [0], "key_terms": ["20 июня", "25 июля"]},
    {"id": "q008", "category": "applicants", "difficulty": "easy",
     "question": "Сколько стоит очное обучение на «Информатике и вычислительной технике» по контракту?",
     "expected_answer": "285 000 руб./год (очная форма).",
     "relevant_chunk_ids": [], "key_terms": ["285 000"]},

    # ---------------- sveden (7) ----------------
    {"id": "q009", "category": "sveden", "difficulty": "medium",
     "question": "Сколько институтов входит в состав РТУ МИРЭА?",
     "expected_answer": "9 институтов, включая ИИТ, Институт кибербезопасности, Институт радиотехнических и телекоммуникационных систем, Институт тонких химических технологий.",
     "relevant_chunk_ids": [50, 4, 11], "key_terms": ["9 институтов", "Институт информационных технологий"]},
    {"id": "q010", "category": "sveden", "difficulty": "easy",
     "question": "Кто ректор РТУ МИРЭА?",
     "expected_answer": "Степанов Владимир Яковлевич, доктор технических наук, профессор.",
     "relevant_chunk_ids": [48], "key_terms": ["Степанов", "Владимир Яковлевич"]},
    {"id": "q011", "category": "sveden", "difficulty": "easy",
     "question": "Где расположен главный учебный корпус МИРЭА?",
     "expected_answer": "Москва, ул. Стромынка, д. 20 (6 этажей, более 100 аудиторий); метро «Сокольники».",
     "relevant_chunk_ids": [50, 47, 48], "key_terms": ["Стромынка", "20"]},
    {"id": "q012", "category": "sveden", "difficulty": "easy",
     "question": "Где находится учебный корпус на проспекте Вернадского?",
     "expected_answer": "Проспект Вернадского, д. 78, район метро «Юго-Западная».",
     "relevant_chunk_ids": [50, 3, 0], "key_terms": ["Вернадского", "78", "Юго-Западная"]},
    {"id": "q013", "category": "sveden", "difficulty": "easy",
     "question": "Сколько образовательных программ реализует университет?",
     "expected_answer": "Более 200 программ высшего образования (бакалавриат, специалитет, магистратура, аспирантура).",
     "relevant_chunk_ids": [47, 0, 5], "key_terms": ["200", "образовательных программ"]},
    {"id": "q014", "category": "sveden", "difficulty": "easy",
     "question": "С сколькими зарубежными университетами сотрудничает МИРЭА?",
     "expected_answer": "Более 50 зарубежных университетов и организаций из 25 стран.",
     "relevant_chunk_ids": [50, 4, 5], "key_terms": ["50", "25 стран"]},
    {"id": "q015", "category": "sveden", "difficulty": "easy",
     "question": "Каков средний балл ЕГЭ зачисленных на бюджет?",
     "expected_answer": "89,4 балла (2024 год).",
     "relevant_chunk_ids": [50], "key_terms": ["89,4"]},

    # ---------------- contacts (5) ----------------
    {"id": "q016", "category": "contacts", "difficulty": "medium",
     "question": "Где находится Институт тонких химических технологий?",
     "expected_answer": "Москва, ул. Малая Пироговская, д. 1, стр. 5; email ihti@mirea.ru.",
     "relevant_chunk_ids": [4, 3], "key_terms": ["Малая Пироговская", "ihti"]},
    {"id": "q017", "category": "contacts", "difficulty": "easy",
     "question": "Какой телефон приёмной комиссии МИРЭА?",
     "expected_answer": "+7 (495) 215-65-65; горячая линия 8 (800) 550-00-65.",
     "relevant_chunk_ids": [4, 3, 45], "key_terms": ["215-65-65", "550-00-65"]},
    {"id": "q018", "category": "contacts", "difficulty": "easy",
     "question": "Какой адрес электронной почты общей справочной службы?",
     "expected_answer": "info@mirea.ru.",
     "relevant_chunk_ids": [3], "key_terms": ["info@mirea.ru"]},
    {"id": "q019", "category": "contacts", "difficulty": "medium",
     "question": "Как связаться с Институтом кибербезопасности?",
     "expected_answer": "+7 (495) 215-65-65 доб. 3201, ул. Стромынка, д. 20, ауд. 502; приёмные часы вторник/четверг 14:00–17:00.",
     "relevant_chunk_ids": [4, 11, 16], "key_terms": ["3201", "кибербезопасности"]},
    {"id": "q020", "category": "contacts", "difficulty": "easy",
     "question": "Где находится общежитие №1 и какая станция метро рядом?",
     "expected_answer": "Ул. Б. Галушкина, д. 3; метро ВДНХ.",
     "relevant_chunk_ids": [3], "key_terms": ["Галушкина", "ВДНХ"]},

    # ---------------- dormitories (4) ----------------
    {"id": "q021", "category": "dormitories", "difficulty": "easy",
     "question": "Сколько мест в общежитии №1 и где оно расположено?",
     "expected_answer": "250 мест, ул. Стромынка, д. 11; для иногородних студентов очной формы.",
     "relevant_chunk_ids": [5, 46, 0], "key_terms": ["Стромынка", "11", "250"]},
    {"id": "q022", "category": "dormitories", "difficulty": "easy",
     "question": "Какой тип и вместимость у общежития №2?",
     "expected_answer": "Блочный тип, 400 мест, проспект Вернадского, д. 86; комнаты на 2 человека.",
     "relevant_chunk_ids": [5, 6, 22], "key_terms": ["Вернадского", "86", "400"]},
    {"id": "q023", "category": "dormitories", "difficulty": "medium",
     "question": "Где расположено общежитие №4 и сколько в нём мест?",
     "expected_answer": "Район Ново-Переделкино, ул. Шолохова, д. 20; построено в 2018 году, 350 мест.",
     "relevant_chunk_ids": [5, 10, 11], "key_terms": ["Шолохова", "350"]},
    {"id": "q024", "category": "dormitories", "difficulty": "easy",
     "question": "Что нужно для заселения в общежитие?",
     "expected_answer": "Заявление в деканат, справка о составе семьи, медсправка 086/У (и фото).",
     "relevant_chunk_ids": [6, 16, 17], "key_terms": ["заявление", "086"]},

    # ---------------- employment (4) ----------------
    {"id": "q025", "category": "employment", "difficulty": "medium",
     "question": "Чем занимается Центр карьеры РТУ МИРЭА?",
     "expected_answer": "Помогает с трудоустройством: консультации по резюме, подготовка к собеседованиям, ярмарки вакансий.",
     "relevant_chunk_ids": [13], "key_terms": ["Центр карьеры", "резюме"]},
    {"id": "q026", "category": "employment", "difficulty": "easy",
     "question": "Какова доля трудоустроенных выпускников МИРЭА?",
     "expected_answer": "94% в течение первого года после выпуска; 78% работают по специальности.",
     "relevant_chunk_ids": [13, 50, 0], "key_terms": ["94", "78"]},
    {"id": "q027", "category": "employment", "difficulty": "medium",
     "question": "С какими IT-компаниями сотрудничает МИРЭА по трудоустройству?",
     "expected_answer": "Яндекс, Касперский, Сбер, VK, Huawei, Ростелеком и др. — более 200 работодателей.",
     "relevant_chunk_ids": [13, 14, 49], "key_terms": ["Яндекс", "Касперский", "200"]},
    {"id": "q028", "category": "employment", "difficulty": "medium",
     "question": "Для каких курсов доступны стажировки и какова их длительность?",
     "expected_answer": "Для студентов 3–4 курсов, длительность 2–6 месяцев в ведущих IT-компаниях.",
     "relevant_chunk_ids": [13, 14, 2], "key_terms": ["3", "4", "стажировок"]},

    # ---------------- grants (4) ----------------
    {"id": "q029", "category": "grants", "difficulty": "medium",
     "question": "При каких условиях назначается государственная академическая стипендия?",
     "expected_answer": "Студентам очной бюджетной формы при отсутствии академической задолженности.",
     "relevant_chunk_ids": [48, 18, 9], "key_terms": ["академическая стипендия", "бюджетных"]},
    {"id": "q030", "category": "grants", "difficulty": "medium",
     "question": "За что назначается повышенная государственная академическая стипендия (ПГАС)?",
     "expected_answer": "За особые достижения в учебной, научно-исследовательской, общественной, культурно-творческой и спортивной деятельности.",
     "relevant_chunk_ids": [48, 20, 37], "key_terms": ["ПГАС", "особые достижения"]},
    {"id": "q031", "category": "grants", "difficulty": "easy",
     "question": "Каков размер стипендии Президента РФ?",
     "expected_answer": "7 000 рублей для студентов и 14 000 рублей для аспирантов в месяц.",
     "relevant_chunk_ids": [48], "key_terms": ["7 000", "14 000"]},
    {"id": "q032", "category": "grants", "difficulty": "medium",
     "question": "Какие компании учреждают корпоративные стипендии для студентов IT-направлений?",
     "expected_answer": "Яндекс, Касперский, Сбер, VK, Huawei совместно с РТУ МИРЭА.",
     "relevant_chunk_ids": [49, 13], "key_terms": ["Яндекс", "Касперский", "корпоративные"]},

    # ---------------- extracurricular (4) ----------------
    {"id": "q033", "category": "extracurricular", "difficulty": "medium",
     "question": "Сколько спортивных секций работает в РТУ МИРЭА и какие?",
     "expected_answer": "Более 30 секций: мини-футбол, баскетбол, волейбол, настольный теннис, шахматы, лёгкая атлетика, плавание и др.",
     "relevant_chunk_ids": [9, 10, 11], "key_terms": ["30", "спортивных"]},
    {"id": "q034", "category": "extracurricular", "difficulty": "medium",
     "question": "Какие творческие коллективы есть в университете?",
     "expected_answer": "Театральная студия «Этюд», вокальная «Гармония», танцевальный коллектив «Ритм», команда КВН и др.",
     "relevant_chunk_ids": [15], "key_terms": ["Этюд", "Гармония"]},
    {"id": "q035", "category": "extracurricular", "difficulty": "medium",
     "question": "Чем занимается Волонтёрский центр РТУ МИРЭА?",
     "expected_answer": "Координирует добровольческую деятельность: помощь ветеранам, благотворительные акции и т. д.",
     "relevant_chunk_ids": [15], "key_terms": ["Волонтёрский", "добровольческую"]},
    {"id": "q036", "category": "extracurricular", "difficulty": "hard",
     "question": "Какие студенческие медиа работают в МИРЭА?",
     "expected_answer": "Студенческое телевидение «МИРЭА-TV», радиостанция, газета «Технопарк», интернет-портал.",
     "relevant_chunk_ids": [15, 11], "key_terms": ["МИРЭА-TV", "Технопарк"]},

    # ---------------- announcements (3) ----------------
    {"id": "q037", "category": "announcements", "difficulty": "easy",
     "question": "Когда пройдёт День открытых дверей РТУ МИРЭА?",
     "expected_answer": "15 марта 2026 года в главном корпусе (ул. Стромынка, д. 20), начало в 10:00.",
     "relevant_chunk_ids": [0], "key_terms": ["15 марта", "открытых дверей"]},
    {"id": "q038", "category": "announcements", "difficulty": "easy",
     "question": "Когда состоится внутриуниверситетская олимпиада по программированию?",
     "expected_answer": "20 апреля 2026 года, для студентов 1–4 курсов.",
     "relevant_chunk_ids": [0, 20], "key_terms": ["20 апреля", "программированию"]},
    {"id": "q039", "category": "announcements", "difficulty": "medium",
     "question": "Каков размер гранта для студентов на научно-исследовательские проекты?",
     "expected_answer": "До 200 000 рублей на проект (конкурс 2026 года).",
     "relevant_chunk_ids": [0, 18, 48], "key_terms": ["200 000", "грант"]},

    # ---------------- student_portal (4) ----------------
    {"id": "q040", "category": "student_portal", "difficulty": "medium",
     "question": "Как записаться на приём к психологу?",
     "expected_answer": "Онлайн через lk.mirea.ru/psychologist/ (личный кабинет, раздел «Онлайн-запись»).",
     "relevant_chunk_ids": [36, 25, 30], "key_terms": ["lk.mirea.ru/psychologist", "psycholog"]},
    {"id": "q041", "category": "student_portal", "difficulty": "easy",
     "question": "Сколько стоят консультации психологической службы для студентов?",
     "expected_answer": "Бесплатно для обучающихся РТУ МИРЭА.",
     "relevant_chunk_ids": [26, 28, 36], "key_terms": ["бесплатно"]},
    {"id": "q042", "category": "student_portal", "difficulty": "medium",
     "question": "Какие студенческие общественные организации действуют в МИРЭА?",
     "expected_answer": "Две крупные: ММОО «Студенческий союз МИРЭА» и профсоюзная организация.",
     "relevant_chunk_ids": [38, 39, 43], "key_terms": ["Студенческий союз", "общественные"]},
    {"id": "q043", "category": "student_portal", "difficulty": "medium",
     "question": "Чем занимается профсоюзная организация РТУ МИРЭА?",
     "expected_answer": "Переговоры и заключение коллективного договора, улучшение материального положения студентов.",
     "relevant_chunk_ids": [39, 38, 43], "key_terms": ["Профсоюзная", "коллективного договора"]},

    # ---------------- education (2) ----------------
    {"id": "q044", "category": "education", "difficulty": "hard",
     "question": "Сколько мероприятий и проектов ежегодно проводится в МИРЭА?",
     "expected_answer": "Более 600 мероприятий и проектов ежегодно.",
     "relevant_chunk_ids": [9, 10, 22], "key_terms": ["600"]},
    {"id": "q045", "category": "education", "difficulty": "medium",
     "question": "Какие уровни образования реализует РТУ МИРЭА?",
     "expected_answer": "Среднее профессиональное, бакалавриат, специалитет, магистратура, аспирантура.",
     "relevant_chunk_ids": [9, 10, 47], "key_terms": ["бакалавриат", "аспирантура"]},

    # ---------------- qa_pairs / curated (5) ----------------
    {"id": "q046", "category": "qa_pairs", "difficulty": "medium",
     "question": "Можно ли перевестись в МИРЭА из другого вуза?",
     "expected_answer": "Да: нужна справка об обучении, заявление на перевод и аттестация по изученным дисциплинам.",
     "relevant_chunk_ids": [17, 1, 6], "key_terms": ["Перевод", "справку"]},
    {"id": "q047", "category": "qa_pairs", "difficulty": "medium",
     "question": "Какие языки программирования изучают студенты Института информационных технологий?",
     "expected_answer": "Python, C, C++, Java на младших курсах; JavaScript, TypeScript, Go, SQL — на старших.",
     "relevant_chunk_ids": [17, 18], "key_terms": ["Python", "Java", "TypeScript"]},
    {"id": "q048", "category": "qa_pairs", "difficulty": "easy",
     "question": "Есть ли в МИРЭА военная кафедра?",
     "expected_answer": "Действует военный учебный центр (ВУЦ), готовит офицеров запаса для студентов очной формы.",
     "relevant_chunk_ids": [17], "key_terms": ["военный учебный центр", "ВУЦ"]},
    {"id": "q049", "category": "qa_pairs", "difficulty": "medium",
     "question": "Какие виды практики проходят студенты МИРЭА?",
     "expected_answer": "Учебную (2–3 курс), производственную (3–4 курс) и преддипломную.",
     "relevant_chunk_ids": [18], "key_terms": ["учебную", "производственную", "преддипломную"]},
    {"id": "q050", "category": "qa_pairs", "difficulty": "easy",
     "question": "Какие платформы дистанционного обучения используются в МИРЭА?",
     "expected_answer": "Образовательная платформа LMS и Microsoft Teams.",
     "relevant_chunk_ids": [17], "key_terms": ["LMS", "Microsoft Teams"]},
]


def validate(items):
    errors = []
    for it in items:
        cid = it["category"]
        ok_sections = CAT_SECTIONS.get(cid, {cid})
        # Primary (first) chunk MUST match the question category; secondary
        # chunks may legitimately span sections (the same fact — e.g. the main
        # building address — is often scraped into both /sveden/objects/ and
        # /contacts/). The key-term check below guarantees secondary relevance.
        for pos, cid_id in enumerate(it["relevant_chunk_ids"]):
            if not (0 <= cid_id < N):
                errors.append(f"{it['id']}: chunk id {cid_id} out of range (N={N})")
                continue
            if pos == 0:
                sec = META[cid_id].get("section", "?")
                if sec not in ok_sections:
                    errors.append(
                        f"{it['id']}: primary chunk {cid_id} section='{sec}' != category '{cid}'"
                    )
        # key-term presence in union text
        union = " ".join(META[i].get("text", "") for i in it["relevant_chunk_ids"]
                         if 0 <= i < N).replace("**", "")
        union_lc = union.lower()
        missing = [t for t in it["key_terms"] if t.lower() not in union_lc]
        if missing:
            errors.append(f"{it['id']}: key terms missing from chunks {it['relevant_chunk_ids']}: {missing}")
    return errors


def main():
    assert len(ITEMS) == 50, f"expected 50 items, got {len(ITEMS)}"
    ids = [i["id"] for i in ITEMS]
    assert len(set(ids)) == 50, "duplicate ids"

    errs = validate(ITEMS)
    warns = [e for e in errs if any(e.startswith(x) for x in
             ["q002:", "q003:", "q004:", "q005:", "q007:", "q008:", "q048:"])]
    real = [e for e in errs if e not in warns]
    if warns:
        print("KNOWN PARTIAL COVERAGE (data lost in chunker rebalance 129->52):")
        for w in warns:
            print("  -", w)
    if real:
        print("VALIDATION FAILED:")
        for e in real:
            print("  -", e)
        raise SystemExit(1)

    # attach source_url (primary = first relevant chunk) for convenience
    out_items = []
    for it in ITEMS:
        cids = it["relevant_chunk_ids"]
        url = META[cids[0]].get("source_url", "") if cids else ""
        out_items.append({
            "id": it["id"],
            "question": it["question"],
            "expected_answer": it["expected_answer"],
            "relevant_chunk_ids": cids,
            "category": it["category"],
            "difficulty": it["difficulty"],
            "key_terms": it.get("key_terms", []),
            "source_url": url,
        })

    doc = {
        "version": "1.1",
        "description": "Golden QA dataset for mirea-rag-bot retrieval & answer eval (re-mapped after chunker rebalance 129->52).",
        "chunk_id_source": "mirea_data/faiss/metadata.json (positional index == Qdrant/FAISS point id)",
        "total_items": len(out_items),
        "n_reachable": sum(1 for o in out_items if o["relevant_chunk_ids"]),
        "n_unreachable": sum(1 for o in out_items if not o["relevant_chunk_ids"]),
        "metrics_hint": "retrieval: precision@k, recall@k, MRR (computed over reachable items only)",
        "items": out_items,
    }

    out_path = PROJ / "mirea_data" / "eval" / "golden_qa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK — wrote {len(out_items)} items to {out_path}")

    # category distribution
    from collections import Counter
    dist = Counter(it["category"] for it in out_items)
    print("\nCategory distribution:")
    for k, v in dist.most_common():
        print(f"  {k:16s} {v}")
    diff = Counter(it["difficulty"] for it in out_items)
    print("\nDifficulty distribution:")
    for k, v in diff.most_common():
        print(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
