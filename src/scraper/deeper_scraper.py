"""
Deeper scraper for mirea.ru — grants, dormitories, employment,
extracurricular, announcements.

Scrapes 5 additional sections of www.mirea.ru using requests + BeautifulSoup.
Handles DDoS-Guard WAF with session persistence and exponential backoff.
If the site is unreachable, generates a demo dataset for pipeline testing.

Usage:
    python -m src.scraper.deeper_scraper
"""

import json
import os
import re
import sys
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.scraper._binary_guard import is_binary_url, is_binary_response, has_binary_content

BASE_URL = "https://www.mirea.ru"
OUTPUT_FILE = os.path.join(config.DATA_RAW_DIR, "deeper.json")

RATE_LIMIT_SEC = 1.5
MAX_RETRIES = 3
BACKOFF_BASE = 2
_CONNECT_TIMEOUT = 8
_READ_TIMEOUT = 10

NAV_CLASSES = {"nav", "footer", "sidebar", "menu", "header", "breadcrumb",
               "footer-wrapper", "header-wrapper", "navigation", "mobile-menu",
               "top-panel", "social", "copyright"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}

SECTIONS = {
    "grants": {
        "url": f"{BASE_URL}/sveden/grants/",
        "label": "Стипендии и социальные выплаты",
    },
    "dormitories": {
        "url": f"{BASE_URL}/about/infrastructure/obshchezhitiya/",
        "label": "Общежития",
    },
    "employment": {
        "url": f"{BASE_URL}/about/employment-of-graduates/",
        "label": "Трудоустройство выпускников",
    },
    "extracurricular": {
        "url": f"{BASE_URL}/about/vneuchebnaya-rabota/",
        "label": "Внеучебная работа",
    },
    "announcements": {
        "url": f"{BASE_URL}/ads/",
        "label": "Объявления",
    },
}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _make_session() -> requests.Session:
    from curl_cffi import requests as cffi_requests
    session = cffi_requests.Session(impersonate="chrome")
    session.headers.update(HEADERS)
    return session


def _fetch(session: requests.Session, url: str, retries: int = MAX_RETRIES
           ) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
            if resp.status_code in (403, 503, 502) and attempt < retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt+1}] {resp.status_code} for {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            return resp
        except requests.ConnectionError:
            print(f"  [SKIP] connection refused: {url}")
            return None
        except requests.RequestException as exc:
            if attempt < retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt+1}] {type(exc).__name__} for {url}, waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [FAILED] {url}: {type(exc).__name__}")
    return None


def _clean_text(html_or_text: str) -> str:
    text = _TAG_RE.sub(" ", html_or_text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag_name in ("nav", "footer", "header", "aside"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag in soup.find_all(True):
        if not hasattr(tag, "attrs") or tag.attrs is None:
            continue
        raw_classes = tag.get("class")
        classes = " ".join(raw_classes) if isinstance(raw_classes, list) else str(raw_classes or "")
        tag_id = tag.get("id") or ""
        combined = f"{classes} {tag_id}".lower()
        if any(nc in combined for nc in NAV_CLASSES):
            tag.decompose()


def _parse_table(table) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""

    header_cells = rows[0].find_all(["th", "td"])
    headers = [_clean_text(cell.get_text()) for cell in header_cells]

    parts = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        cell_texts = [_clean_text(cell.get_text()) for cell in cells]
        if not any(cell_texts):
            continue
        row_parts = []
        for i, val in enumerate(cell_texts):
            h = headers[i] if i < len(headers) else f"Col{i+1}"
            if val:
                row_parts.append(f"{h}: {val}")
        if row_parts:
            parts.append(" | ".join(row_parts))
    return "\n".join(parts)


def _extract_content(soup: BeautifulSoup, url: str, section: str,
                     subsection: str, items: list[dict[str, str]]) -> None:
    _remove_noise(soup)

    title_tag = soup.find("h1")
    title = _clean_text(title_tag.get_text()) if title_tag else ""

    content_area = (
        soup.find("div", class_=re.compile(r"content|article|detail|main.*text", re.I))
        or soup.find("main")
        or soup.find("div", role="main")
        or soup.find("section")
        or soup.body
    )
    if content_area is None:
        return

    tables = content_area.find_all("table")

    blocks = content_area.find_all(["p", "li", "h2", "h3", "h4", "blockquote"])
    text_parts: list[str] = []
    for block in blocks:
        txt = _clean_text(block.get_text())
        if len(txt) < 10:
            continue
        text_parts.append(txt)

    table_parts: list[str] = []
    for table in tables:
        table_text = _parse_table(table)
        if table_text.strip():
            table_parts.append(table_text)

    combined = "\n".join(text_parts)
    if table_parts:
        combined += "\n\n" + "\n\n".join(table_parts)

    combined = combined.strip()
    if not combined or len(combined) < 30:
        return

    items.append({
        "title": title,
        "text": combined,
        "url": url,
        "section": section,
        "subsection": subsection,
    })


def _extract_links_on_page(soup: BeautifulSoup, base: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href: str = str(a_tag["href"])
        text = _clean_text(a_tag.get_text())
        if not text or len(text) < 3:
            continue
        full = href if href.startswith("http") else base.rstrip("/") + href
        if full not in seen and base in full:
            seen.add(full)
            links.append((full, text))
    return links


def scrape_deeper() -> list[dict[str, str]]:
    session = _make_session()
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    print("[*] Warming up session on main page...")
    warmup = _fetch(session, BASE_URL + "/", retries=1)
    time.sleep(RATE_LIMIT_SEC)

    site_reachable = warmup is not None and warmup.status_code == 200

    if not site_reachable:
        print("[!] Site unreachable — skipping live scrape")
        return []

    for section_key, section_info in SECTIONS.items():
        url = section_info["url"]
        subsection = section_info["label"]
        print(f"[*] Scraping {url}")
        resp = _fetch(session, url)
        if resp and resp.status_code == 200:
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "lxml")
            _extract_content(soup, url, section_key, subsection, items)
            seen_urls.add(url)

            child_links = _extract_links_on_page(soup, BASE_URL)
            for child_url, child_text in child_links[:10]:
                if child_url in seen_urls:
                    continue
                seen_urls.add(child_url)
                if is_binary_url(child_url):
                    continue

                print(f"  [+] {child_url}")
                child_resp = _fetch(session, child_url)
                if child_resp and child_resp.status_code == 200:
                    if is_binary_response(child_resp):
                        continue
                    child_resp.encoding = "utf-8"
                    if has_binary_content(child_resp.text):
                        print(f"  [skip] binary content detected: {child_url}")
                        continue
                    child_soup = BeautifulSoup(child_resp.text, "lxml")
                    _extract_content(child_soup, child_url, section_key,
                                     child_text[:60], items)
                time.sleep(RATE_LIMIT_SEC)
        else:
            print(f"  [skip] {url} — not reachable")

        time.sleep(RATE_LIMIT_SEC)

    print(f"[*] Scraped {len(items)} items from mirea.ru")
    return items


def _demo_dataset() -> list[dict[str, str]]:
    return [
        # ── grants (7 items) ──
        {
            "title": "Государственная академическая стипендия",
            "text": "Государственная академическая стипендия назначается студентам очной формы "
                    "обучения, обучающимся за счёт бюджетных ассигнований, при отсутствии "
                    "академической задолженности. Размер стипендии составляет 4 826 рублей в месяц "
                    "для бакалавров и специалитета, 5 085 рублей — для магистрантов. Стипендия "
                    "выплачивается ежемесячно. Назначение производится приказом ректора по "
                    "представлению стипендиальной комиссии института.",
            "url": "https://www.mirea.ru/sveden/grants/",
            "section": "grants",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Повышенная государственная академическая стипендия (ПГАС)",
            "text": "Повышенная государственная академическая стипендия назначается студентам "
                    "за особые достижения в учебной, научно-исследовательской, общественной, "
                    "культурно-творческой или спортивной деятельности. Размер ПГАС составляет "
                    "от 8 000 до 12 000 рублей в месяц. Для получения ПГАС студент должен "
                    "предоставить портфолио достижений и пройти конкурсный отбор.",
            "url": "https://www.mirea.ru/sveden/grants/",
            "section": "grants",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Стипендия Президента Российской Федерации",
            "text": "Стипендия Президента РФ назначается студентам и аспирантам, проявившим "
                    "выдающиеся способности в учебной и научной деятельности. Размер стипендии "
                    "для студентов — 7 000 рублей, для аспирантов — 14 000 рублей в месяц. "
                    "Кандидаты выдвигаются Учёным советом университета. Отбор проводится "
                    "Министерством науки и высшего образования РФ.",
            "url": "https://www.mirea.ru/sveden/grants/",
            "section": "grants",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Социальная стипендия",
            "text": "Государственная социальная стипендия назначается студентам, нуждающимся "
                    "в социальной помощи: детям-сиротам, детям, оставшимся без попечения "
                    "родителей, инвалидам I и II групп, ветеранам боевых действий, студентам "
                    "из малоимущих семей. Размер социальной стипендии составляет не менее "
                    "3 617 рублей в месяц. Назначается с даты предоставления подтверждающих "
                    "документов.",
            "url": "https://www.mirea.ru/sveden/grants/",
            "section": "grants",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Именная стипендия Университета",
            "text": "Именные стипендии РТУ МИРЭА учреждены в память выдающихся учёных и "
                    "преподавателей университета. Назначаются студентам 3–4 курсов, "
                    "демонстрирующим высокие академические результаты и активно участвующим "
                    "в научно-исследовательской работе. Размер именной стипендии — "
                    "6 000 рублей в месяц. Конкурс проводится ежегодно в сентябре.",
            "url": "https://www.mirea.ru/sveden/grants/",
            "section": "grants",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Стипендия компании-партнёра",
            "text": "РТУ МИРЭА совместно с компаниями-партнёрами (Яндекс, Касперский, "
                    "Сбер, VK, Huawei) учреждает корпоративные стипендии для студентов "
                    "IT-направлений. Размер стипендии — от 15 000 до 25 000 рублей в месяц. "
                    "Студенты-стипендиаты получают приоритетное право на стажировку и "
                    "трудоустройство в компанию-партнёра.",
            "url": "https://www.mirea.ru/sveden/grants/",
            "section": "grants",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Материальная помощь студентам",
            "text": "Студенты бюджетной формы обучения имеют право на получение материальной "
                    "помощи в размере до 12 000 рублей. Основания: рождение ребёнка, смерть "
                    "близкого родственника, повреждение имущества в результате стихийного "
                    "бедствия, тяжёлое материальное положение. Заявление подаётся в деканат "
                    "с приложением подтверждающих документов.",
            "url": "https://www.mirea.ru/sveden/grants/",
            "section": "grants",
            "subsection": "Стипендии и социальные выплаты",
        },

        # ── dormitories (6 items) ──
        {
            "title": "Общежитие №1 (Стромынка)",
            "text": "Общежитие №1 расположено по адресу: г. Москва, ул. Стромынка, д. 11. "
                    "Предназначено для иногородних студентов очной формы обучения. "
                    "В общежитии 250 мест, комнаты на 2–3 человека. Имеется прачечная, "
                    "кухня на каждом этаже, комната самоподготовки, спортивная площадка. "
                    "Стоимость проживания — 1 200 рублей в месяц. Заселение производится "
                    "на основании заявления и справки о регистрации.",
            "url": "https://www.mirea.ru/about/infrastructure/obshchezhitiya/",
            "section": "dormitories",
            "subsection": "Общежития",
        },
        {
            "title": "Общежитие №2 (Вернадского)",
            "text": "Общежитие №2 расположено по адресу: г. Москва, проспект Вернадского, "
                    "д. 86. Блочный тип, 400 мест. Комнаты на 2 человека с санузлом в блоке. "
                    "Оборудовано Wi-Fi, душевыми, постирочной, спортивным залом. Рядом станция "
                    "метро «Юго-Западная». Стоимость проживания — 1 300 рублей в месяц. "
                    "Приоритет при заселении — студенты 1 курса.",
            "url": "https://www.mirea.ru/about/infrastructure/obshchezhitiya/",
            "section": "dormitories",
            "subsection": "Общежития",
        },
        {
            "title": "Общежитие №3 (Малая Филёвская)",
            "text": "Общежитие №3 расположено по адресу: г. Москва, ул. Малая Филёвская, "
                    "д. 4. Коридорный тип, 180 мест. Комнаты на 2–3 человека. На этаже — "
                    "общая кухня, душевые, санузлы. Имеется комната отдыха и библиотека. "
                    "Стоимость проживания — 1 100 рублей в месяц.",
            "url": "https://www.mirea.ru/about/infrastructure/obshchezhitiya/",
            "section": "dormitories",
            "subsection": "Общежития",
        },
        {
            "title": "Общежитие №4 (Новопеределкино)",
            "text": "Общежитие №4 расположено в районе Ново-Переделкино, ул. Шолохова, д. 20. "
                    "Современное здание 2018 года постройки, 350 мест. Комнаты на 1–2 человека, "
                    "душ и туалет в номере. Оснащено лифтами, видеонаблюдением, электронными "
                    "пропусками. На первом этаже — кафе, медпункт, комнаты для самоподготовки. "
                    "Стоимость — 1 500 рублей в месяц.",
            "url": "https://www.mirea.ru/about/infrastructure/obshchezhitiya/",
            "section": "dormitories",
            "subsection": "Общежития",
        },
        {
            "title": "Правила проживания в общежитиях",
            "text": "Проживающие в общежитии обязаны соблюдать правила внутреннего распорядка: "
                    "соблюдать тишину с 23:00 до 07:00, поддерживать чистоту в жилых помещениях "
                    "и местах общего пользования, не допускать посторонних лиц после 22:00. "
                    "Курение на территории общежития запрещено. За нарушение правил предусмотрен "
                    "дисциплинарный арест до выселения.",
            "url": "https://www.mirea.ru/about/infrastructure/obshchezhitiya/",
            "section": "dormitories",
            "subsection": "Общежития",
        },
        {
            "title": "Порядок заселения в общежитие",
            "text": "Для заселения в общежитие необходимо: подать заявление в деканат, "
                    "предоставить справку о составе семьи, медицинскую справку формы 086/У, "
                    "фотографии 3×4 (6 шт.), копию паспорта. Заселение проводится в августе "
                    "перед началом учебного года. Иногородние студенты 1 курса обеспечиваются "
                    "местами в общежитии в приоритетном порядке.",
            "url": "https://www.mirea.ru/about/infrastructure/obshchezhitiya/",
            "section": "dormitories",
            "subsection": "Общежития",
        },

        # ── employment (6 items) ──
        {
            "title": "Центр карьеры РТУ МИРЭА",
            "text": "Центр карьеры РТУ МИРЭА оказывает содействие студентам и выпускникам "
                    "в трудоустройстве. Услуги: консультации по составлению резюме, "
                    "подготовка к собеседованиям, организация ярмарок вакансий, проведение "
                    "дней карьеры с участием ведущих работодателей. Центр карьеры расположен "
                    "в корпусе на проспекте Вернадского, д. 78, каб. А-216.",
            "url": "https://www.mirea.ru/about/employment-of-graduates/",
            "section": "employment",
            "subsection": "Трудоустройство выпускников",
        },
        {
            "title": "Статистика трудоустройства выпускников",
            "text": "По данным мониторинга Минобрнауки, доля трудоустроенных выпускников "
                    "РТУ МИРЭА составляет 94% в течение первого года после окончания обучения. "
                    "Из них 78% работают по специальности. Средний срок поиска первого "
                    "работодателя — 1,5 месяца. 42% выпускников получают предложение о работе "
                    "ещё во время обучения на последнем курсе.",
            "url": "https://www.mirea.ru/about/employment-of-graduates/",
            "section": "employment",
            "subsection": "Трудоустройство выпускников",
        },
        {
            "title": "Партнёры по трудоустройству",
            "text": "РТУ МИРЭА сотрудничает с более чем 200 организациями-работодателями. "
                    "Основные партнёры: Яндекс, Касперский, Сбер, VK, Huawei, Ростелеком, "
                    "Росатом, Роскосмос, Ростех, МТС, Билайн, Газпром-медиа, Лаборатория "
                    "Касперского, Positive Technologies, Инфосистемы Джет. Ежегодно "
                    "подписывается более 500 договоров на практику и стажировку.",
            "url": "https://www.mirea.ru/about/employment-of-graduates/",
            "section": "employment",
            "subsection": "Трудоустройство выпускников",
        },
        {
            "title": "Программа стажировок",
            "text": "Программа стажировок РТУ МИРЭА позволяет студентам 3–4 курсов пройти "
                    "практику в ведущих компаниях IT-отрасли. Длительность стажировки — "
                    "от 2 до 6 месяцев. Студенты получают реальный опыт работы над "
                    "проектами, менторскую поддержку и возможность последующего "
                    "трудоустройства. Средняя оплата стажировки — 30 000–50 000 рублей в месяц.",
            "url": "https://www.mirea.ru/about/employment-of-graduates/",
            "section": "employment",
            "subsection": "Трудоустройство выпускников",
        },
        {
            "title": "Ярмарка вакансий",
            "text": "Ежегодная ярмарка вакансий РТУ МИРЭА проводится в октябре и марте. "
                    "Участвуют более 50 компаний-работодателей. Студенты могут пройти "
                    "собеседование на месте, оставить резюме и узнать о стажировках. "
                    "В 2024 году на ярмарке было представлено более 1 200 вакансий "
                    "для студентов и выпускников.",
            "url": "https://www.mirea.ru/about/employment-of-graduates/",
            "section": "employment",
            "subsection": "Трудоустройство выпускников",
        },
        {
            "title": "Трудоустройство иностранных выпускников",
            "text": "Иностранные выпускники РТУ МИРЭА имеют право на трудоустройство в "
                    "Российской Федерации в упрощённом порядке. Университет оказывает "
                    "содействие в оформлении патента на работу и взаимодействии с "
                    "Федеральной миграционной службой. Для выпускников из стран ЕАЭС "
                    "доступен упрощённый режим трудоустройства без патента.",
            "url": "https://www.mirea.ru/about/employment-of-graduates/",
            "section": "employment",
            "subsection": "Трудоустройство выпускников",
        },

        # ── extracurricular (6 items) ──
        {
            "title": "Студенческий совет РТУ МИРЭА",
            "text": "Студенческий совет — высший представительный орган студенческого "
                    "самоуправления университета. В состав входят представители всех "
                    "институтов и факультетов. Деятельность: организация культурно-массовых "
                    "мероприятий, защита прав и интересов студентов, участие в управлении "
                    "университетом, развитие студенческих инициатив. Заседания проводятся "
                    "еженедельно по средам в актовом зале главного корпуса.",
            "url": "https://www.mirea.ru/about/vneuchebnaya-rabota/",
            "section": "extracurricular",
            "subsection": "Внеучебная работа",
        },
        {
            "title": "Спортивные секции и клубы",
            "text": "В РТУ МИРЭА действуют более 30 спортивных секций: мини-футбол, "
                    "баскетбол, волейбол, настольный теннис, шахматы, лёгкая атлетика, "
                    "плавание, единоборства (каратэ, дзюдо, бокс), фитнес, йога. "
                    "Сборные команды университета участвуют в соревнованиях Московского "
                    "межвузовского округа и Всероссийской универсиады.",
            "url": "https://www.mirea.ru/about/vneuchebnaya-rabota/",
            "section": "extracurricular",
            "subsection": "Внеучебная работа",
        },
        {
            "title": "Творческие объединения",
            "text": "В университете работают творческие коллективы: театральная студия "
                    "«Этюд», вокальная студия «Гармония», танцевальный коллектив «Ритм», "
                    "КВН-команда «МИРЭА-сборная», литературный клуб «Слово», фотоклуб "
                    "«Объектив». Коллективы регулярно участвуют в городских и всероссийских "
                    "конкурсах и фестивалях.",
            "url": "https://www.mirea.ru/about/vneuchebnaya-rabota/",
            "section": "extracurricular",
            "subsection": "Внеучебная работа",
        },
        {
            "title": "Волонтёрский центр",
            "text": "Волонтёрский центр РТУ МИРЭА координирует добровольческую деятельность "
                    "студентов. Направления: помощь ветеранам и пожилым людям, благоустройство "
                    "территории, организация благотворительных акций, участие в городских "
                    "волонтёрских проектах. В 2024 году волонтёрами университета было "
                    "реализовано более 40 проектов, охвачено 800+ участников.",
            "url": "https://www.mirea.ru/about/vneuchebnaya-rabota/",
            "section": "extracurricular",
            "subsection": "Внеучебная работа",
        },
        {
            "title": "Научные студенческие общества (НСО)",
            "text": "Научные студенческие общества действуют при каждом институте РТУ МИРЭА. "
                    "Студенты участвуют в научно-исследовательских проектах, публикуют статьи "
                    "в рецензируемых журналах, выступают на конференциях. Ежегодно проводится "
                    "Межвузовская научно-практическая конференция студентов, аспирантов и "
                    "молодых учёных с участием более 500 докладчиков.",
            "url": "https://www.mirea.ru/about/vneuchebnaya-rabota/",
            "section": "extracurricular",
            "subsection": "Внеучебная работа",
        },
        {
            "title": "Студенческие медиа",
            "text": "Студенческие медиа РТУ МИРЭА включают: студенческое телевидение "
                    "«МИРЭА-TV», радиостанцию, газету «Технопарк», интернет-портал и группы "
                    "в социальных сетях. Объединяют более 100 студентов-журналистов, "
                    "операторов, дизайнеров и SMM-специалистов. Медиа освещают жизнь "
                    "университета, интервьюируют преподавателей и приглашённых экспертов.",
            "url": "https://www.mirea.ru/about/vneuchebnaya-rabota/",
            "section": "extracurricular",
            "subsection": "Внеучебная работа",
        },

        # ── announcements (6 items) ──
        {
            "title": "Приёмная кампания 2026",
            "text": "Началась приёмная кампания 2026 года. Приём документов осуществляется "
                    "в приёмной комиссии (пр. Вернадского, д. 78, каб. А-101) и через "
                    "личный кабинет абитуриента на сайте университета. Сроки приёма: "
                    "бакалавриат/специалитет — до 25 июля 2026, магистратура — до "
                    "15 августа 2026. Необходимые документы: паспорт, СНИЛС, аттестат "
                    "или диплом, результаты ЕГЭ.",
            "url": "https://www.mirea.ru/ads/",
            "section": "announcements",
            "subsection": "Объявления",
        },
        {
            "title": "День открытых дверей",
            "text": "День открытых дверей РТУ МИРЭА состоится 15 марта 2026 года в главном "
                    "корпусе (ул. Стромынка, д. 20). Программа: презентация направлений "
                    "подготовки, экскурсия по лабораториям, консультации с преподавателями "
                    "и студентами, мастер-класс по программированию. Начало в 10:00, "
                    "вход свободный. Регистрация на сайте университета.",
            "url": "https://www.mirea.ru/ads/",
            "section": "announcements",
            "subsection": "Объявления",
        },
        {
            "title": "Конкурс проектов «Цифровой прорыв»",
            "text": "РТУ МИРЭА приглашает студентов принять участие во Всероссийском "
                    "конкурсе проектов «Цифровой прорыв». Номинации: искусственный интеллект, "
                    "анализ данных, кибербезопасность, интернет вещей. Призовой фонд — "
                    "2 000 000 рублей. Регистрация команд до 1 апреля 2026 года. "
                    "Финал состоится в мае 2026 года в Москве.",
            "url": "https://www.mirea.ru/ads/",
            "section": "announcements",
            "subsection": "Объявления",
        },
        {
            "title": "Гранты на научно-исследовательские проекты",
            "text": "Объявлен конкурс на предоставление грантов студентам для выполнения "
                    "научно-исследовательских проектов в 2026 году. Размер гранта — "
                    "до 200 000 рублей на один проект. Срок реализации — 6 месяцев. "
                    "Заявки принимаются до 15 марта 2026 года. Тематика: информационные "
                    "технологии, кибербезопасность, радиотехника, химические технологии.",
            "url": "https://www.mirea.ru/ads/",
            "section": "announcements",
            "subsection": "Объявления",
        },
        {
            "title": "Студенческая олимпиада по программированию",
            "text": "Внутриуниверситетская олимпиада по программированию состоится "
                    "20 апреля 2026 года. Участие открытое для всех студентов 1–4 курсов. "
                    "Задачи по алгоритмам, структурам данных и дискретной математике. "
                    "Победители получают повышенную стипендию и право представлять "
                    "университет на Всероссийской олимпиаде. Регистрация на сайте кафедры ПИ.",
            "url": "https://www.mirea.ru/ads/",
            "section": "announcements",
            "subsection": "Объявления",
        },
        {
            "title": "Изменение расписания занятий",
            "text": "В связи с проведением Всероссийской научно-практической конференции "
                    "занятия 25–27 марта 2026 года для студентов 1–2 курсов переводятся "
                    "в дистанционный формат. Ссылки на онлайн-занятия будут направлены "
                    "на корпоративную почту. Расписание в электронном журнале обновлено. "
                    "При возникновении вопросов обращаться в деканат.",
            "url": "https://www.mirea.ru/ads/",
            "section": "announcements",
            "subsection": "Объявления",
        },
    ]


def save_items(items: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {len(items)} items to {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("MIREA Deeper Scraper")
    print("=" * 60)

    items = scrape_deeper()

    if not items:
        print("[!] mirea.ru unreachable or no data scraped — FAILING (no silent demo fallback)")
        sys.exit(1)

    save_items(items, OUTPUT_FILE)
