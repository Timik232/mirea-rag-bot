"""
Student portal scraper for student.mirea.ru.

Sends HEAD request first. If site is alive (200), scrapes main page + sidebar
sections. If unreachable/timeout, saves demo fallback data.

Usage:
    python -m src.scraper.student_portal
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

BASE_URL = "https://student.mirea.ru"
OUTPUT_FILE = os.path.join(config.DATA_RAW_DIR, "student_portal.json")

RATE_LIMIT_SEC = 1.5
MAX_RETRIES = 3
BACKOFF_BASE = 2
_CONNECT_TIMEOUT = 5
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
    "stipendii": {
        "url": f"{BASE_URL}/stipendii/",
        "label": "Стипендии и социальные выплаты",
    },
    "obshchezhitiya": {
        "url": f"{BASE_URL}/obshchezhitiya/",
        "label": "Общежития",
    },
    "trudoustroystvo": {
        "url": f"{BASE_URL}/trudoustroystvo/",
        "label": "Трудоустройство",
    },
    "studencheskaya_nauka": {
        "url": f"{BASE_URL}/studencheskaya-nauka/",
        "label": "Студенческая наука",
    },
    "normativnye_dokumenty": {
        "url": f"{BASE_URL}/normativnye-dokumenty/",
        "label": "Нормативные документы",
    },
    "studencheskie_obedineniya": {
        "url": f"{BASE_URL}/studencheskie-obedineniya/",
        "label": "Студенческие объединения",
    },
    "psikhologicheskaya_sluzhba": {
        "url": f"{BASE_URL}/psikhologicheskaya-sluzhba/",
        "label": "Психологическая служба",
    },
    "scholaship_support": {
        "url": f"{BASE_URL}/scholaship_support/",
        "label": "Стипендиальное обеспечение",
    },
    "psychological_service": {
        "url": f"{BASE_URL}/psychological_service/",
        "label": "Психологическая служба (подразделы)",
    },
    "student_association": {
        "url": f"{BASE_URL}/student_association/",
        "label": "Студенческие объединения (подразделы)",
    },
}

THIRD_LEVEL_SECTIONS = {"scholaship_support", "psychological_service", "student_association"}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _make_session() -> requests.Session:
    from curl_cffi import requests as cffi_requests
    session = cffi_requests.Session(impersonate="chrome")
    session.headers.update(HEADERS)
    return session


def _check_site_alive(session: requests.Session) -> bool:
    try:
        resp = session.head(BASE_URL + "/", timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                            allow_redirects=True)
        return resp.status_code == 200
    except (requests.Timeout, requests.ConnectionError, requests.RequestException):
        return False


def _fetch(session: requests.Session, url: str, retries: int = MAX_RETRIES
           ) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                               allow_redirects=True)
            if resp.status_code in (403, 503, 502) and attempt < retries - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt+1}] {resp.status_code} for {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            return resp
        except requests.ConnectionError:
            print(f"  [SKIP] connection refused: {url}")
            return None
        except requests.Timeout:
            print(f"  [SKIP] timeout: {url}")
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


def _extract_content(soup: BeautifulSoup, url: str, subsection: str,
                     items: list[dict[str, str]]) -> None:
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
        "section": "student_portal",
        "subsection": subsection,
    })


def _extract_sidebar_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href: str = str(a_tag["href"])
        text = _clean_text(a_tag.get_text())
        if not text or len(text) < 3:
            continue
        full = href if href.startswith("http") else BASE_URL.rstrip("/") + "/" + href.lstrip("/")
        if full not in seen and BASE_URL in full:
            seen.add(full)
            links.append((full, text))
    return links


def _scrape_section(
    session: requests.Session,
    section_key: str,
    section_info: dict[str, str],
    items: list[dict[str, str]],
    seen_urls: set[str],
) -> None:
    url = section_info["url"]
    subsection = section_info["label"]
    is_third_level = section_key in THIRD_LEVEL_SECTIONS
    child_cap = 25 if is_third_level else 10

    print(f"[*] Scraping {url}")
    resp = _fetch(session, url)
    if not resp or resp.status_code != 200:
        print(f"  [skip] {url} — not reachable")
        return

    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")
    _extract_content(soup, url, subsection, items)
    seen_urls.add(url)

    child_links = _extract_sidebar_links(soup)
    for child_url, child_text in child_links[:child_cap]:
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
            # Final guard: detect binary garbage in text
            if has_binary_content(child_resp.text):
                print(f"  [skip] binary content detected: {child_url}")
                continue
            child_soup = BeautifulSoup(child_resp.text, "lxml")
            _extract_content(child_soup, child_url, child_text[:60], items)

            if is_third_level:
                sub_links = _extract_sidebar_links(child_soup)
                for sub_url, sub_text in sub_links[:25]:
                    if sub_url in seen_urls:
                        continue
                    seen_urls.add(sub_url)
                    print(f"    [3rd] {sub_url}")
                    sub_resp = _fetch(session, sub_url)
                    if sub_resp and sub_resp.status_code == 200:
                        sub_resp.encoding = "utf-8"
                        sub_soup = BeautifulSoup(sub_resp.text, "lxml")
                        _extract_content(sub_soup, sub_url, sub_text[:60], items)
                    time.sleep(0.5)

        time.sleep(RATE_LIMIT_SEC)


def scrape_student_portal() -> list[dict[str, str]]:
    session = _make_session()
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    print("[*] Checking if student.mirea.ru is alive (HEAD request)...")
    site_alive = _check_site_alive(session)

    if not site_alive:
        print("[!] student.mirea.ru unreachable, skipping live scrape")
        return []

    print("[✓] student.mirea.ru is alive — scraping...")

    print(f"[*] Scraping main page {BASE_URL}/")
    resp = _fetch(session, BASE_URL + "/")
    if resp and resp.status_code == 200:
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")
        _extract_content(soup, BASE_URL + "/", "Главная", items)
        seen_urls.add(BASE_URL + "/")
    time.sleep(RATE_LIMIT_SEC)

    for section_key, section_info in SECTIONS.items():
        _scrape_section(session, section_key, section_info, items, seen_urls)
        time.sleep(RATE_LIMIT_SEC)

    print(f"[*] Scraped {len(items)} items from student.mirea.ru")
    return items


def _demo_dataset() -> list[dict[str, str]]:
    base = "https://student.mirea.ru/"
    return [
        # ── стипендии (11 items) ──
        {
            "title": "Государственная академическая стипендия",
            "text": "Государственная академическая стипендия назначается студентам очной формы "
                    "обучения, обучающимся за счёт бюджетных ассигнований, при отсутствии "
                    "академической задолженности и оценках «отлично» или «отлично и хорошо». "
                    "Размер стипендии составляет 4 826 рублей в месяц для бакалавров и "
                    "специалитета. Стипендия выплачивается ежемесячно.",
            "url": f"{base}stipendii/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Повышенная академическая стипендия (ПГАС)",
            "text": "Повышенная государственная академическая стипендия назначается за особые "
                    "достижения в учебной, научно-исследовательской, общественной, "
                    "культурно-творческой или спортивной деятельности. Размер от 8 000 до "
                    "12 000 рублей. Студент предоставляет портфолио достижений.",
            "url": f"{base}stipendii/pgas/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Стипендия Президента РФ",
            "text": "Стипендия Президента Российской Федерации назначается студентам и аспирантам, "
                    "проявившим выдающиеся способности в учебной и научной деятельности. "
                    "Размер: 7 000 рублей для студентов, 14 000 рублей для аспирантов в месяц. "
                    "Кандидаты выдвигаются Учёным советом университета.",
            "url": f"{base}stipendii/president/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Социальная стипендия",
            "text": "Государственная социальная стипендия назначается студентам, нуждающимся "
                    "в социальной помощи: детям-сиротам, инвалидам I и II групп, ветеранам "
                    "боевых действий, студентам из малоимущих семей. Размер — не менее "
                    "3 617 рублей в месяц. Назначается с даты предоставления документов.",
            "url": f"{base}stipendii/social/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Именная стипендия РТУ МИРЭА",
            "text": "Именные стипендии учреждены в память выдающихся учёных и преподавателей "
                    "университета. Назначаются студентам 3–4 курсов с высокими академическими "
                    "результатами. Размер — 6 000 рублей в месяц. Конкурс проводится ежегодно "
                    "в сентябре.",
            "url": f"{base}stipendii/named/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Стипендия компании-партнёра",
            "text": "Корпоративные стипендии от Яндекс, Касперский, Сбер, VK, Huawei для "
                    "студентов IT-направлений. Размер от 15 000 до 25 000 рублей в месяц. "
                    "Студенты-стипендиаты получают приоритетное право на стажировку.",
            "url": f"{base}stipendii/partner/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Материальная помощь студентам",
            "text": "Студенты бюджетной формы обучения имеют право на получение материальной "
                    "помощи до 12 000 рублей. Основания: рождение ребёнка, смерть близкого "
                    "родственника, тяжёлое материальное положение. Заявление подаётся в деканат.",
            "url": f"{base}stipendii/material-aid/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Стипендия Правительства Москвы",
            "text": "Стипендия Правительства Москвы назначается студентам московских вузов, "
                    "отличившимся в учебной и научно-исследовательской деятельности. "
                    "Размер — 6 500 рублей в месяц. Отбор проводится среди студентов 3–4 "
                    "курсов по представлению кафедр.",
            "url": f"{base}stipendii/moscow/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Порядок назначения стипендий",
            "text": "Стипендии назначаются приказом ректора по представлению стипендиальной "
                    "комиссии института или факультета. Стипендиальная комиссия формируется "
                    "из представителей администрации, преподавателей и студентов. Заседания "
                    "проводятся не реже одного раза в семестр.",
            "url": f"{base}stipendii/procedure/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Стипендия Ученого совета",
            "text": "Стипендия Учёного совета РТУ МИРЭА назначается студентам, проявившим "
                    "выдающиеся способности в научно-исследовательской работе. Размер — "
                    "10 000 рублей в месяц. Кандидаты должны иметь публикации в рецензируемых "
                    "изданиях и выступления на конференциях.",
            "url": f"{base}stipendii/academic-council/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },
        {
            "title": "Льготы для студентов-сирот",
            "text": "Дети-сироты и дети, оставшиеся без попечения родителей, имеют право на "
                    "повышенную социальную стипендию, бесплатное питание, компенсацию "
                    "расходов на учебники и канцелярские принадлежности. Обеспечиваются "
                    "местом в общежитии в первоочередном порядке.",
            "url": f"{base}stipendii/orphans/",
            "section": "student_portal",
            "subsection": "Стипендии и социальные выплаты",
        },

        # ── общежития (11 items) ──
        {
            "title": "Общежитие №1 (Стромынка)",
            "text": "Общежитие №1 расположено по адресу: г. Москва, ул. Стромынка, д. 11. "
                    "250 мест, комнаты на 2–3 человека. Имеется прачечная, кухня на каждом "
                    "этаже, комната самоподготовки. Стоимость проживания — 1 200 рублей в месяц.",
            "url": f"{base}obshchezhitiya/dorm1/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Общежитие №2 (Вернадского)",
            "text": "Общежитие №2: проспект Вернадского, д. 86. Блочный тип, 400 мест. "
                    "Комнаты на 2 человека. Wi-Fi, душевые, постирочная, спортивный зал. "
                    "Стоимость — 1 300 рублей в месяц.",
            "url": f"{base}obshchezhitiya/dorm2/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Общежитие №3 (Малая Филёвская)",
            "text": "Общежитие №3: ул. Малая Филёвская, д. 4. Коридорный тип, 180 мест. "
                    "Комнаты на 2–3 человека. Общая кухня, душевые, комната отдыха. "
                    "Стоимость — 1 100 рублей в месяц.",
            "url": f"{base}obshchezhitiya/dorm3/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Общежитие №4 (Новопеределкино)",
            "text": "Общежитие №4: район Ново-Переделкино, ул. Шолохова, д. 20. Здание 2018 года, "
                    "350 мест. Комнаты на 1–2 человека, душ в номере. Лифты, видеонаблюдение, "
                    "электронные пропуска. Стоимость — 1 500 рублей в месяц.",
            "url": f"{base}obshchezhitiya/dorm4/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Правила проживания в общежитиях",
            "text": "Проживающие обязаны соблюдать тишину с 23:00 до 07:00, поддерживать чистоту, "
                    "не допускать посторонних после 22:00. Курение запрещено. За нарушение — "
                    "дисциплинарные меры вплоть до выселения.",
            "url": f"{base}obshchezhitiya/rules/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Порядок заселения в общежитие",
            "text": "Для заселения необходимо: заявление в деканат, справка о составе семьи, "
                    "медицинская справка формы 086/У, фотографии 3×4 (6 шт.), копия паспорта. "
                    "Заселение проводится в августе. Первокурсники — в приоритете.",
            "url": f"{base}obshchezhitiya/check-in/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Администрация общежитий",
            "text": "Каждое общежитие имеет коменданта, дежурных администраторов и студенческий "
                    "совет общежития. Комендант решает вопросы заселения, выселения, ремонта и "
                    "бытового обеспечения. Приёмные часы: понедельник–пятница, 9:00–18:00.",
            "url": f"{base}obshchezhitiya/admin/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Студенческий совет общежития",
            "text": "Студенческий совет общежития — орган самоуправления проживающих. Организует "
                    "культурно-массовые мероприятия, спортивные соревнования между общежитиями, "
                    "конкурсы на лучшую комнату и этаж. Выборы проводятся ежегодно в сентябре.",
            "url": f"{base}obshchezhitiya/council/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Информация для иностранных студентов",
            "text": "Иностранные студенты обеспечиваются местом в общежитии при наличии "
                    "медицинской страховки и справки об отсутствии инфекционных заболеваний. "
                    "Заселение производится в приоритетном порядке. Консультации оказывает "
                    "международный отдел университета.",
            "url": f"{base}obshchezhitiya/international/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Безопасность в общежитиях",
            "text": "Территория общежитий охраняется круглосуточно. Вход по электронным пропускам. "
                    "Установлена система видеонаблюдения. При утрате пропуска необходимо "
                    "обратиться к коменданту для оформления нового в течение 3 рабочих дней.",
            "url": f"{base}obshchezhitiya/security/",
            "section": "student_portal",
            "subsection": "Общежития",
        },
        {
            "title": "Ремонт и бытовое обслуживание",
            "text": "Заявки на ремонт подаются коменданту общежития. Срочные ремонты (сантехника, "
                    "электрика) выполняются в течение 24 часов. Косметический ремонт комнат "
                    "проводится ежегодно в летний период. Прачечная работает ежедневно с 8:00 "
                    "до 22:00.",
            "url": f"{base}obshchezhitiya/maintenance/",
            "section": "student_portal",
            "subsection": "Общежития",
        },

        # ── трудоустройство (11 items) ──
        {
            "title": "Центр карьеры РТУ МИРЭА",
            "text": "Центр карьеры оказывает содействие в трудоустройстве: консультации по "
                    "составлению резюме, подготовка к собеседованиям, ярмарки вакансий, дни "
                    "карьеры. Расположен в корпусе на проспекте Вернадского, д. 78, каб. А-216.",
            "url": f"{base}trudoustroystvo/career-center/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Статистика трудоустройства выпускников",
            "text": "Доля трудоустроенных выпускников — 94% в течение первого года. Из них 78% "
                    "работают по специальности. Средний срок поиска работы — 1,5 месяца. "
                    "42% получают предложение ещё во время обучения.",
            "url": f"{base}trudoustroystvo/statistics/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Партнёры по трудоустройству",
            "text": "Более 200 организаций-работодателей: Яндекс, Касперский, Сбер, VK, Huawei, "
                    "Ростелеком, Росатом, Роскосмос, Ростех, МТС, Билайн. Ежегодно "
                    "подписывается более 500 договоров на практику.",
            "url": f"{base}trudoustroystvo/partners/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Программа стажировок",
            "text": "Студенты 3–4 курсов проходят практику в ведущих IT-компаниях. "
                    "Длительность — от 2 до 6 месяцев. Средняя оплата — 30 000–50 000 "
                    "рублей в месяц. Менторская поддержка и возможность трудоустройства.",
            "url": f"{base}trudoustroystvo/internships/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Ярмарка вакансий",
            "text": "Ежегодная ярмарка в октябре и марте. Участвуют более 50 компаний. "
                    "Собеседования на месте, более 1 200 вакансий. Студенты могут оставить "
                    "резюме и узнать о стажировках.",
            "url": f"{base}trudoustroystvo/job-fair/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Трудоустройство иностранных выпускников",
            "text": "Иностранные выпускники имеют право на трудоустройство в упрощённом порядке. "
                    "Университет оказывает содействие в оформлении патента на работу. "
                    "Для выпускников из стран ЕАЭС — упрощённый режим без патента.",
            "url": f"{base}trudoustroystvo/international/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Составление резюме",
            "text": "Центр карьеры проводит мастер-классы по составлению резюме и сопроводительных "
                    "писем. Индивидуальные консультации помогают студентам выделить ключевые "
                    "компетенции и опыт. Шаблоны резюме доступны на портале карьеры.",
            "url": f"{base}trudoustroystvo/resume/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Подготовка к собеседованию",
            "text": "Тренинги по подготовке к техническим и HR-собеседованиям. Проводятся "
                    "пробные собеседования с представителями компаний-партнёров. Студенты "
                    "получают обратную связь и рекомендации по улучшению навыков.",
            "url": f"{base}trudoustroystvo/interview/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Банк вакансий",
            "text": "Электронный банк вакансий РТУ МИРЭА содержит более 3 000 актуальных "
                    "предложений от компаний-партнёров. Студенты могут фильтровать вакансии "
                    "по направлению подготовки, типу занятости и уровню дохода.",
            "url": f"{base}trudoustroystvo/vacancies/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Преддипломная практика",
            "text": "Преддипломная практика проводится на последнем курсе обучения. Студенты "
                    "направляются в организации-партнёры для выполнения практических задач. "
                    "Результаты практики используются при подготовке выпускной квалификационной "
                    "работы.",
            "url": f"{base}trudoustroystvo/pre-graduation/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },
        {
            "title": "Дни карьеры",
            "text": "Дни карьеры проводятся ежемесячно с участием конкретных компаний-партнёров. "
                    "Формат: презентация компании, мастер-класс, собеседование. Студенты "
                    "узнают о корпоративной культуре и требованиях к кандидатам.",
            "url": f"{base}trudoustroystvo/career-days/",
            "section": "student_portal",
            "subsection": "Трудоустройство",
        },

        # ── студенческая наука (11 items) ──
        {
            "title": "Научные студенческие общества (НСО)",
            "text": "НСО действуют при каждом институте. Студенты участвуют в НИР, публикуют "
                    "статьи, выступают на конференциях. Ежегодная Межвузовская конференция "
                    "с участием более 500 докладчиков.",
            "url": f"{base}studencheskaya-nauka/nso/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Конкурс научно-исследовательских работ",
            "text": "Ежегодный университетский конкурс НИР среди студентов 1–4 курсов. "
                    "Номинации: естественные науки, технические науки, гуманитарные науки. "
                    "Победители получают гранты на продолжение исследований и путёвки "
                    "на всероссийские конкурсы.",
            "url": f"{base}studencheskaya-nauka/competitions/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Студенческие научные конференции",
            "text": "Внутриуниверситетские конференции проводятся два раза в год: в декабре и "
                    "в апреле. Тематика охватывает все направления подготовки. Лучшие доклады "
                    "рекомендуются к публикации в журналах ВАК и Scopus.",
            "url": f"{base}studencheskaya-nauka/conferences/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Публикационная активность студентов",
            "text": "Студенты РТУ МИРЭА публикуют статьи в рецензируемых журналах, сборниках "
                    "конференций и на портале eLibrary. Среднее количество публикаций на "
                    "одного студента-исследователя — 2 статьи в год. В 2024 году опубликовано "
                    "более 1 200 студенческих работ.",
            "url": f"{base}studencheskaya-nauka/publications/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Гранты на студенческие научные проекты",
            "text": "Университет выделяет гранты до 200 000 рублей на выполнение студенческих "
                    "научных проектов. Срок реализации — 6 месяцев. Тематика: ИТ, "
                    "кибербезопасность, радиотехника, химические технологии. Заявки принимаются "
                    "два раза в год.",
            "url": f"{base}studencheskaya-nauka/grants/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Студенческое конструкторское бюро",
            "text": "Студенческое конструкторское бюро (СКБ) — площадка для проектной "
                    "деятельности. Студенты работают над реальными инженерными задачами "
                    "под руководством преподавателей и специалистов промышленности. "
                    "Проекты: беспилотные системы, IoT-устройства, робототехника.",
            "url": f"{base}studencheskaya-nauka/skb/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Участие в олимпиадах и хакатонах",
            "text": "Студенты РТУ МИРЭА участвуют во Всероссийских и международных олимпиадах, "
                    "хакатонах и соревнованиях по программированию (ICPC, Яндекс.Алгоритм, "
                    "VK Cup). Университет обеспечивает подготовку и командировку участников.",
            "url": f"{base}studencheskaya-nauka/olympiads/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Летние научные школы",
            "text": "Ежегодно в июле-августе проводятся летние научные школы по направлениям: "
                    "машинное обучение, кибербезопасность, радиотехника, нанотехнологии. "
                    "Длительность — 2 недели. Участие бесплатное для студентов РТУ МИРЭА.",
            "url": f"{base}studencheskaya-nauka/summer-schools/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Научные руководители и менторство",
            "text": "Каждый студент-исследователь закрепляется за научным руководителем из числа "
                    "преподавателей кафедры. Научный руководитель помогает сформировать "
                    "тему исследования, составляет план работы и консультирует по методологии.",
            "url": f"{base}studencheskaya-nauka/mentors/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Инновационная инфраструктура",
            "text": "В университете действуют научно-образовательные центры, лаборатории "
                    "коллективного пользования и технопарк. Студенты получают доступ к "
                    "современному оборудованию: 3D-принтеры, станки с ЧПУ, спектрометры, "
                    "кластерные вычислительные системы.",
            "url": f"{base}studencheskaya-nauka/infrastructure/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },
        {
            "title": "Сотрудничество с РАН",
            "text": "РТУ МИРЭА сотрудничает с институтами Российской академии наук. Студенты "
                    "проходят научные стажировки в лабораториях РАН, участвуют в "
                    "совместных исследовательских проектах. Результаты публикуются в "
                    "соавторстве с ведущими учёными.",
            "url": f"{base}studencheskaya-nauka/ran/",
            "section": "student_portal",
            "subsection": "Студенческая наука",
        },

        # ── нормативные документы (11 items) ──
        {
            "title": "Устав РТУ МИРЭА",
            "text": "Устав РТУ МИРЭА — основной документ, определяющий правовой статус, структуру, "
                    "порядок управления и организации образовательной деятельности университета. "
                    "Утверждён Министерством науки и высшего образования РФ.",
            "url": f"{base}normativnye-dokumenty/charter/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Правила внутреннего распорядка",
            "text": "Правила внутреннего распорядка регламентируют поведение студентов и "
                    "сотрудников на территории университета. Включают: расписание занятий, "
                    "порядок посещения, пропускной режим, правила поведения в аудиториях "
                    "и библиотеке.",
            "url": f"{base}normativnye-dokumenty/internal-rules/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Положение о стипендиальном обеспечении",
            "text": "Положение определяет порядок назначения и выплаты государственных "
                    "академических, социальных и именных стипендий. Регламентирует работу "
                    "стипендиальных комиссий и сроки назначения стипендий.",
            "url": f"{base}normativnye-dokumenty/stipendiation/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Положение о практике студентов",
            "text": "Положение регламентирует организацию и проведение всех видов практики: "
                    "учебной, производственной, преддипломной. Определяет права и обязанности "
                    "студентов, руководителей практики и принимающих организаций.",
            "url": f"{base}normativnye-dokumenty/practice/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Положение о курсовых и дипломных работах",
            "text": "Положение устанавливает требования к структуре, оформлению и защите "
                    "курсовых работ и выпускных квалификационных работ (ВКР). Определяет "
                    "критерии оценки, порядок формирования комиссий и проведения защиты.",
            "url": f"{base}normativnye-dokumenty/coursework/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Положение об оценке качества освоения ООП",
            "text": "Положение определяет формы и процедуры оценки качества освоения "
                    "основных образовательных программ: текущий контроль, промежуточная "
                    "и итоговая аттестация. Включает порядок подачи апелляций.",
            "url": f"{base}normativnye-dokumenty/assessment/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Порядок перевода и восстановления",
            "text": "Документ регламентирует порядок перевода студентов между направлениями "
                    "подготовки, формами обучения и восстановления отчисленных студентов. "
                    "Определяет сроки и условия перевода, перечень необходимых документов.",
            "url": f"{base}normativnye-dokumenty/transfer/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Антиплагиат. Порядок проверки ВКР",
            "text": "Все выпускные квалификационные работы проверяются через систему "
                    "«Антиплагиат.ВУЗ». Порог оригинальности — не менее 75% для бакалавров "
                    "и 80% для магистрантов. При несоответствии работа к защите не допускается.",
            "url": f"{base}normativnye-dokumenty/antiplagiat/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Правила приёма в РТУ МИРЭА",
            "text": "Правила приёма определяют условия и порядок поступления на программы "
                    "бакалавриата, специалитета и магистратуры. Включают перечень вступительных "
                    "испытаний, сроки приёма документов и порядок зачисления.",
            "url": f"{base}normativnye-dokumenty/admission/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Политика обработки персональных данных",
            "text": "Политика определяет порядок сбора, хранения, обработки и защиты "
                    "персональных данных студентов и сотрудников. Соответствует требованиям "
                    "Федерального закона №152-ФЗ «О персональных данных».",
            "url": f"{base}normativnye-dokumenty/personal-data/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },
        {
            "title": "Лицензия и аккредитация",
            "text": "РТУ МИРЭА осуществляет образовательную деятельность на основании лицензии "
                    "и свидетельства о государственной аккредитации. Документы подтверждают "
                    "право ведения образовательной деятельности и выдачу дипломов "
                    "государственного образца.",
            "url": f"{base}normativnye-dokumenty/license/",
            "section": "student_portal",
            "subsection": "Нормативные документы",
        },

        # ── студенческие объединения (11 items) ──
        {
            "title": "Студенческий совет РТУ МИРЭА",
            "text": "Студенческий совет — высший орган студенческого самоуправления. Состоит "
                    "из представителей всех институтов. Организует мероприятия, защищает права "
                    "студентов, участвует в управлении университетом. Заседания — еженедельно.",
            "url": f"{base}studencheskie-obedineniya/student-council/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Спортивный клуб",
            "text": "Спортивный клуб объединяет более 30 секций: мини-футбол, баскетбол, "
                    "волейбол, настольный теннис, шахматы, плавание, единоборства. Сборные "
                    "участвуют в Московском межвузовском округе и Всероссийской универсиаде.",
            "url": f"{base}studencheskie-obedineniya/sports/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Творческие коллективы",
            "text": "Театральная студия «Этюд», вокальная «Гармония», танцевальный «Ритм», "
                    "КВН-команда «МИРЭА-сборная», литературный клуб «Слово», фотоклуб "
                    "«Объектив». Участие в городских и всероссийских конкурсах.",
            "url": f"{base}studencheskie-obedineniya/creative/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Волонтёрский центр",
            "text": "Координирует добровольческую деятельность: помощь ветеранам, "
                    "благоустройство, благотворительные акции. В 2024 году реализовано "
                    "более 40 проектов, охвачено 800+ участников.",
            "url": f"{base}studencheskie-obedineniya/volunteer/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Студенческие медиа",
            "text": "Студенческое ТВ «МИРЭА-TV», радиостанция, газета «Технопарк», "
                    "интернет-портал, группы в соцсетях. Более 100 студентов-журналистов, "
                    "операторов, дизайнеров и SMM-специалистов.",
            "url": f"{base}studencheskie-obedineniya/media/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Профсоюз студентов",
            "text": "Профсоюзная организация студентов защищает социальные и трудовые права "
                    "студентов. Членам профсоюза доступны: путёвки в санатории, материальная "
                    "помощь, скидки на питание, юридические консультации. Членский взнос — "
                    "1% от стипендии.",
            "url": f"{base}studencheskie-obedineniya/trade-union/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Клуб интеллектуальных игр",
            "text": "Клуб «Что? Где? Когда?» и «Своя игра» собирает любителей интеллектуальных "
                    "соревнований. Команда университета участвует в Московском студенческом "
                    "чемпионате. Тренировки проводятся еженедельно по вторникам.",
            "url": f"{base}studencheskie-obedineniya/quiz/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "IT-клуб",
            "text": "IT-клуб организует лекции, воркшопы и хакатоны по актуальным темам: "
                    "машинное обучение, веб-разработка, мобильная разработка, DevOps. "
                    "Члены клуба участвуют в проектах с реальными заказчиками.",
            "url": f"{base}studencheskie-obedineniya/it-club/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Дебатный клуб",
            "text": "Дебатный клуб развивает навыки аргументации, публичных выступлений и "
                    "критического мышления. Формат: британские парламентские дебаты. "
                    "Команда участвует в российских и международных турнирах.",
            "url": f"{base}studencheskie-obedineniya/debate/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Клуб интернациональной дружбы",
            "text": "Клуб интернациональной дружбы объединяет российских и иностранных "
                    "студентов. Проводятся вечера культур, языковые обмены, экскурсии "
                    "по Москве, спортивные турниры. Члены клуба помогают адаптироваться "
                    "иностранным студентам к жизни в России.",
            "url": f"{base}studencheskie-obedineniya/international-club/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },
        {
            "title": "Экологическое объединение «Зелёный МИРЭА»",
            "text": "Экологическое объединение проводит акции по раздельному сбору мусора, "
                    "озеленению территории университета, просветительские лекции об экологии. "
                    "В 2024 году организовано 15 экологических акций с участием 300+ студентов.",
            "url": f"{base}studencheskie-obedineniya/eco/",
            "section": "student_portal",
            "subsection": "Студенческие объединения",
        },

        # ── психологическая служба (11 items) ──
        {
            "title": "Психологическая служба РТУ МИРЭА",
            "text": "Психологическая служба оказывает бесплатную психологическую помощь "
                    "студентам и сотрудникам. Услуги: индивидуальные консультации, "
                    "групповые тренинги, диагностика, кризисная помощь. Кабинеты расположены "
                    "в каждом учебном корпусе.",
            "url": f"{base}psikhologicheskaya-sluzhba/about/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Индивидуальные психологические консультации",
            "text": "Индивидуальные консультации проводятся по предварительной записи. "
                    "Продолжительность — 50 минут. Темы: адаптация к обучению, межличностные "
                    "отношения, стресс, тревожность, мотивация. Конфиденциальность гарантирована.",
            "url": f"{base}psikhologicheskaya-sluzhba/individual/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Групповые тренинги",
            "text": "Групповые тренинги проводятся еженедельно. Темы: управление стрессом, "
                    "коммуникативные навыки, уверенность в себе, тайм-менеджмент, "
                    "профилактика выгорания. Группа — от 8 до 15 человек. Запись через "
                    "электронную форму.",
            "url": f"{base}psikhologicheskaya-sluzhba/group/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Кризисная психологическая помощь",
            "text": "Кризисная помощь доступна в экстренных ситуациях: острая стрессовая "
                    "реакция, тяжёлая утрата, конфликт, суицидальный риск. Горячая линия "
                    "работает круглосуточно. Телефон: +7 (495) 000-00-00. Обращение "
                    "анонимно.",
            "url": f"{base}psikhologicheskaya-sluzhba/crisis/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Адаптация первокурсников",
            "text": "Программа адаптации первокурсников включает: вводные тренинги в сентябре, "
                    "встречи с психологом на кураторских часах, индивидуальные консультации "
                    "по запросу. Цель — помочь студентам адаптироваться к новой среде "
                    "и учебной нагрузке.",
            "url": f"{base}psikhologicheskaya-sluzhba/freshmen/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Профилактика экзаменационного стресса",
            "text": "В период сессии психологическая служба проводит специальные тренинги "
                    "по управлению экзаменационным стрессом. Студенты обучаются техникам "
                    "релаксации, дыхательным упражнениям, позитивному мышлению. "
                    "Консультации доступны в режиме онлайн.",
            "url": f"{base}psikhologicheskaya-sluzhba/exam-stress/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Психологическая диагностика",
            "text": "Психологическая служба проводит диагностику: определение уровня тревожности, "
                    "склонности к выгоранию, типа темперамента, мотивационного профиля. "
                    "Результаты помогают студентам понять свои особенности и выбрать "
                    "стратегии совладания.",
            "url": f"{base}psikhologicheskaya-sluzhba/diagnostics/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Тренинг уверенности в себе",
            "text": "Тренинг направлен на развитие уверенного поведения, умения отстаивать "
                    "свои границы и конструктивно решать конфликты. Состоит из 8 занятий "
                    "по 1,5 часа. Проводится один раз в семестр. Запись обязательна.",
            "url": f"{base}psikhologicheskaya-sluzhba/confidence/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Онлайн-консультирование",
            "text": "Для студентов, которые не могут прийти лично, доступно онлайн-"
                    "консультирование через видеосвязь. Запись через форму на портале. "
                    "Сеанс проводится на платформе университета с обеспечением "
                    "конфиденциальности.",
            "url": f"{base}psikhologicheskaya-sluzhba/online/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Полезные материалы и ресурсы",
            "text": "На портале психологической службы доступны: статьи по психологии, "
                    "видеолекции, упражнения для самопомощи, рекомендуемая литература, "
                    "ссылки на кризисные центры Москвы. Материалы обновляются ежемесячно.",
            "url": f"{base}psikhologicheskaya-sluzhba/resources/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
        {
            "title": "Контактная информация",
            "text": "Психологическая служба: главный корпус, каб. Б-305. Телефон: +7 (495) "
                    "000-00-00. Электронная почта: psychology@mirea.ru. Приёмные часы: "
                    "понедельник–пятница, 9:00–18:00. Запись на консультацию через "
                    "личный кабинет студента.",
            "url": f"{base}psikhologicheskaya-sluzhba/contacts/",
            "section": "student_portal",
            "subsection": "Психологическая служба",
        },
    ]


def save_items(items: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {len(items)} items to {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Student Portal Scraper (student.mirea.ru)")
    print("=" * 60)

    items = scrape_student_portal()

    if not items:
        print("[!] student.mirea.ru unreachable or no data scraped — FAILING (no silent demo fallback)")
        sys.exit(1)

    save_items(items, OUTPUT_FILE)
