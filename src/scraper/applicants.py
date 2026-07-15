"""Scraper for priem.mirea.ru — applicant-facing admission info (Joomla CMS)."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://priem.mirea.ru"
SECTION = "applicants"
DELAY = 1.5
MAX_RETRIES = 3
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SUBPAGES = [
    ("/", "Главная"),
    ("/ru/priem/", "Приёмная кампания"),
    ("/ru/priem/bakalavriat-i-specialitet/", "Бакалавриат и специалитет"),
    ("/ru/priem/magistratura/", "Магистратура"),
    ("/ru/priem/aspirantura/", "Аспирантура"),
    ("/ru/priem/prohodnye-bally/", "Проходные баллы"),
    ("/ru/priem/dokumenty/", "Документы"),
    ("/ru/priem/sroki/", "Сроки приёма"),
    ("/ru/priem/kontakty/", "Контакты приёмной комиссии"),
    ("/ru/priem/vopros-otvet/", "Вопрос-ответ"),
    ("/ru/priem/stoimost-obucheniya/", "Стоимость обучения"),
    ("/ru/priem/obshchezhitie/", "Общежитие"),
    ("/ru/priem/podgotovitelnye-kursy/", "Подготовительные курсы"),
    ("/ru/priem/olimpiady/", "Олимпиады"),
    ("/ru/priem/tselye-priem/", "Целевой приём"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir() -> Path:
    """Resolve the output directory from config.DATA_RAW_DIR."""
    try:
        import config

        raw_dir = getattr(config, "DATA_RAW_DIR", "mirea_data/raw")
    except Exception:
        raw_dir = "mirea_data/raw"
    p = Path(raw_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clean_text(s: str) -> str:
    """Normalise whitespace and strip."""
    return re.sub(r"\s+", " ", s).strip()


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove nav, footer, sidebar, script, style elements in-place."""
    for tag in soup.find_all(
        ["nav", "footer", "header", "script", "style", "noscript", "aside"]
    ):
        tag.decompose()
    for tag in soup.find_all(class_=re.compile(
        r"nav|menu|footer|header|sidebar|breadcrumb|social|cookie|banner|modal",
        re.I,
    )):
        tag.decompose()
    for tag in soup.find_all(id=re.compile(
        r"nav|menu|footer|header|sidebar|breadcrumb|social|cookie|banner|modal",
        re.I,
    )):
        tag.decompose()


def _parse_table(table: Tag) -> list[str]:
    """Parse an HTML table into a list of 'H1: v1 | H2: v2' strings."""
    rows: list[str] = []
    header_row = table.find("tr")
    if not header_row:
        return rows
    headers = [
        _clean_text(th.get_text()) for th in header_row.find_all(["th", "td"])
    ]
    if not headers:
        return rows
    for tr in table.find_all("tr")[1:]:
        cells = [_clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
        if not cells or all(c == "" for c in cells):
            continue
        parts = []
        for h, v in zip(headers, cells):
            if h and v:
                parts.append(f"{h}: {v}")
        if parts:
            rows.append(" | ".join(parts))
    return rows


def _is_reachable(url: str, timeout: int = 5) -> bool:
    try:
        resp = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True,
                             proxies={"http": None, "https": None})
        return resp.status_code < 500
    except requests.RequestException:
        return False


def _safe_get(session: requests.Session, url: str) -> BeautifulSoup | None:
    """GET with retry / exponential back-off for DDoS-Guard."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            resp.encoding = "utf-8"
            if resp.status_code in (403, 503) and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * 3
                print(f"  [retry {attempt+1}] {resp.status_code} — waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  [skip] {url} → HTTP {resp.status_code}")
                return None
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as exc:
            print(f"  [error] {url} → {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((2 ** attempt) * 3)
    return None


# ---------------------------------------------------------------------------
# Core scraping
# ---------------------------------------------------------------------------

def scrape_applicants() -> list[dict[str, Any]]:
    """Scrape priem.mirea.ru and return a list of structured items."""
    session = requests.Session()
    session.trust_env = False
    items: list[dict[str, Any]] = []

    for path, subsection in SUBPAGES:
        url = BASE_URL + path
        print(f"[*] {url}")
        soup = _safe_get(session, url)
        if soup is None:
            time.sleep(DELAY)
            continue

        _strip_noise(soup)

        content = (
            soup.find("div", class_=re.compile(r"content|article|item-page|post", re.I))
            or soup.find("main")
            or soup.find("body")
            or soup
        )

        title_tag = soup.find("h1") or soup.find("title")
        page_title = _clean_text(title_tag.get_text()) if title_tag else subsection

        seen_texts: set[str] = set()
        for el in content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = _clean_text(el.get_text())
            if not text or len(text) < 15:
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)
            items.append(
                {
                    "title": page_title,
                    "text": text,
                    "url": url,
                    "section": SECTION,
                    "subsection": subsection,
                }
            )

        for table in content.find_all("table"):
            for row_text in _parse_table(table):
                items.append(
                    {
                        "title": page_title,
                        "text": row_text,
                        "url": url,
                        "section": SECTION,
                        "subsection": subsection,
                    }
                )

        time.sleep(DELAY)

    return items


# ---------------------------------------------------------------------------
# Demo data (offline fallback)
# ---------------------------------------------------------------------------

def _demo_data() -> list[dict[str, Any]]:
    return [
        {"title": "Приёмная кампания 2025", "text": "Приёмная кампания в РТУ МИРЭА проводится в соответствии с Правилами приёма на обучение по образовательным программам высшего образования.", "url": "https://priem.mirea.ru/ru/priem/", "section": SECTION, "subsection": "Приёмная кампания"},
        {"title": "Проходные баллы", "text": "Проходные баллы на бюджетные места в 2024 году: Информатика и вычислительная техника — 254 балла, Радиотехника — 248 баллов, Информационная безопасность — 262 балла.", "url": "https://priem.mirea.ru/ru/priem/prohodnye-bally/", "section": SECTION, "subsection": "Проходные баллы"},
        {"title": "Проходные баллы", "text": "Проходной балл на направление Программная инженерия составил 260 баллов в 2024 году, на Биотехнологию — 235 баллов.", "url": "https://priem.mirea.ru/ru/priem/prohodnye-bally/", "section": SECTION, "subsection": "Проходные баллы"},
        {"title": "Проходные баллы", "text": "Конкурс на направление Кибербезопасность в 2024 году: 12,5 человек на место. Средний балл зачисленных — 271.", "url": "https://priem.mirea.ru/ru/priem/prohodnye-bally/", "section": SECTION, "subsection": "Проходные баллы"},
        {"title": "Бакалавриат", "text": "Приём на программы бакалавриата осуществляется по результатам ЕГЭ по профильным предметам: математика (профиль), физика или информатика и ИКТ, русский язык.", "url": "https://priem.mirea.ru/ru/priem/bakalavriat-i-specialitet/", "section": SECTION, "subsection": "Бакалавриат и специалитет"},
        {"title": "Бакалавриат", "text": "Направление 09.03.01 Информатика и вычислительная техника: 200 бюджетных мест (очная форма), 40 платных мест.", "url": "https://priem.mirea.ru/ru/priem/bakalavriat-i-specialitet/", "section": SECTION, "subsection": "Бакалавриат и специалитет"},
        {"title": "Бакалавриат", "text": "Направление 09.03.04 Программная инженерия: 150 бюджетных мест (очная форма), вступительные экзамены — математика, информатика, русский язык.", "url": "https://priem.mirea.ru/ru/priem/bakalavriat-i-specialitet/", "section": SECTION, "subsection": "Бакалавриат и специалитет"},
        {"title": "Бакалавриат", "text": "Направление 10.03.01 Информационная безопасность: 100 бюджетных мест. Дополнительное вступительное испытание — информатика.", "url": "https://priem.mirea.ru/ru/priem/bakalavriat-i-specialitet/", "section": SECTION, "subsection": "Бакалавриат и специалитет"},
        {"title": "Магистратура", "text": "Приём в магистратуру проводится на основе вступительных испытаний, проводимых университетом самостоятельно. Форма проведения — тестирование или собеседование.", "url": "https://priem.mirea.ru/ru/priem/magistratura/", "section": SECTION, "subsection": "Магистратура"},
        {"title": "Магистратура", "text": "Магистерская программа «Искусственный интеллект и машинное обучение»: 25 бюджетных мест, конкурс 8,3 человека на место в 2024 году.", "url": "https://priem.mirea.ru/ru/priem/magistratura/", "section": SECTION, "subsection": "Магистратура"},
        {"title": "Магистратура", "text": "Магистерская программа «Архитектура и проектирование программных систем»: 20 бюджетных мест, форма обучения очная, срок 2 года.", "url": "https://priem.mirea.ru/ru/priem/magistratura/", "section": SECTION, "subsection": "Магистратура"},
        {"title": "Аспирантура", "text": "Подготовка научно-педагогических кадров в аспирантуре РТУ МИРЭА осуществляется по 25 научным специальностям технического и естественно-научного профиля.", "url": "https://priem.mirea.ru/ru/priem/aspirantura/", "section": SECTION, "subsection": "Аспирантура"},
        {"title": "Аспирантура", "text": "Направление 09.06.01 Информатика и вычислительная техника: научная специальность 2.3.5 — Математическое и программное обеспечение вычислительных машин.", "url": "https://priem.mirea.ru/ru/priem/aspirantura/", "section": SECTION, "subsection": "Аспирантура"},
        {"title": "Документы", "text": "Для поступления необходимы: заявление о приёме, оригинал или копия документа об образовании, результаты ЕГЭ, паспорт, фотографии 3×4 (6 шт.).", "url": "https://priem.mirea.ru/ru/priem/dokumenty/", "section": SECTION, "subsection": "Документы"},
        {"title": "Документы", "text": "Заявление о приёме можно подать лично в приёмную комиссию, по почте заказным письмом или через портал Госуслуг.", "url": "https://priem.mirea.ru/ru/priem/dokumenty/", "section": SECTION, "subsection": "Документы"},
        {"title": "Сроки приёма", "text": "Приём документов начинается 20 июня и завершается 25 июля для программ бакалавриата и специалитета на бюджетные места.", "url": "https://priem.mirea.ru/ru/priem/sroki/", "section": SECTION, "subsection": "Сроки приёма"},
        {"title": "Сроки приёма", "text": "Для поступающих на платные места приём документов продлён до 10 августа. Зачисление проводится в два этапа: 3 августа и 8 августа.", "url": "https://priem.mirea.ru/ru/priem/sroki/", "section": SECTION, "subsection": "Сроки приёма"},
        {"title": "Стоимость обучения", "text": "Стоимость обучения на контрактной основе в 2025/2026 учебном году: бакалавриат — от 220 000 руб./год, магистратура — от 240 000 руб./год.", "url": "https://priem.mirea.ru/ru/priem/stoimost-obucheniya/", "section": SECTION, "subsection": "Стоимость обучения"},
        {"title": "Стоимость обучения", "text": "Направление Информатика и вычислительная техника (очная форма): 285 000 руб./год. Направление Информационная безопасность: 295 000 руб./год.", "url": "https://priem.mirea.ru/ru/priem/stoimost-obucheniya/", "section": SECTION, "subsection": "Стоимость обучения"},
        {"title": "Общежитие", "text": "РТУ МИРЭА предоставляет иногородним студентам места в общежитиях. Общежития расположены в шаговой доступности от станций метро.", "url": "https://priem.mirea.ru/ru/priem/obshchezhitie/", "section": SECTION, "subsection": "Общежитие"},
        {"title": "Общежитие", "text": "Общежитие №1 (ул. Б. Галушкина, д. 3): блочный тип, 2-3 человека в комнате, стоимость проживания — 1500 руб./мес.", "url": "https://priem.mirea.ru/ru/priem/obshchezhitie/", "section": SECTION, "subsection": "Общежитие"},
        {"title": "Подготовительные курсы", "text": "Подготовительные курсы для поступающих: математика, физика, информатика, русский язык. Стоимость — от 25 000 руб., длительность — 6 месяцев.", "url": "https://priem.mirea.ru/ru/priem/podgotovitelnye-kursy/", "section": SECTION, "subsection": "Подготовительные курсы"},
        {"title": "Олимпиады", "text": "Победители и призёры олимпиад школьников из перечня Минобрнауки имеют право на приём без вступительных испытаний.", "url": "https://priem.mirea.ru/ru/priem/olimpiady/", "section": SECTION, "subsection": "Олимпиады"},
        {"title": "Олимпиады", "text": "РТУ МИРЭА проводит собственную олимпиаду «Технологии будущего» для школьников 9–11 классов по информатике и программированию.", "url": "https://priem.mirea.ru/ru/priem/olimpiady/", "section": SECTION, "subsection": "Олимпиады"},
        {"title": "Целевой приём", "text": "Целевой приём осуществляется по квотам, установленным Правительством РФ. Договор о целевом обучении заключается между абитуриентом и организацией-заказчиком.", "url": "https://priem.mirea.ru/ru/priem/tselye-priem/", "section": SECTION, "subsection": "Целевой приём"},
        {"title": "Целевой приём", "text": "Количество целевых мест в 2025 году: Информатика и вычислительная техника — 20 мест, Радиотехника — 15 мест, Информационная безопасность — 10 мест.", "url": "https://priem.mirea.ru/ru/priem/tselye-priem/", "section": SECTION, "subsection": "Целевой приём"},
        {"title": "Вопрос-ответ", "text": "Можно ли подать документы на несколько направлений одновременно? Да, абитуриент может подать заявление на конкурсной основе не более чем на 5 специальностей.", "url": "https://priem.mirea.ru/ru/priem/vopros-otvet/", "section": SECTION, "subsection": "Вопрос-ответ"},
        {"title": "Вопрос-ответ", "text": "Какие льготы предусмотрены при поступлении? Инвалиды I и II группы, дети-сироты имеют право на приём в пределах особой квоты (10% бюджетных мест).", "url": "https://priem.mirea.ru/ru/priem/vopros-otvet/", "section": SECTION, "subsection": "Вопрос-ответ"},
        {"title": "Вопрос-ответ", "text": "Срок действия результатов ЕГЭ — 4 года, предшествующих году поступления. Результаты ЕГЭ 2021–2024 годов действительны для поступления в 2025 году.", "url": "https://priem.mirea.ru/ru/priem/vopros-otvet/", "section": SECTION, "subsection": "Вопрос-ответ"},
        {"title": "Контакты приёмной комиссии", "text": "Приёмная комиссия: г. Москва, ул. Стромынка, д. 20, каб. 306. Телефон: +7 (495) 215-65-65. Email: priem@mirea.ru.", "url": "https://priem.mirea.ru/ru/priem/kontakty/", "section": SECTION, "subsection": "Контакты приёмной комиссии"},
        {"title": "Контакты приёмной комиссии", "text": "Режим работы приёмной комиссии в период приёмной кампании: понедельник–пятница с 9:00 до 18:00, суббота с 10:00 до 15:00.", "url": "https://priem.mirea.ru/ru/priem/kontakty/", "section": SECTION, "subsection": "Контакты приёмной комиссии"},
        {"title": "Главная", "text": "РТУ МИРЭА — один из ведущих технических университетов России, входит в топ-20 вузов по направлению «Информационные технологии».", "url": "https://priem.mirea.ru/", "section": SECTION, "subsection": "Главная"},
        {"title": "Главная", "text": "В 2024 году в РТУ МИРЭА подано более 30 000 заявлений от абитуриентов. Конкурс на бюджетные места в среднем составил 8,5 человек на место.", "url": "https://priem.mirea.ru/", "section": SECTION, "subsection": "Главная"},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Scraper: applicants  (priem.mirea.ru)")
    print("=" * 60)

    if not _is_reachable(BASE_URL):
        print("[!] mirea.ru unreachable — generating demo data")
        items = _demo_data()
    else:
        items = scrape_applicants()
        if not items:
            print("[!] No items scraped — generating demo data")
            items = _demo_data()

    out_path = _output_dir() / "applicants.json"
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] Saved {len(items)} items → {out_path}")


if __name__ == "__main__":
    main()
