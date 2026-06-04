"""Playwright scraper for Australian visa info pages.

Handles JS-rendered accordion content by clicking expand triggers
before extracting text. Run this whenever source pages change.
"""
from __future__ import annotations

from playwright.sync_api import Page, sync_playwright

from config import DATA_RAW_DIR, SCRAPE_URLS


EXPAND_SELECTORS = [
    "button[aria-expanded='false']",
    "[role='button'][aria-expanded='false']",
    "summary",
    "details summary",
    ".accordion-header",
    ".expander button",
]

NOISY_SELECTORS = [
    ".select2",
    ".select2-container",
    ".select2-results",
    "ul.select2-results__options",
    "[role='listbox']",
    "#select2-productcode-container",
    "[aria-labelledby='select2-productcode-container']",
    "header",
    "footer",
    "nav",
]

CONTENT_SELECTORS = [
    "article",
    "[role='main'] article",
    ".main-content",
    ".content",
    "main",
]


def expand_all_sections(page: Page) -> int:
    """Click every collapsible element so its content becomes visible to .inner_text()."""
    clicked = 0
    for selector in EXPAND_SELECTORS:
        try:
            elements = page.locator(selector)
            count = elements.count()
            for i in range(count):
                el = elements.nth(i)
                try:
                    el.scroll_into_view_if_needed(timeout=2000)
                    el.click(timeout=2000)
                    page.wait_for_timeout(400)
                    clicked += 1
                except Exception:
                    pass
        except Exception:
            pass
    return clicked


def remove_noise(page: Page) -> None:
    """Strip out chrome/nav/widgets that pollute extracted text."""
    for selector in NOISY_SELECTORS:
        try:
            loc = page.locator(selector)
            for i in range(loc.count()):
                try:
                    loc.nth(i).evaluate("el => el.remove()")
                except Exception:
                    pass
        except Exception:
            pass


def extract_best_content(page: Page) -> str:
    """Try semantic content selectors first, fall back to body text."""
    for selector in CONTENT_SELECTORS:
        try:
            loc = page.locator(selector)
            if loc.count() > 0:
                text = loc.first.inner_text().strip()
                if len(text) > 500:
                    print(f"  Using selector: {selector}")
                    return text
        except Exception:
            pass
    print("  Fallback: using body")
    return page.locator("body").inner_text()


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        for name, url in SCRAPE_URLS.items():
            print(f"\nScraping: {name}")
            print(f"URL: {url}")

            page = browser.new_page(viewport={"width": 1440, "height": 2200})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(3000)

                clicked = expand_all_sections(page)
                remove_noise(page)
                full_text = extract_best_content(page)

                output_file = DATA_RAW_DIR / f"{name}.txt"
                output_file.write_text(full_text, encoding="utf-8")

                print(f"  Saved: {output_file}")
                print(f"  Accordion clicks: {clicked}")
                print(f"  Preview: {full_text[:200]}...")

            except Exception as e:
                print(f"  Failed to scrape {name}: {e}")
            finally:
                page.close()

        browser.close()


if __name__ == "__main__":
    main()