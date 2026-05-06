from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re

from fastapi.testclient import TestClient

from gui.server import app

STATIC_DIR = Path(__file__).resolve().parents[1] / "gui" / "static"
BANNED_FRONTEND_COLOR_PATTERNS = (
    r"\bteal\b",
    r"\bemerald\b",
    r"\bgreen\b",
    r"--rr-teal\b",
    r"#1d6b53",
    r"#e4f3ec",
    r"#166534",
    r"#15803d",
    r"#16a34a",
)


class _StaticContractParser(HTMLParser):
    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.script_srcs: list[str] = []
        self.link_hrefs: list[str] = []
        self._depth = 0
        self._hidden_depths: list[int] = []
        self.focusables_under_aria_hidden: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._depth += 1
        attr = {name: value or "" for name, value in attrs}
        if tag == "script" and attr.get("src"):
            self.script_srcs.append(attr["src"])
        if tag == "link" and attr.get("href"):
            self.link_hrefs.append(attr["href"])

        is_hidden = attr.get("aria-hidden") == "true"
        if is_hidden:
            self._hidden_depths.append(self._depth)

        if self._hidden_depths:
            hidden_owner = attr.get("id") or attr.get("class") or tag
            if self._is_focusable(tag, attr):
                self.focusables_under_aria_hidden.append((tag, hidden_owner))

        if tag in self._VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._hidden_depths and self._hidden_depths[-1] == self._depth:
            self._hidden_depths.pop()
        self._depth = max(0, self._depth - 1)

    @staticmethod
    def _is_focusable(tag: str, attr: dict[str, str]) -> bool:
        if "disabled" in attr or attr.get("tabindex") == "-1":
            return False
        if tag in {"button", "select", "textarea"}:
            return True
        if tag == "input" and attr.get("type") != "hidden":
            return True
        if tag == "a" and attr.get("href"):
            return True
        if "tabindex" in attr:
            return True
        return False


def _parse_index() -> _StaticContractParser:
    html = TestClient(app).get("/").text
    parser = _StaticContractParser()
    parser.feed(html)
    return parser


def test_index_serves_static_frontend() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "RRRIE | Clinical Intelligence" in response.text


def test_favicon_is_served() -> None:
    response = TestClient(app).get("/favicon.ico")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")


def test_index_uses_local_frontend_assets_not_runtime_cdn() -> None:
    parser = _parse_index()
    urls = parser.script_srcs + parser.link_hrefs

    assert not any("cdn.tailwindcss.com" in url for url in urls)
    assert not any("marked.min.js" in url for url in urls)


def test_analysis_progress_has_accessible_state_text() -> None:
    html = TestClient(app).get("/").text

    assert 'id="analysisProgressRing"' in html
    assert 'role="progressbar"' in html
    assert 'aria-valuetext="No analysis running"' in html


def test_frontend_js_uses_project_classes_not_tailwind_utility_leftovers() -> None:
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    stale_tokens = (
        "text-slate",
        "text-primary",
        "bg-white",
        "rounded-lg",
        "shadow-sm",
        "border-slate",
        "scale-110",
        "text-blue-600",
        "bg-blue-50",
        "bg-amber-50",
        "text-amber",
    )

    offenders = [token for token in stale_tokens if token in app_js]

    assert offenders == []


def test_aria_hidden_regions_do_not_contain_focusable_controls() -> None:
    parser = _parse_index()

    assert parser.focusables_under_aria_hidden == []


def test_frontend_uses_neutral_blue_palette_not_green_or_teal() -> None:
    checked_files = [
        STATIC_DIR / "index.html",
        STATIC_DIR / "style.css",
        STATIC_DIR / "app.js",
    ]
    offenders: list[tuple[str, str]] = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8").lower()
        for pattern in BANNED_FRONTEND_COLOR_PATTERNS:
            if re.search(pattern, text):
                offenders.append((path.name, pattern))

    assert offenders == []
