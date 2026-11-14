"""Explicit, bounded web acquisition for Cerebro's local knowledge base.

This module is deliberately not an agent tool.  It runs only when a user invokes
the reviewed CLI or an authenticated owner-only Telegram command.  Fetched pages
remain pending until explicitly approved; only approved records are eligible for
local retrieval by the chat client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import socket
import sqlite3
import os
from cerebro_core.vector_engine import VectorEngine
from typing import Callable, Final
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib import robotparser


USER_AGENT: Final[str] = "CerebroResearch/1.0 (+local-first; explicit-user-request)"
ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
ALLOWED_PORTS: Final[frozenset[int]] = frozenset({80, 443})
MAX_LIMIT_PAGES: Final[int] = 200
MAX_LIMIT_TOTAL_BYTES: Final[int] = 100 * 1024 * 1024
MAX_LIMIT_PAGE_BYTES: Final[int] = 10 * 1024 * 1024


class WebKnowledgeError(ValueError):
    """A web acquisition request could not be completed safely."""


class UnsafeWebTarget(WebKnowledgeError):
    """A URL could reach a local or otherwise non-public target."""


@dataclass(frozen=True)
class CrawlLimits:
    max_pages: int = 50
    max_total_bytes: int = 5 * 1024 * 1024
    max_page_bytes: int = 1 * 1024 * 1024

    def __post_init__(self) -> None:
        if not 1 <= self.max_pages <= MAX_LIMIT_PAGES:
            raise WebKnowledgeError(f"max_pages debe estar entre 1 y {MAX_LIMIT_PAGES}")
        if not 1 <= self.max_total_bytes <= MAX_LIMIT_TOTAL_BYTES:
            raise WebKnowledgeError("max_total_bytes fuera del límite seguro")
        if not 1 <= self.max_page_bytes <= MAX_LIMIT_PAGE_BYTES:
            raise WebKnowledgeError("max_page_bytes fuera del límite seguro")
        if self.max_page_bytes > self.max_total_bytes:
            raise WebKnowledgeError("max_page_bytes no puede superar max_total_bytes")


@dataclass(frozen=True)
class FetchedPage:
    requested_url: str
    final_url: str
    content_type: str
    body: bytes


def _normalise_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeWebTarget("la URL no incluye un host público")
    port = parsed.port
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def _resolve_public_addresses(host: str) -> list[str]:
    try:
        records = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise UnsafeWebTarget("no se pudo resolver el dominio") from error
    addresses = sorted({record[4][0] for record in records})
    if not addresses:
        raise UnsafeWebTarget("el dominio no tiene una dirección pública")
    return addresses


def _is_public_address(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


def validate_public_url(
    value: str,
    *,
    resolver: Callable[[str], list[str]] = _resolve_public_addresses,
) -> str:
    """Normalize a normal web URL and reject local/private SSRF targets.

    Each requested URL is validated before fetching.  Redirect targets are also
    passed through this function, so a public page cannot redirect Cerebro into a
    loopback or LAN service.
    """

    if not isinstance(value, str) or not value.strip():
        raise UnsafeWebTarget("la URL debe ser texto no vacío")
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeWebTarget("sólo se permiten URLs http y https")
    if parsed.username or parsed.password:
        raise UnsafeWebTarget("no se permiten credenciales dentro de la URL")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise UnsafeWebTarget("no se permiten hosts locales")
    try:
        port = parsed.port
    except ValueError as error:
        raise UnsafeWebTarget("el puerto de la URL no es válido") from error
    if port is not None and port not in ALLOWED_PORTS:
        raise UnsafeWebTarget("sólo se permiten los puertos web 80 y 443")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise UnsafeWebTarget("no se permiten direcciones IP no públicas")
    addresses = [str(literal)] if literal is not None else resolver(host)
    if not all(_is_public_address(address) for address in addresses):
        raise UnsafeWebTarget("el dominio resuelve a una dirección no pública")
    return _normalise_url(value)


class _ValidatedRedirect(HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], str]) -> None:
        super().__init__()
        self._validator = validator

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        self._validator(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_public_page(url: str, max_bytes: int, *, validator: Callable[[str], str] = validate_public_url) -> FetchedPage:
    """Fetch a small HTML/plain-text response while enforcing public redirects."""

    if max_bytes < 1:
        raise WebKnowledgeError("el límite de descarga debe ser positivo")
    requested_url = validator(url)
    opener = build_opener(_ValidatedRedirect(validator))
    request = Request(
        requested_url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9"},
    )
    try:
        with opener.open(request, timeout=15) as response:
            final_url = validator(response.geturl())
            content_type = response.headers.get_content_type().lower()
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                raise WebKnowledgeError("el recurso no es HTML ni texto")
            declared_size = response.headers.get("Content-Length")
            if declared_size and declared_size.isdigit() and int(declared_size) > max_bytes:
                raise WebKnowledgeError("el recurso supera el límite por página")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(min(65_536, max_bytes - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise WebKnowledgeError("el recurso supera el límite por página")
                chunks.append(chunk)
    except (HTTPError, URLError, OSError) as error:
        raise WebKnowledgeError(f"no se pudo descargar la URL: {type(error).__name__}") from error
    return FetchedPage(requested_url, final_url, content_type, b"".join(chunks))


def robots_allows(url: str, *, validator: Callable[[str], str] = validate_public_url) -> bool:
    """Respect robots.txt; inability to read it is a conservative denial."""

    parsed = urlsplit(validator(url))
    robots_url = urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))
    try:
        robots = fetch_public_page(robots_url, 256 * 1024, validator=validator)
    except WebKnowledgeError:
        return False
    parser = robotparser.RobotFileParser()
    parser.set_url(robots.final_url)
    parser.parse(robots.body.decode("utf-8", errors="replace").splitlines())
    return parser.can_fetch(USER_AGENT, url)


class _ReadablePage(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "template"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)


def extract_readable_page(page: FetchedPage) -> tuple[str, str, list[str]]:
    """Extract readable text and links without executing page code."""

    raw = page.body.decode("utf-8", errors="replace")
    if page.content_type == "text/plain":
        text = re.sub(r"\s+", " ", raw).strip()
        return "", text, []
    parser = _ReadablePage()
    parser.feed(raw)
    parser.close()
    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()[:512]
    text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
    return title, text, parser.links


class WebKnowledgeStore:
    """Local source store with pending and explicitly approved records."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.database_path = self.root / "knowledge.db"
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA busy_timeout=5000;")
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'approved')),
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    fetched_at REAL NOT NULL,
                    approved_at REAL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS document_search USING fts5(id UNINDEXED, title, text)"
            )
        os.chmod(self.database_path, 0o600)

    def record_pending(self, *, url: str, title: str, text: str, fetched_at: float) -> str:
        clean_text = text.strip()[:200_000]
        if not clean_text:
            raise WebKnowledgeError("la página no contiene texto indexable")
        digest = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        identifier = hashlib.sha256(f"{url}\n{digest}\n{fetched_at:.6f}".encode("utf-8")).hexdigest()[:16]
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO documents(id, status, url, title, text, sha256, fetched_at, approved_at) VALUES (?, 'pending', ?, ?, ?, ?, ?, NULL)",
                (identifier, url, title.strip()[:512], clean_text, digest, fetched_at),
            )
        return identifier

    def list_documents(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        if status not in {None, "pending", "approved"}:
            raise WebKnowledgeError("estado de conocimiento no válido")
        if not 1 <= limit <= 500:
            raise WebKnowledgeError("límite de documentos no válido")
        query = "SELECT id, status, url, title, sha256, fetched_at, approved_at, length(text) AS characters FROM documents"
        parameters: tuple[object, ...] = ()
        if status:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY fetched_at DESC LIMIT ?"
        parameters += (limit,)
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def get_document(self, identifier: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, status, url, title, text, sha256, fetched_at, approved_at FROM documents WHERE id = ?", (identifier,)
            ).fetchone()
        if row is None:
            raise WebKnowledgeError("no existe ese documento de conocimiento")
        return dict(row)

    def approve(self, identifier: str, *, approved_at: float | None = None) -> dict[str, object]:
        record = self.get_document(identifier)
        if record["status"] == "approved":
            return record
        timestamp = datetime.now(timezone.utc).timestamp() if approved_at is None else approved_at
        with self._connect() as connection:
            connection.execute("UPDATE documents SET status = 'approved', approved_at = ? WHERE id = ?", (timestamp, identifier))
            connection.execute(
                "INSERT INTO document_search(id, title, text) VALUES (?, ?, ?)",
                (record["id"], record["title"], record["text"]),
            )
            try:
                lancedb_root = self.root / "lancedb" if hasattr(self, "root") else Path("~/.mlx-stack/knowledge/lancedb").expanduser()
                engine = VectorEngine(root=lancedb_root)
                engine.upsert(
                    doc_id=identifier, 
                    text=record["text"][:2000], 
                    metadata={
                        "source": "web_knowledge",
                        "url": record["url"],
                        "title": record["title"],
                        "status": "approved"
                    }
                )
            except Exception as e:
                logger.debug("[WebKnowledge] Vector Engine error: %s", e)
        return self.get_document(identifier)

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[\wÀ-ÿ]{2,}", query.lower())[:12]
        return " OR ".join(f'"{term}"' for term in terms)


    def search(self, query: str, *, limit: int = 3) -> list[dict[str, object]]:
        if not query or not query.strip() or limit <= 0:
            return []

        results: list[dict[str, object]] = []
        seen_ids: set[str] = set()

        # 1. Búsqueda léxica SQLite FTS5 (local, rápida y determinista)
        fts = self._fts_query(query)
        with self._connect() as connection:
            if fts:
                try:
                    rows = connection.execute(
                        """
                        SELECT d.id, d.url, d.title, d.sha256, substr(d.text, 1, 300) || '...' AS excerpt
                        FROM document_search s
                        JOIN documents d ON d.id = s.id
                        WHERE d.status = 'approved' AND document_search MATCH ?
                        ORDER BY bm25(document_search)
                        LIMIT ?
                        """,
                        (fts, limit),
                    ).fetchall()
                    for row in rows:
                        d = dict(row)
                        results.append(d)
                        seen_ids.add(d["id"])
                except Exception as e:
                    logger.debug("FTS5 query failed: %s", e)

        # 2. Búsqueda vectorial semántica (LanceDB)
        if len(results) < limit:
            try:
                lancedb_root = self.root / "lancedb" if hasattr(self, "root") else Path("~/.mlx-stack/knowledge/lancedb").expanduser()
                engine = VectorEngine(root=lancedb_root)
                vec_results = engine.search(text=query, limit=limit)
                if vec_results:
                    with self._connect() as connection:
                        for r in vec_results:
                            doc_id = r.get("id")
                            if doc_id and doc_id not in seen_ids:
                                row = connection.execute(
                                    "SELECT id, url, title, sha256, substr(text, 1, 300) || '...' AS excerpt FROM documents WHERE id=? AND status='approved'",
                                    (doc_id,),
                                ).fetchone()
                                if row:
                                    d = dict(row)
                                    results.append(d)
                                    seen_ids.add(d["id"])
                                    if len(results) >= limit:
                                        break
            except Exception as e:
                logger.debug("Vector search fallback failed: %s", e)

        return results


    def rag_context(self, query: str, *, max_characters: int = 4_000) -> str:
        if max_characters < 200:
            raise WebKnowledgeError("el contexto RAG debe permitir al menos 200 caracteres")
        sections: list[str] = []
        remaining = max_characters
        for record in self.search(query):
            text = f"Fuente local aprobada: {record['title'] or 'sin título'}\nURL: {record['url']}\nExtracto: {record['excerpt']}"
            if len(text) > remaining:
                text = text[:remaining].rstrip()
            if not text:
                break
            sections.append(text)
            remaining -= len(text) + 2
            if remaining < 200:
                break
        return "\n\n".join(sections)


class WebKnowledgeCrawler:
    """Same-host crawler that writes pending, never automatically active, knowledge."""

    def __init__(
        self,
        store: WebKnowledgeStore,
        *,
        limits: CrawlLimits = CrawlLimits(),
        fetcher: Callable[[str, int], FetchedPage] = fetch_public_page,
        robots_allowed: Callable[[str], bool] = robots_allows,
        validator: Callable[[str], str] = validate_public_url,
    ) -> None:
        self.store = store
        self.limits = limits
        self.fetcher = fetcher
        self.robots_allowed = robots_allowed
        self.validator = validator

    def crawl(self, start_url: str) -> dict[str, object]:
        start = self.validator(start_url)
        start_host = urlsplit(start).netloc.lower()
        queue = [start]
        visited: set[str] = set()
        identifiers: list[str] = []
        total_bytes = 0
        blocked_robots = 0
        skipped_cross_host = 0
        errors: list[str] = []

        while queue and len(identifiers) < self.limits.max_pages and total_bytes < self.limits.max_total_bytes:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            if not self.robots_allowed(url):
                blocked_robots += 1
                continue
            page_limit = min(self.limits.max_page_bytes, self.limits.max_total_bytes - total_bytes)
            try:
                page = self.fetcher(url, page_limit)
                final_url = self.validator(page.final_url)
            except (KeyError, WebKnowledgeError, UnsafeWebTarget) as error:
                errors.append(f"{url}: {error}")
                continue
            if urlsplit(final_url).netloc.lower() != start_host:
                skipped_cross_host += 1
                continue
            if len(page.body) > page_limit:
                errors.append(f"{url}: respuesta por encima del límite")
                continue
            total_bytes += len(page.body)
            title, text, links = extract_readable_page(page)
            if text:
                identifiers.append(
                    self.store.record_pending(
                        url=final_url,
                        title=title,
                        text=text,
                        fetched_at=datetime.now(timezone.utc).timestamp(),
                    )
                )
            for href in links:
                next_url = urljoin(final_url, href)
                try:
                    candidate = self.validator(next_url)
                except UnsafeWebTarget:
                    continue
                if urlsplit(candidate).netloc.lower() != start_host:
                    skipped_cross_host += 1
                elif candidate not in visited and candidate not in queue:
                    queue.append(candidate)

        return {
            "start_url": start,
            "pages_pending": len(identifiers),
            "pending_ids": identifiers,
            "pages_blocked_robots": blocked_robots,
            "pages_skipped_cross_host": skipped_cross_host,
            "bytes_downloaded": total_bytes,
            "errors": errors,
            "approval_required": True,
        }


def default_store(base_dir: Path | str) -> WebKnowledgeStore:
    return WebKnowledgeStore(Path(base_dir).resolve() / "knowledge")


def as_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
