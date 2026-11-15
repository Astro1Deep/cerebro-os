import os
import sys
import tempfile
import unittest
import stat
from unittest.mock import patch
from pathlib import Path


PAYLOAD = Path(__file__).resolve().parent.parent
if str(PAYLOAD) not in sys.path:
    sys.path.insert(0, str(PAYLOAD))


class WebKnowledgeTests(unittest.TestCase):

    def test_database_connections_close_and_vector_failure_keeps_lexical_search(self):
        from cerebro_core.web_knowledge import WebKnowledgeStore
        with tempfile.TemporaryDirectory() as directory:
            with patch('cerebro_core.web_knowledge.VectorEngine', side_effect=RuntimeError('offline')):
                store = WebKnowledgeStore(Path(directory))
                doc = store.record_pending(url='https://example.com/', title='Harmony',
                                           text='Cadence resolves the dominant.', fetched_at=1.0)
                store.approve(doc)
                self.assertEqual(store.search('cadence')[0]['id'], doc)
                self.assertEqual(store.search('unmatched'), [])
            store.database_path.unlink()


    def test_public_url_validation_rejects_local_and_credentialed_targets(self):
        from cerebro_core.web_knowledge import UnsafeWebTarget, validate_public_url

        for candidate in (
            "http://127.0.0.1:8080/admin",
            "http://localhost/private",
            "http://user:password@example.com/",
            "file:///tmp/private.txt",
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaises(UnsafeWebTarget):
                    validate_public_url(candidate, resolver=lambda _host: ["93.184.216.34"])

    def test_crawler_keeps_same_host_honours_robots_and_records_pending_sources(self):
        from cerebro_core.web_knowledge import CrawlLimits, FetchedPage, WebKnowledgeCrawler, WebKnowledgeStore

        pages = {
            "https://example.com/": FetchedPage(
                requested_url="https://example.com/",
                final_url="https://example.com/",
                content_type="text/html",
                body=b"<title>Inicio</title><p>Conocimiento musical local verificable.</p><a href='/guide'>Guia</a><a href='https://other.example/x'>Fuera</a>",
            ),
            "https://example.com/guide": FetchedPage(
                requested_url="https://example.com/guide",
                final_url="https://example.com/guide",
                content_type="text/html",
                body=b"<title>Guia</title><p>Armonia y composicion con fuentes trazables.</p>",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            store = WebKnowledgeStore(Path(directory))
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(store.database_path.stat().st_mode), 0o600)
            crawler = WebKnowledgeCrawler(
                store,
                limits=CrawlLimits(max_pages=5, max_total_bytes=100_000, max_page_bytes=50_000),
                fetcher=lambda url, _limit: pages[url],
                robots_allowed=lambda _url: True,
                validator=lambda url: url,
            )
            report = crawler.crawl("https://example.com/")

            self.assertEqual(report["pages_pending"], 2)
            self.assertEqual(report["pages_skipped_cross_host"], 1)
            pending = store.list_documents(status="pending")
            self.assertEqual(len(pending), 2)
            self.assertTrue(all(item["sha256"] for item in pending))
            self.assertTrue(all(item["status"] == "pending" for item in pending))

    def test_robots_denial_creates_no_knowledge(self):
        from cerebro_core.web_knowledge import CrawlLimits, FetchedPage, WebKnowledgeCrawler, WebKnowledgeStore

        page = FetchedPage("https://example.com/", "https://example.com/", "text/html", b"<p>Texto</p>")
        with tempfile.TemporaryDirectory() as directory:
            store = WebKnowledgeStore(Path(directory))
            crawler = WebKnowledgeCrawler(
                store,
                limits=CrawlLimits(max_pages=1, max_total_bytes=10_000, max_page_bytes=10_000),
                fetcher=lambda _url, _limit: page,
                robots_allowed=lambda _url: False,
                validator=lambda url: url,
            )
            report = crawler.crawl("https://example.com/")

            self.assertEqual(report["pages_pending"], 0)
            self.assertEqual(report["pages_blocked_robots"], 1)
            self.assertEqual(store.list_documents(status="pending"), [])

    def test_only_explicitly_approved_documents_are_searchable_by_local_rag(self):
        from cerebro_core.web_knowledge import WebKnowledgeStore

        with tempfile.TemporaryDirectory() as directory:
            store = WebKnowledgeStore(Path(directory))
            pending_id = store.record_pending(
                url="https://example.com/guia",
                title="Guia de armonia",
                text="La cadencia autentica resuelve dominante hacia tonica.",
                fetched_at=1.0,
            )
            self.assertEqual(store.search("cadencia dominante"), [])

            approved = store.approve(pending_id)
            results = store.search("cadencia dominante")

            self.assertEqual(approved["status"], "approved")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], pending_id)
            context = store.rag_context("cadencia", max_characters=500)
            self.assertIn("Fuente local aprobada", context)
            self.assertIn("example.com", context)
