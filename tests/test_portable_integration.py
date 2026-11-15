import hashlib
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from cerebro_core.beatos import beatos_api_base, library_status, search_samples
from cerebro_cli.__main__ import main
from cerebro_release.packager import build_archives
from composer_server import create_app
from web_jarvis import app


def test_catalog_discovery_is_live_and_read_only(tmp_path, monkeypatch):
    root = tmp_path / "External drive #2"
    root.mkdir()
    db_path = root / "Samples #2.db"
    with closing(sqlite3.connect(db_path)) as db:
        db.execute("CREATE TABLE sounds(title TEXT, path TEXT, tempo INTEGER)")
        db.execute("INSERT INTO sounds VALUES (?, ?, ?)", ("Kick.wav", "Kick.wav", 172))
        db.commit()
    before = hashlib.sha256(db_path.read_bytes()).digest()
    bridge = tmp_path / "bridge.json"
    bridge.write_text(json.dumps({
        "schema": "cerebro-beatos-bridge-v2",
        "api_base": "http://127.0.0.1:9899",
        "mounted_libraries": [{
            "id": "library", "db_path": str(db_path),
            "table_name": "sounds", "mount_path": str(root),
        }],
    }))
    monkeypatch.setenv("CEREBRO_BEATOS_BRIDGE", str(bridge))

    assert beatos_api_base() == "http://127.0.0.1:9899"
    assert library_status()["mounted_count"] == 1
    assert search_samples("Kick", bpm_min=171)[0]["bpm"] == 172
    assert not search_samples("Kick", disk="absent")
    assert hashlib.sha256(db_path.read_bytes()).digest() == before
    db_path.rename(root / "unmounted.db")
    assert not search_samples("Kick")


def test_model_chat_uses_loopback_and_rejects_redirects():
    with app.test_client() as client, mock.patch("web_jarvis.local_http_session") as request:
        request.return_value.post.return_value = SimpleNamespace(status_code=307)
        response = client.post("/api/chat", json={"text": "Hi", "messages": [None]})
        assert response.status_code == 503
        assert request.return_value.post.call_args.kwargs["allow_redirects"] is False


def test_web_interfaces_enforce_local_hosts():
    with app.test_client() as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/health", headers={"Host": "invalid.example"}).status_code == 400
    composer = TestClient(create_app(), base_url="http://localhost", client=("192.0.2.1", 8123))
    assert composer.get("/").status_code == 403


def test_no_repairs_when_probes_are_disabled():
    with mock.patch("auto_doctor.repair_local_services") as repair:
        assert main(["doctor", "--offline", "--repair"]) == 2
    repair.assert_not_called()


def test_theme_supports_obsidian(tmp_path, monkeypatch):
    path = tmp_path / "theme.json"
    monkeypatch.setenv("CEREBRO_THEME_FILE", str(path))
    with app.test_client() as client:
        assert client.post("/api/theme", json={"theme": "astro1deep-obsidian"}).status_code == 200
        assert client.get("/api/theme").json["theme"] == "astro1deep-obsidian"
        assert client.post("/api/theme", json={"theme": "unknown"}).status_code == 400


def test_packaging_is_reproducible(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    first = build_archives(root, tmp_path / "one", release_version="1.5.0-beta.3")
    second = build_archives(root, tmp_path / "two", release_version="1.5.0-beta.3")
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()
