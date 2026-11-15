import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from cerebro_core.persistent_memory import PersistentMemoryStore


def test_all_sqlite_handles_are_closed_after_operations():
    opened = []
    real_connect = sqlite3.connect
    class TrackedConnection(sqlite3.Connection):
        closed_by_store = False
        def close(self):
            self.closed_by_store = True
            super().close()
    def tracked_connect(*args, **kwargs):
        connection = real_connect(*args, factory=TrackedConnection, **kwargs)
        opened.append(connection)
        return connection
    class FakeVectors:
        def __init__(self, **_kwargs): pass
        def upsert(self, **_kwargs): return True
        def search(self, **_kwargs): return []
        def delete(self, **_kwargs): return True
    with tempfile.TemporaryDirectory() as folder:
        with patch('cerebro_core.persistent_memory.VectorEngine', FakeVectors), patch('cerebro_core.persistent_memory.sqlite3.connect', tracked_connect):
            store = PersistentMemoryStore(Path(folder))
            entry = store.remember('Prueba de memoria y cierre')
            assert store.list_recent()[0]['id'] == entry['id']
            assert store._bm25_search('memoria')
            assert store.delete(entry['id'])
            assert opened and all(connection.closed_by_store for connection in opened)
            store.database_path.unlink()
