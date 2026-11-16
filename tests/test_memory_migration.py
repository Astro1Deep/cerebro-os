import sqlite3
from contextlib import closing
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from cerebro_core.memory_migration import migrate, MigrationError
from cerebro_core.vector_engine import VectorEngine


def create_source(path):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE memories (id TEXT PRIMARY KEY, text TEXT, source TEXT, created_at REAL)')
        db.executemany('INSERT INTO memories VALUES (?,?,?,?)', [('1','Uno de prueba','explicit_user',1.0),('2','Dos de prueba','explicit_user',2.0)])


def test_dry_run_never_creates_destination():
    with tempfile.TemporaryDirectory() as base:
        source = Path(base) / 'source' / 'memory.db'; source.parent.mkdir(); create_source(source)
        target = Path(base) / 'candidate'
        result = migrate(source, target, 'memory')
        assert result['source_records'] == 2 and not result['activated'] and not target.exists()


def test_interrupted_candidate_resumes_and_source_changes_are_rejected():
    with tempfile.TemporaryDirectory() as base:
        source = Path(base) / 'source' / 'memory.db'; source.parent.mkdir(); create_source(source)
        target = Path(base) / 'candidate'
        v = [1.0] + [0.0] * 1023
        original = VectorEngine.upsert
        count = [0]
        def interrupted(self, *args, **kwargs):
            count[0] += 1
            return False if count[0] == 2 else original(self, *args, **kwargs)
        with patch('cerebro_core.memory_migration._ollama_embedding_identity', return_value='verified-A'), patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'), patch.object(VectorEngine, 'get_embedding', return_value=v):
            with patch.object(VectorEngine, 'upsert', interrupted):
                with pytest.raises(MigrationError): migrate(source, target, 'memory', apply=True)
            with closing(sqlite3.connect(source)) as db, db:
                db.execute("UPDATE memories SET text='changed' WHERE id='1'")
            with pytest.raises(MigrationError, match='changed'):
                migrate(source, target, 'memory', apply=True)
            with closing(sqlite3.connect(source)) as db, db:
                db.execute("UPDATE memories SET text='Uno de prueba' WHERE id='1'")
            result = migrate(source, target, 'memory', apply=True)
            assert result['candidate_records'] == 2 and not result['activated']
            assert migrate(source, target, 'memory', apply=True)['candidate_records'] == 2
            assert VectorEngine(root=target / 'lancedb', default_table='migration_candidate').get_table(identity='verified-A').count_rows() == 2


def test_model_change_blocks_resume_and_destination_cannot_overlap_source():
    with tempfile.TemporaryDirectory() as base:
        source = Path(base) / 'source' / 'memory.db'; source.parent.mkdir(); create_source(source)
        target = Path(base) / 'candidate'
        with pytest.raises(MigrationError, match='separate'):
            migrate(source, source.parent / 'candidate', 'memory')
        with patch('cerebro_core.memory_migration._ollama_embedding_identity', return_value='verified-A'), patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'), patch.object(VectorEngine, 'get_embedding', return_value=[1.] + [0.] * 1023):
            migrate(source, target, 'memory', apply=True)
        with patch('cerebro_core.memory_migration._ollama_embedding_identity', return_value='verified-B'):
            with pytest.raises(MigrationError, match='changed'):
                migrate(source, target, 'memory', apply=True)


def test_terminated_process_leaves_resumable_identity_bound_candidate():
    import os
    import subprocess
    import sys
    with tempfile.TemporaryDirectory() as base:
        source = Path(base) / 'source' / 'memory.db'; source.parent.mkdir(); create_source(source)
        target = Path(base) / 'candidate'
        code = r'''
import os, sys
from pathlib import Path
from unittest.mock import patch
from cerebro_core.memory_migration import migrate
from cerebro_core.vector_engine import VectorEngine
old = VectorEngine.upsert
count = [0]
def die_after_first(self, *args, **kwargs):
    result = old(self, *args, **kwargs)
    count[0] += 1
    if count[0] == 1: os._exit(23)
    return result
with patch('cerebro_core.memory_migration._ollama_embedding_identity', return_value='verified-A'), patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'), patch.object(VectorEngine, 'get_embedding', return_value=[1.]+[0.]*1023), patch.object(VectorEngine, 'upsert', die_after_first):
    migrate(Path(sys.argv[1]), Path(sys.argv[2]), 'memory', apply=True)
'''
        result = subprocess.run([sys.executable, '-c', code, str(source), str(target)], timeout=30, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
        assert result.returncode == 23
        with patch('cerebro_core.memory_migration._ollama_embedding_identity', return_value='verified-A'), patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'), patch.object(VectorEngine, 'get_embedding', return_value=[1.] + [0.] * 1023):
            assert migrate(source, target, 'memory', apply=True)['candidate_records'] == 2


def test_approved_web_only_migrates_and_candidate_is_not_activated():
    with tempfile.TemporaryDirectory() as base:
        source = Path(base) / 'source' / 'web.db'; source.parent.mkdir()
        with closing(sqlite3.connect(source)) as db, db:
            db.execute('CREATE TABLE documents (id TEXT, text TEXT, url TEXT, title TEXT, approved_at REAL, status TEXT)')
            db.executemany('INSERT INTO documents VALUES (?,?,?,?,?,?)', [('a','Approved content','https://example.com','Approved',1,'approved'), ('b','Pending content','https://example.com','Pending',None,'pending')])
        report = migrate(source, Path(base) / 'candidate', 'web-approved')
        assert report['source_records'] == 1 and not report['activated']


def test_candidate_lock_prevents_concurrent_writer_then_can_resume():
    with tempfile.TemporaryDirectory() as base:
        source = Path(base) / 'source' / 'memory.db'; source.parent.mkdir(); create_source(source)
        target = Path(base) / 'candidate'
        with patch('cerebro_core.memory_migration._ollama_embedding_identity', return_value='verified-A'), patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'), patch.object(VectorEngine, 'get_embedding', return_value=[1.] + [0.] * 1023):
            assert migrate(source, target, 'memory', apply=True)['candidate_records'] == 2
            with closing(sqlite3.connect(target / 'migration.sqlite3')) as lock:
                lock.execute('BEGIN IMMEDIATE')
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    migrate(source, target, 'memory', apply=True)
                lock.rollback()
            assert migrate(source, target, 'memory', apply=True)['candidate_records'] == 2
