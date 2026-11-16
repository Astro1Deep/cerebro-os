import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from cerebro_core.vector_engine import VectorEngine, _EmbeddingBackend, _fit_dim


def vec(n=0):
    v = [0.0] * 1024
    v[n] = 1.0
    return v


def test_reject_invalid_vector_dimensions_and_nonfinite_values():
    assert _fit_dim([1] * 1023) is None
    assert _fit_dim([1] * 1025) is None
    assert _fit_dim([float('nan')] * 1024) is None
    assert _fit_dim([0] * 1024) is None
    assert _fit_dim(vec()) == vec()


def test_missing_bge_does_not_switch_to_incompatible_native_embeddings():
    backend = _EmbeddingBackend()
    with patch.object(backend, '_encode_ollama', return_value=[None]), \
         patch.object(backend, '_load', side_effect=AssertionError('unsafe fallback')):
        assert backend.encode(['example']) == [None]


@patch("cerebro_core.vector_engine._ollama_embedding_identity", return_value="test-identity")
def test_failed_replacement_keeps_previous_record(_identity):
    with tempfile.TemporaryDirectory() as root:
        engine = VectorEngine(root=Path(root))
        assert engine.upsert('item', vector=vec(), vector_identity='test-identity', text='original')
        real_table = engine.get_table(identity='test-identity')
        class FailingTable:
            def merge_insert(self, *args):
                raise RuntimeError('simulated write failure')
        with patch.object(engine, 'get_table', return_value=FailingTable()):
            assert not engine.upsert('item', vector=vec(1), vector_identity='test-identity', text='replacement')
        rows = real_table.search().where("id = 'item'").limit(10).to_list()
        assert len(rows) == 1 and rows[0]['text'] == 'original'
        assert engine.upsert('item', vector=vec(1), vector_identity='test-identity', text='replacement')
        rows = real_table.search().where("id = 'item'").limit(10).to_list()
        # Reopen after update to avoid a stale table snapshot.
        rows = engine.get_table(identity='test-identity').search().where("id = 'item'").limit(10).to_list()
        assert len(rows) == 1 and rows[0]['text'] == 'replacement'


@patch("cerebro_core.vector_engine._ollama_embedding_identity", return_value="test-identity")
def test_other_tables_never_use_default_table_hot_cache(_identity):
    with tempfile.TemporaryDirectory() as root:
        engine = VectorEngine(root=Path(root))
        assert engine.upsert('same', vector=vec(), vector_identity='test-identity', text='private default')
        assert engine.upsert('same', vector=vec(), vector_identity='test-identity', text='other table', table_name='other')
        class SentinelCache:
            def search(self, *args, **kwargs):
                raise AssertionError('foreign table used default cache')
            def remove(self, *args, **kwargs):
                raise AssertionError('foreign table removed default cache')
        engine.hot_cache = SentinelCache()
        results = engine.search(vector=vec(), vector_identity='test-identity', table_name='other')
        assert [x['text'] for x in results] == ['other table']
        assert engine.delete('same', table_name='other')
        assert engine.get_table(identity='test-identity').count_rows() == 1


def test_legacy_index_is_blocked_and_preserved():
    import pyarrow as pa
    import lancedb
    with tempfile.TemporaryDirectory() as root:
        db = lancedb.connect(root)
        schema = pa.schema([pa.field('id', pa.string()), pa.field('vector', pa.list_(pa.float32(), 1024)),
                            pa.field('text', pa.string()), pa.field('source', pa.string()),
                            pa.field('created_at', pa.float64()), pa.field('metadata_json', pa.string())])
        table = db.create_table('cerebro_unified_memory', schema=schema)
        table.add([{'id':'legacy', 'vector':vec(), 'text':'legacy text', 'source':'test',
                    'created_at':0., 'metadata_json':'{}'}])
        engine = VectorEngine(root=Path(root))
        with patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'):
            assert not engine.upsert('legacy', vector=vec(1), vector_identity='verified-A', text='replacement')
            assert engine.search(vector=vec(), vector_identity='verified-A') == []
            assert not engine.delete('legacy')
        rows = db.open_table('cerebro_unified_memory').search().limit(10).to_list()
        assert len(rows) == 1 and rows[0]['text'] == 'legacy text'


def test_model_change_and_external_vector_without_identity_are_blocked():
    with tempfile.TemporaryDirectory() as root:
        engine = VectorEngine(root=Path(root))
        with patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'):
            assert not engine.upsert('a', vector=vec(), text='a')
            assert engine.upsert('a', vector=vec(), vector_identity='verified-A', text='a')
        with patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-B'):
            assert engine.search(vector=vec(), vector_identity='verified-B') == []
            assert not engine.upsert('a', vector=vec(), vector_identity='verified-B', text='b')
        with patch('cerebro_core.vector_engine._ollama_embedding_identity', side_effect=['verified-A', 'verified-B']):
            assert not engine.upsert('b', vector=vec(), vector_identity='verified-A', text='b')
        with patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'):
            assert engine.get_table(identity='verified-A').count_rows() == 1


def test_two_concurrent_instances_preserve_distinct_writes():
    from concurrent.futures import ThreadPoolExecutor
    with tempfile.TemporaryDirectory() as root:
        with patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'):
            first = VectorEngine(root=Path(root))
            # Create the identified table before the concurrent writers start.
            first.get_table(identity='verified-A', create=True)
            second = VectorEngine(root=Path(root))
            def write(pair):
                engine, index = pair
                return engine.upsert(str(index), vector=vec(), vector_identity='verified-A', text=f'entry {index}')
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(write, [(first, 1), (second, 2)]))
            assert outcomes == [True, True]
            assert first.db.open_table(first.default_table).count_rows() == 2


def test_model_tag_identity_requires_single_sha256_and_local_endpoint():
    import cerebro_core.vector_engine as module
    class Reply:
        def __init__(self, models): self.models = models
        def raise_for_status(self): pass
        def iter_content(self, _size):
            import json
            yield json.dumps({'models': self.models}).encode()
    class Session:
        def __init__(self, models): self.models = models
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            assert url == 'http://127.0.0.1:11434/api/tags'
            assert kwargs['allow_redirects'] is False
            return Reply(self.models)
    cases = [([], None), ([{'name':'bge-m3:latest','digest':'bad'}], None),
             ([{'name':'bge-m3:latest','digest':'a'*64}, {'name':'bge-m3','digest':'b'*64}], None),
             ([{'name':'bge-m3:latest','digest':'a'*64}], 'a'*64)]
    for models, expected in cases:
        with patch.object(module, 'OLLAMA_EMBED_URL', 'http://127.0.0.1:11434/api/embeddings'), \
             patch.object(module, 'local_http_session', return_value=Session(models)):
            identity = module._ollama_embedding_identity()
        if expected:
            assert expected in identity and 'bge-m3:latest' in identity
        else:
            assert identity is None


def test_reindex_to_separate_table_can_resume_without_modifying_legacy():
    import lancedb
    import pyarrow as pa
    with tempfile.TemporaryDirectory() as root:
        db = lancedb.connect(root)
        schema = pa.schema([pa.field('id', pa.string()), pa.field('vector', pa.list_(pa.float32(), 1024)),
                            pa.field('text', pa.string()), pa.field('source', pa.string()),
                            pa.field('created_at', pa.float64()), pa.field('metadata_json', pa.string())])
        legacy = db.create_table('cerebro_unified_memory', schema=schema)
        legacy.add([{'id':str(i), 'vector':vec(), 'text':f'legacy text {i}', 'source':'test',
                     'created_at':0., 'metadata_json':'{}'} for i in range(2)])
        original_rows = legacy.search().limit(10).to_list()
        engine = VectorEngine(root=Path(root))
        with patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'):
            assert engine.upsert('0', vector=vec(), vector_identity='verified-A', text='legacy text 0', table_name='migration_candidate')
            # Simulate interruption and resume into the same independently identified target.
            assert engine.upsert('0', vector=vec(), vector_identity='verified-A', text='legacy text 0', table_name='migration_candidate')
            assert engine.upsert('1', vector=vec(), vector_identity='verified-A', text='legacy text 1', table_name='migration_candidate')
            assert engine.get_table('migration_candidate', identity='verified-A').count_rows() == 2
        assert db.open_table('cerebro_unified_memory').search().limit(10).to_list() == original_rows


def test_concurrent_first_writes_create_one_compatible_index():
    from concurrent.futures import ThreadPoolExecutor
    with tempfile.TemporaryDirectory() as root:
        with patch('cerebro_core.vector_engine._ollama_embedding_identity', return_value='verified-A'):
            engines = [VectorEngine(root=Path(root)), VectorEngine(root=Path(root))]
            def write(index):
                return engines[index].upsert(str(index), vector=vec(), vector_identity='verified-A', text=str(index))
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(write, range(2)))
            assert outcomes == [True, True]
            assert engines[0].db.open_table(engines[0].default_table).count_rows() == 2
