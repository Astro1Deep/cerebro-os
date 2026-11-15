import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from cerebro_core.vector_engine import VectorEngine, _fit_dim


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


def test_failed_replacement_keeps_previous_record():
    with tempfile.TemporaryDirectory() as root:
        engine = VectorEngine(root=Path(root))
        assert engine.upsert('item', vector=vec(), text='original')
        real_table = engine.get_table()
        class FailingTable:
            def merge_insert(self, *args):
                raise RuntimeError('simulated write failure')
        with patch.object(engine, 'get_table', return_value=FailingTable()):
            assert not engine.upsert('item', vector=vec(1), text='replacement')
        rows = real_table.search().where("id = 'item'").limit(10).to_list()
        assert len(rows) == 1 and rows[0]['text'] == 'original'
        assert engine.upsert('item', vector=vec(1), text='replacement')
        rows = real_table.search().where("id = 'item'").limit(10).to_list()
        # Reopen after update to avoid a stale table snapshot.
        rows = engine.get_table().search().where("id = 'item'").limit(10).to_list()
        assert len(rows) == 1 and rows[0]['text'] == 'replacement'


def test_other_tables_never_use_default_table_hot_cache():
    with tempfile.TemporaryDirectory() as root:
        engine = VectorEngine(root=Path(root))
        assert engine.upsert('same', vector=vec(), text='private default')
        assert engine.upsert('same', vector=vec(), text='other table', table_name='other')
        class SentinelCache:
            def search(self, *args, **kwargs):
                raise AssertionError('foreign table used default cache')
            def remove(self, *args, **kwargs):
                raise AssertionError('foreign table removed default cache')
        engine.hot_cache = SentinelCache()
        results = engine.search(vector=vec(), table_name='other')
        assert [x['text'] for x in results] == ['other table']
        assert engine.delete('same', table_name='other')
        assert engine.get_table().count_rows() == 1
