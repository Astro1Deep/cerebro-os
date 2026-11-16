import json
from unittest.mock import patch
from cerebro_core.capabilities import probe_ollama_loaded_models

class Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, max_bytes): return self.payload[:max_bytes]


def test_probe_reports_placement_without_claiming_generation_or_gpu():
    payload = {'models': [{'name': 'example', 'digest': 'abc', 'size': 800,
                           'size_vram': 400, 'context_length': 4096,
                           'details': {'quantization_level': 'Q4'}}]}
    with patch('cerebro_core.capabilities.local_http_open', return_value=Response(json.dumps(payload).encode())) as opened:
        result = probe_ollama_loaded_models()
    assert result['api_reachable'] is True
    assert result['loaded_models'][0]['vram_bytes'] == 400
    assert result['generation_verified'] is False
    assert result['gpu_execution_verified'] is False
    assert opened.call_args.args[0].full_url == 'http://127.0.0.1:11434/api/ps'


def test_probe_failures_are_not_reported_as_gpu():
    for payload in (b'{', b'[]', b'x' * (1024 * 1024 + 1)):
        with patch('cerebro_core.capabilities.local_http_open', return_value=Response(payload)):
            assert not probe_ollama_loaded_models()['api_reachable']
    with patch('cerebro_core.capabilities.local_http_open', side_effect=OSError('offline')):
        result = probe_ollama_loaded_models()
    assert result['loaded_models'] == [] and not result['gpu_execution_verified']
