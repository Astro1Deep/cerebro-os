from __future__ import annotations

import argparse
import csv
import ctypes
import dataclasses
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psutil
from rich.console import Console
from rich.table import Table

console = Console()
HISTORY_FILE = Path.home() / ".mlx-stack" / "experiments" / "cerebro_eval_history.csv"
DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_PROMPT = "Explain quantum computing in 50 words."


@dataclass
class EvaluationReport:
    model_path: str
    ttft_ms: float
    throughput_tps: float
    peak_unified_ram_mb: float
    metal_active_ram_mb: float
    load_ssd_read_mb: float
    load_ssd_bandwidth_mbps: float
    gen_ssd_read_mb: float
    gen_ssd_bandwidth_mbps: float
    tokens_generated: int
    generated_text: str
    model_load_ms: float = 0.0
    generation_ms: float = 0.0
    backend: str = "mlx"


def calculate_ssd_bandwidth_mbps(read_bytes: int, duration_s: float) -> tuple[float, float]:
    read_mb = max(read_bytes, 0) / (1024 * 1024)
    return read_mb, read_mb / duration_s if duration_s > 0 else 0.0


def get_ram_usage() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_swap_usage() -> float:
    return psutil.swap_memory().used / (1024 * 1024)


def get_disk_io() -> int:
    counters = psutil.disk_io_counters()
    return counters.read_bytes if counters is not None else 0


def _get_process_disk_read_bytes() -> int | None:
    if sys.platform != "darwin":
        return None

    buffer = ctypes.create_string_buffer(288)
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    proc_pid_rusage = libproc.proc_pid_rusage
    proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    proc_pid_rusage.restype = ctypes.c_int
    result = proc_pid_rusage(os.getpid(), 4, ctypes.byref(buffer))
    if result != 0:
        return None
    return ctypes.c_uint64.from_buffer(buffer, 144).value


def _read_bytes_snapshot() -> int:
    process_bytes = _get_process_disk_read_bytes()
    return process_bytes if process_bytes is not None else get_disk_io()


def _format_prompt(tokenizer: object, prompt: str) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template and "messages[::-1]" not in chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if chat_template:
        # This template pattern segfaults in the installed Transformers Jinja compiler.
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def _is_local_jang_model(path: Path) -> bool:
    return any(
        (path / filename).is_file()
        for filename in ("jang_config.json", "jjqf_config.json", "jang_cfg.json", "mxq_config.json")
    )


def _load_evaluation_model(model_path: str, use_jang: bool) -> tuple[object, object, bool]:
    local_path = Path(model_path).expanduser()
    local_jang = local_path.is_dir() and _is_local_jang_model(local_path)
    if not use_jang and not local_jang:
        from mlx_lm import load

        model, tokenizer = load(model_path)
        return model, tokenizer, False

    if local_path.is_dir():
        jang_path = local_path
    else:
        try:
            from huggingface_hub import snapshot_download
            try:
                from huggingface_hub.errors import HfHubHTTPError as HfHubError
            except ImportError:
                HfHubError = Exception
        except ImportError as error:
            raise RuntimeError(
                "La descarga de modelos JANG requiere huggingface_hub."
            ) from error
        try:
            jang_path = Path(snapshot_download(repo_id=model_path))
        except HfHubError as error:
            raise RuntimeError(
                f"No se pudo descargar el modelo JANG {model_path!r}: {error}"
            ) from error

    if not _is_local_jang_model(jang_path):
        raise ValueError(f"No se encontró configuración JANG en el modelo {model_path!r}.")

    try:
        from jang_tools.loader import load_jang_model
    except ImportError as error:
        raise RuntimeError(
            "Para cargar modelos JANG instala las dependencias opcionales con "
            "`python -m pip install -r requirements-jang.txt`."
        ) from error

    try:
        model, tokenizer = load_jang_model(jang_path)
    except ImportError as error:
        raise RuntimeError(
            "Falta una dependencia del loader JANG; reinstala requirements-jang.txt."
        ) from error
    return model, tokenizer, True


def get_thermal_state() -> str:
    return "N/A"


def get_mlx_version() -> str:
    runtime_alias = os.environ.get("CEREBRO_RUNTIME_ALIAS")
    if runtime_alias:
        return runtime_alias
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return "Unknown"


def save_to_history(data_dict: dict[str, object]) -> None:
    fieldnames = [
        "timestamp", "runtime_version", "model_revision", "quantization",
        "expert_retention", "context_tokens", "ram_before_mb", "ram_peak_mb",
        "swap_mb", "ttft_sec", "prompt_tok_sec", "gen_tok_sec", "cache_hit",
        "ssd_reads_mb", "temperature", "tool_success",
    ]
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_FILE.is_file()
    with HISTORY_FILE.open(mode="a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)


class CerebroEvaluator:
    def __init__(
        self,
        model_path: str,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 64,
        temp: float = 0.0,
        use_jang: bool = False,
    ) -> None:
        if not model_path.strip():
            raise ValueError("El identificador o ruta del modelo no puede estar vacío.")
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío.")
        if not 1 <= max_tokens <= 8192:
            raise ValueError("--max-tokens debe estar entre 1 y 8192.")
        if not 0.0 <= temp <= 2.0:
            raise ValueError("--temp debe estar entre 0.0 y 2.0.")
        self.model_path = model_path
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temp = temp
        self.use_jang = use_jang

    def evaluate(self) -> EvaluationReport:
        try:
            import mlx.core as mx
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except ImportError as error:
            raise RuntimeError("La evaluación requiere mlx y mlx-lm instalados en este entorno.") from error

        mx.reset_peak_memory()
        ram_before = get_ram_usage()
        swap_before = get_swap_usage()

        load_io_before = _read_bytes_snapshot()
        load_started = time.perf_counter()
        model, tokenizer, is_jang = _load_evaluation_model(self.model_path, self.use_jang)
        mx.synchronize()
        load_duration = time.perf_counter() - load_started
        load_io_after = _read_bytes_snapshot()
        load_read_mb, load_bandwidth = calculate_ssd_bandwidth_mbps(
            load_io_after - load_io_before, load_duration
        )

        prompt = _format_prompt(tokenizer, self.prompt)
        prompt_tokens = len(tokenizer.encode(prompt))

        sampler = make_sampler(temp=self.temp)
        logits_processors = make_logits_processors(
            repetition_penalty=1.15,
            repetition_context_size=20,
        )
        generation_io_before = _read_bytes_snapshot()
        generation_started = time.perf_counter()
        first_token_at: float | None = None
        generated_tokens = 0
        generated_parts: list[str] = []
        peak_rss_mb = get_ram_usage()

        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            if first_token_at is None:
                mx.synchronize()
                first_token_at = time.perf_counter()
            generated_tokens = max(generated_tokens + 1, int(response.generation_tokens))
            generated_parts.append(response.text)
            peak_rss_mb = max(peak_rss_mb, get_ram_usage())

        mx.synchronize()
        generation_finished = time.perf_counter()
        generation_io_after = _read_bytes_snapshot()
        if generated_tokens == 0 or first_token_at is None:
            raise RuntimeError("MLX no generó ningún token; no se puede emitir una evaluación válida.")

        ttft_ms = max(0.0, (first_token_at - generation_started) * 1000.0)
        decode_duration = generation_finished - first_token_at
        throughput = (generated_tokens - 1) / decode_duration if generated_tokens > 1 and decode_duration > 0 else 0.0
        generation_read_mb, generation_bandwidth = calculate_ssd_bandwidth_mbps(
            generation_io_after - generation_io_before,
            generation_finished - generation_started,
        )
        metal_active_mb = mx.get_active_memory() / (1024 * 1024)
        metal_peak_mb = mx.get_peak_memory() / (1024 * 1024)
        process_peak_mb = max(peak_rss_mb, get_ram_usage())
        report = EvaluationReport(
            model_path=self.model_path,
            ttft_ms=ttft_ms,
            throughput_tps=throughput,
            peak_unified_ram_mb=max(process_peak_mb, metal_peak_mb),
            metal_active_ram_mb=metal_active_mb,
            load_ssd_read_mb=load_read_mb,
            load_ssd_bandwidth_mbps=load_bandwidth,
            gen_ssd_read_mb=generation_read_mb,
            gen_ssd_bandwidth_mbps=generation_bandwidth,
            tokens_generated=generated_tokens,
            generated_text="".join(generated_parts),
            model_load_ms=load_duration * 1000.0,
            generation_ms=(generation_finished - generation_started) * 1000.0,
            backend="jang" if is_jang else "mlx",
        )

        total_time = generation_finished - generation_started
        telemetry_record = {
            "timestamp": datetime.now().isoformat(),
            "runtime_version": get_mlx_version(),
            "model_revision": self.model_path.rstrip("/").split("/")[-1],
            "quantization": "Q4" if "4bit" in self.model_path.lower() else "Unknown",
            "expert_retention": "N/A",
            "context_tokens": prompt_tokens,
            "ram_before_mb": round(ram_before, 2),
            "ram_peak_mb": round(report.peak_unified_ram_mb, 2),
            "swap_mb": round(max(0.0, get_swap_usage() - swap_before), 2),
            "ttft_sec": round(report.ttft_ms / 1000.0, 3),
            "prompt_tok_sec": round(prompt_tokens / (report.ttft_ms / 1000.0), 2) if report.ttft_ms > 0 else 0.0,
            "gen_tok_sec": round(report.throughput_tps, 2),
            "cache_hit": "N/A",
            "ssd_reads_mb": round(report.load_ssd_read_mb + report.gen_ssd_read_mb, 2),
            "temperature": get_thermal_state(),
            "tool_success": "N/A",
        }
        save_to_history(telemetry_record)
        del model, tokenizer
        import gc

        gc.collect()
        mx.clear_cache()
        return report


def render_report(report: EvaluationReport) -> None:
    table = Table(title="CEREBRO-EVAL · Telemetría")
    table.add_column("Métrica")
    table.add_column("Valor", justify="right")
    table.add_row("Modelo", report.model_path)
    table.add_row("TTFT", f"{report.ttft_ms:.2f} ms")
    table.add_row("Decode throughput", f"{report.throughput_tps:.2f} tok/s")
    table.add_row("RAM unificada pico", f"{report.peak_unified_ram_mb:.2f} MB")
    table.add_row("Metal RAM activa", f"{report.metal_active_ram_mb:.2f} MB")
    table.add_row("Lectura SSD carga", f"{report.load_ssd_read_mb:.2f} MB")
    table.add_row("Lectura SSD generación", f"{report.gen_ssd_read_mb:.2f} MB")
    table.add_row("Tokens generados", str(report.tokens_generated))
    console.print(table)
    safe_text = "".join(
        character for character in report.generated_text
        if character in "\n\t" or character.isprintable()
    )
    print(f"\nGeneración:\n{safe_text or 'Sin texto generado.'}")
    console.print(f"Historial: {HISTORY_FILE}")


def run_cli_eval(
    model: str,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 64,
    temp: float = 0.0,
    as_json: bool = False,
    use_jang: bool = False,
) -> int:
    try:
        report = CerebroEvaluator(model, prompt, max_tokens, temp, use_jang).evaluate()
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error en CEREBRO-EVAL: {error}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(dataclasses.asdict(report), ensure_ascii=False))
    else:
        render_report(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cerebro-eval")
    parser.add_argument("model", help="Model repository ID or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt de evaluación")
    parser.add_argument("--max-tokens", type=int, default=64, help="Número máximo de tokens")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperatura de muestreo (0.0–2.0)")
    parser.add_argument(
        "--jang",
        action="store_true",
        help="Cargar como modelo JANG (para IDs remotos requiere descarga explícita)",
    )
    parser.add_argument("--json", action="store_true", help="Emitir telemetría JSON")
    return parser


def main(args: list[str] | None = None) -> int:
    parsed = build_parser().parse_args(args)
    return run_cli_eval(
        model=parsed.model,
        prompt=parsed.prompt,
        max_tokens=parsed.max_tokens,
        temp=parsed.temp,
        as_json=parsed.json,
        use_jang=parsed.jang,
    )


if __name__ == "__main__":
    raise SystemExit(main())
