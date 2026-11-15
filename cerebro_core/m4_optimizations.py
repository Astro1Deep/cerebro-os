"""
M4 Pro Hardware Optimizations & Darwin QoS Governor (idempotente y segura).

Notas de ingeniería y arquitectura:
- La inferencia LLM en `mlx_lm.server` corre en la GPU Metal (ancho de banda unificado de ~273 GB/s).
- macOS Darwin no expone afinidad fija de núcleos (sched_setaffinity), sino Clases de Calidad de
  Servicio (QoS) en el Kernel XNU:
    * QOS_CLASS_USER_INTERACTIVE (0x21): Máxima prioridad, reloj turbo, asegurado en P-Cores.
    * QOS_CLASS_USER_INITIATED (0x19): Prioridad alta, sesgo fuerte hacia P-Cores.
    * QOS_CLASS_UTILITY (0x11): Modo eficiente, asegurado en E-Cores SIN estrangulamiento de I/O de disco.
    * QOS_CLASS_BACKGROUND (0x09): Fondo estricto con I/O throttling.
- `tag_as_performance_critical()` promueve hilos a P-Cores para evitar micro-stutters.
- `tag_as_background_utility()` confina daemons (Mayordomo, Indexadores, Rotator) a E-Cores.
"""
import ctypes
import ctypes.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

_APPLIED = False

# Constantes oficiales de Darwin / sys/qos.h
QOS_CLASS_USER_INTERACTIVE = 0x21
QOS_CLASS_USER_INITIATED   = 0x19
QOS_CLASS_DEFAULT          = 0x15
QOS_CLASS_UTILITY          = 0x11
QOS_CLASS_BACKGROUND       = 0x09


def set_thread_qos(qos_class: int) -> bool:
    """Aplica la clase QoS de Darwin al hilo actual mediante libSystem (sin requerir root)."""
    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)
        fn = getattr(libc, "pthread_set_qos_class_self_np", None)
        if fn is None:
            return False
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        return fn(qos_class, 0) == 0
    except Exception as exc:
        logger.debug("No se pudo fijar QoS Darwin (0x%x): %s", qos_class, exc)
        return False


def tag_as_performance_critical() -> bool:
    """Garantiza ejecución en Performance Cores (P-Cores) para MLX, Audio y Gateway."""
    ok = set_thread_qos(QOS_CLASS_USER_INTERACTIVE)
    if not ok:
        ok = set_thread_qos(QOS_CLASS_USER_INITIATED)
    if ok:
        logger.debug("Hilo clasificado como PERFORMANCE CRITICAL (P-Cores asegurados).")
    return ok


def tag_as_background_utility() -> bool:
    """Confinar hilo a Efficiency Cores (E-Cores) sin restringir ancho de banda de disco."""
    ok = set_thread_qos(QOS_CLASS_UTILITY)
    if ok:
        logger.debug("Hilo clasificado como BACKGROUND UTILITY (E-Cores asegurados, I/O pleno).")
    return ok


def apply_m4_hardware_tuning() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    # 1. PyTorch MPS (embeddings bge-m3) & FAISS OpenMP coexistence
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

    # 2. MLX: sólo si este proceso ya usa MLX (evita cargar Metal en clientes que no infieren).
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            import psutil
            limit = int(psutil.virtual_memory().total * 0.75)
            if hasattr(mx, "set_memory_limit"):
                mx.set_memory_limit(limit)
            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(limit // 4)  # caché acotada: devuelve RAM al SO entre turnos
        except Exception as exc:
            logger.debug("Ajuste MLX omitido: %s", exc)

    # 3. QoS por defecto para procesos interactivos
    tag_as_performance_critical()


if __name__ == "__main__":
    apply_m4_hardware_tuning()
    print("M4 Hardware Tuning & QoS Governor: OK")
