"""A review-first inbox organiser for Cerebro Central.

The module deliberately works on one approved inbox file at a time.  It never
writes to mounted volumes, and does not inspect or
move anything until a caller explicitly requests a proposal or confirms one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Final

from cerebro_core.local_endpoint import (
    configured_local_ai_endpoint, configured_local_ai_model, local_http_open, models_endpoint,
)
from cerebro_runtime import load_runtime_config

CATEGORY_BY_EXTENSION: Final[dict[str, str]] = {
    ".als": "Produccion_Audio",
    ".alp": "Produccion_Audio",
    ".aif": "Produccion_Audio",
    ".aiff": "Produccion_Audio",
    ".flac": "Produccion_Audio",
    ".m4a": "Produccion_Audio",
    ".mid": "Produccion_Audio",
    ".midi": "Produccion_Audio",
    ".mp3": "Produccion_Audio",
    ".wav": "Produccion_Audio",
    ".py": "Desarrollo_Codigo",
    ".js": "Desarrollo_Codigo",
    ".ts": "Desarrollo_Codigo",
    ".json": "Desarrollo_Codigo",
    ".yaml": "Desarrollo_Codigo",
    ".yml": "Desarrollo_Codigo",
    ".md": "Documentos_PDF",
    ".pdf": "Documentos_PDF",
    ".docx": "Documentos_PDF",
    ".txt": "Documentos_PDF",
    ".png": "Imagenes_Diseño",
    ".jpg": "Imagenes_Diseño",
    ".jpeg": "Imagenes_Diseño",
    ".webp": "Imagenes_Diseño",
    ".svg": "Imagenes_Diseño",
    ".dmg": "Instaladores_Archivos",
    ".pkg": "Instaladores_Archivos",
    ".zip": "Instaladores_Archivos",
}

# Categorías semánticas dirigidas exclusivamente para análisis cognitivo LLM y heurístico
SEMANTIC_CATEGORIES: Final[list[str]] = [
    "Facturas",
    "Apuntes_Universidad",
    "Memes",
    "Recibos",
    "Personal",
    "Trabajo",
]
ALLOWED_CATEGORIES: Final[frozenset[str]] = frozenset(
    {*CATEGORY_BY_EXTENSION.values(), *SEMANTIC_CATEGORIES, "Otros", "Papelera_Reciclaje"}
)
SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")

# Reglas heurísticas de emergencia para fallback rápido multi-categoría
HEURISTIC_RULES: Final[list[tuple[str, re.Pattern[str], str]]] = [
    (
        "Facturas",
        re.compile(
            r"\b(factura|facturas|invoice|invoices|cif|nif|total\s+a\s+pagar|base\s+imponible|importe\s+total)\b|\biva\s+\d+%(?!\w)",
            re.IGNORECASE,
        ),
        "Términos de facturación o impuestos detectados",
    ),
    (
        "Recibos",
        re.compile(
            r"\b(recibo|recibos|ticket|tickets|comprobante|justificante|pago\s+con\s+tarjeta|tpv|operacion\s+aprobada)\b",
            re.IGNORECASE,
        ),
        "Ticket de compra o comprobante de pago detectado",
    ),
    (
        "Apuntes_Universidad",
        re.compile(
            r"\b(universidad|examen|examenes|apuntes|asignatura|profesor|catedra|parcial|semestre|facultad|matricula)\b",
            re.IGNORECASE,
        ),
        "Términos académicos universitarios detectados",
    ),
    (
        "Trabajo",
        re.compile(
            r"\b(trabajo|trabajos|reunion|minuta|proyecto|cliente|entregable|roadmap|sprint|jira|presupuesto|brief|propuesta\s+comercial|acta)\b",
            re.IGNORECASE,
        ),
        "Términos corporativos o profesionales detectados",
    ),
    (
        "Personal",
        re.compile(
            r"\b(personal|diario|privado|confidencial|salud|medico|analitica|receta\s+medica|vacaciones)\b",
            re.IGNORECASE,
        ),
        "Documento de índole personal o privada detectado",
    ),
    (
        "Memes",
        re.compile(
            r"\b(meme|memes|chiste|humor|shitpost)\b",
            re.IGNORECASE,
        ),
        "Contenido humorístico detectado",
    ),
]


def _resolve_inside(path: Path | str, root: Path | str, label: str) -> Path:
    resolved_path = Path(path).expanduser().resolve(strict=False)
    resolved_root = Path(root).expanduser().resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} debe permanecer dentro de {resolved_root}") from error
    return resolved_path


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_text_heuristic(path: Path) -> str:
    """Extrae rápidamente texto del archivo usando OCR nativo para imágenes o lectura directa."""
    ext = path.suffix.lower()

    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        ocr_bin = Path(__file__).parent / "mac_ocr"
        if ocr_bin.exists():
            try:
                res = subprocess.run(
                    [str(ocr_bin), str(path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return res.stdout.strip()
            except Exception:
                return ""
        return ""

    if ext == ".pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            extracted: list[str] = []
            for page in reader.pages[:3]:
                t = page.extract_text()
                if t:
                    extracted.append(t)
                if sum(len(x) for x in extracted) >= 4000:
                    break
            pdf_text = "\n".join(extracted)[:4000].strip()
            if pdf_text:
                return pdf_text
        except Exception:
            pass
        return ""

    if ext in [".txt", ".md", ".csv", ".json", ".rtf", ".log", ".docx"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read(4000).strip()
        except Exception:
            return ""

    return ""


def _resolve_active_model(api_url: str) -> str:
    """Resuelve dinámicamente el modelo activo de chat evitando modelos no compatibles como FLUX."""
    try:
        models_url = models_endpoint(api_url)
        req = urllib.request.Request(models_url)
        with local_http_open(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                for m in data.get("data", []):
                    mod_id = m.get("id", "")
                    mod_lower = mod_id.lower()
                    if (
                        "bge" not in mod_lower
                        and "embed" not in mod_lower
                        and "flux" not in mod_lower
                        and "diffusion" not in mod_lower
                    ):
                        return mod_id
    except Exception:
        pass

    return "omlx-default"


def _parse_llm_semantic_category(raw_response: str) -> str | None:
    """Parsea estrictamente la respuesta del LLM exigiendo una categoría de SEMANTIC_CATEGORIES."""
    if not raw_response or not raw_response.strip():
        return None

    cleaned = raw_response.strip()

    # 1. Si viene envuelto en markdown code blocks, extraer el contenido
    if "```" in cleaned:
        match_code = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match_code:
            cleaned = match_code.group(1).strip()
        else:
            match_plain = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match_plain:
                cleaned = match_plain.group(1).strip()

    # 2. Intentar parseo directo como JSON
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "category" in data:
            candidate = str(data["category"]).strip()
            for cat in SEMANTIC_CATEGORIES:
                if cat.lower() == candidate.lower():
                    return cat
    except Exception:
        pass

    # 3. Búsqueda regex de {"category": "..."} o {'category': '...'}
    match = re.search(r'["\']category["\']\s*:\s*["\']([^"\']+)["\']', raw_response, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        for cat in SEMANTIC_CATEGORIES:
            if cat.lower() == candidate.lower():
                return cat

    # 4. Coincidencia exacta de token limpio contra SEMANTIC_CATEGORIES
    token = cleaned.strip("`'\" \n\r\t{}")
    for cat in SEMANTIC_CATEGORIES:
        if cat.lower() == token.lower():
            return cat

    return None


def _call_local_llm_classify(snippet: str) -> str | None:
    """Invoca el LLM local con un prompt estricto y JSON mode pidiendo solo SEMANTIC_CATEGORIES."""
    root = Path(__file__).resolve().parents[1]
    config = load_runtime_config(root)
    api_url = configured_local_ai_endpoint(os.environ.get("CEREBRO_AI_URL") or config.get("CEREBRO_AI_URL"))
    model = configured_local_ai_model(os.environ.get("CEREBRO_AI_MODEL") or config.get("CEREBRO_AI_MODEL"))
    target_model = model or _resolve_active_model(api_url)

    prompt = (
        "Clasifica el siguiente texto asignándole EXACTAMENTE UNA de estas categorías semánticas:\n"
        "- Facturas (facturas comerciales, facturación con impuestos, CIF/NIF)\n"
        "- Apuntes_Universidad (estudio académico, materias, exámenes, universidad)\n"
        "- Memes (contenido humorístico, memes)\n"
        "- Recibos (tickets de compra, justificantes de pago, recibos de caja)\n"
        "- Personal (documentos privados, salud, médicos, familia, diario)\n"
        "- Trabajo (proyectos profesionales, empresa, reuniones laborales, clientes)\n\n"
        "INSTRUCCIÓN ESTRICTA:\n"
        'Responde ÚNICAMENTE con un objeto JSON válido con la estructura: {"category": "<CATEGORIA>"}\n'
        f"donde <CATEGORIA> sea estrictamente una de: {json.dumps(SEMANTIC_CATEGORIES, ensure_ascii=False)}.\n"
        "No incluyas explicaciones, comentarios ni texto fuera del JSON.\n\n"
        f"TEXTO:\n{snippet[:3500]}"
    )

    payload = json.dumps({
        "model": target_model,
        "messages": [
            {"role": "system", "content": "Eres el Mayordomo de Cerebro OS. Clasificador semántico."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 30,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with local_http_open(req, timeout=5) as response:
        res_data = json.loads(response.read().decode())
        raw_ai = res_data["choices"][0]["message"]["content"].strip()
        return _parse_llm_semantic_category(raw_ai)


def semantic_classify(path: Path) -> tuple[str, str]:
    """Feature 1: Enrutamiento cognitivo de Mayordomo (LLM local -> Heurística -> Extensión)."""
    ext = path.suffix.lower()

    # Si es un tipo puramente técnico (código, audio, instalador), enrutamos directamente por extensión
    # para garantizar la preservación de reglas técnicas de archivo y compatibilidad con tests.
    if ext in CATEGORY_BY_EXTENSION and CATEGORY_BY_EXTENSION[ext] in (
        "Desarrollo_Codigo",
        "Produccion_Audio",
        "Instaladores_Archivos",
    ):
        cat = CATEGORY_BY_EXTENSION[ext]
        return cat, f"Clasificado por extensión técnica {ext} ({cat})."

    content_snippet = ""
    try:
        content_snippet = _extract_text_heuristic(path)
    except Exception:
        pass

    # 1. Si hay texto relevante (> 20 caracteres), intentar clasificación con LLM local
    if content_snippet and len(content_snippet) > 20:
        try:
            ai_category = _call_local_llm_classify(content_snippet)
            if ai_category and ai_category in SEMANTIC_CATEGORIES:
                return ai_category, f"Clasificado inteligentemente vía Motor Local ({ai_category})."
        except Exception:
            # Fallo esperado si el daemon LLM está offline, timeout o en test unitario
            pass

        # 2. Heurística de emergencia (último recurso para contenido extraído)
        clean_name = path.name.replace("_", " ")
        for cat, pattern, reason in HEURISTIC_RULES:
            if pattern.search(content_snippet) or pattern.search(clean_name):
                return cat, f"Heurística de emergencia: {reason}."

    # Heurística de emergencia por nombre de archivo si el contenido era breve
    clean_name = path.name.replace("_", " ")
    for cat, pattern, reason in HEURISTIC_RULES:
        if pattern.search(clean_name):
            return cat, f"Heurística de emergencia: {reason}."

    # Heurística especial para imágenes tipo meme (bajo texto OCR)
    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        if len(content_snippet) < 50:
            return "Memes", "Imagen con poco texto OCR (Heurística Meme)."

    # 3. Fallback final por extensión de archivo
    cat = CATEGORY_BY_EXTENSION.get(ext, "Otros")
    return cat, f"Clasificado por extensión {ext or 'sin extensión'}."



def _safe_filename(filename: str) -> str:
    candidate = SAFE_NAME.sub("_", Path(filename).name).strip(" ._")
    if not candidate:
        raise ValueError("El nombre del archivo no permite una propuesta segura")
    return candidate[:180]


def check_expiration(source_path: Path) -> bool:
    """Feature 4: Auto-Trash para instaladores viejos"""
    ext = source_path.suffix.lower()
    if ext in [".dmg", ".pkg", ".zip"]:
        # Si tiene más de 30 días
        age_days = (time.time() - source_path.stat().st_mtime) / (60 * 60 * 24)
        if age_days > 30:
            return True
    return False


def build_proposal(
    source: Path | str,
    *,
    inbox_root: Path | str,
    organized_root: Path | str,
) -> dict[str, Any]:
    """Create a transparent, non-mutating organisation proposal for one file."""

    source_path = _resolve_inside(source, inbox_root, "El archivo de origen")
    if not source_path.is_file():
        raise FileNotFoundError(f"No existe un archivo regular en la bandeja: {source_path}")

    # Feature 4: Expiración
    is_expired = check_expiration(source_path)
    
    if is_expired:
        category = "Papelera_Reciclaje"
        reason = "El instalador/zip tiene más de 30 días de antigüedad y probablemente ya no es útil."
    else:
        # Feature 1: Clasificación Semántica
        category, reason = semantic_classify(source_path)
        
    if category not in ALLOWED_CATEGORIES:
        raise ValueError("La categoría propuesta no está permitida")
        
    source_hash = sha256_file(source_path)
    archive_root = Path(organized_root).expanduser().resolve(strict=False)
    base_name = _safe_filename(source_path.name)
    destination = archive_root / category / base_name
    
    action_type = "move"
    
    if destination.exists():
        if sha256_file(destination) == source_hash:
            # Feature 3: Smart Deduplication (Hardlinks o eliminación)
            # Ya existe exactamente el mismo archivo en el destino.
            action_type = "deduplicate"
            reason += " (El archivo ya existe en el destino. Se recomienda desduplicarlo para ahorrar espacio SSD)."
        else:
            stem, suffix = destination.stem, destination.suffix
            destination = destination.with_name(f"{stem}--{source_hash[:12]}{suffix}")
            
    destination = _resolve_inside(destination, archive_root, "El destino propuesto")
    proposal_id = source_hash[:16]

    return {
        "schema_version": 2,
        "id": proposal_id,
        "state": "pending_approval",
        "requires_user_confirmation": True,
        "approval_token": f"APPLY-{proposal_id}",
        "source": str(source_path),
        "source_sha256": source_hash,
        "inbox_root": str(Path(inbox_root).expanduser().resolve(strict=False)),
        "organized_root": str(archive_root),
        "category": category,
        "destination": str(destination),
        "action_type": action_type,
        "reason": reason,
        "questions": [
            f"¿Confirmas procesar {source_path.name} -> {category}?",
            "Puedes rechazarla o cambiar la categoría antes de aprobar.",
        ],
    }


def apply_proposal(proposal: dict[str, Any], *, approval_token: str) -> dict[str, Any]:
    """Apply one exact, user-approved proposal within Cerebro Central only."""

    if not isinstance(proposal, dict):
        raise ValueError("La propuesta debe ser un objeto")
    proposal_id = proposal.get("id")
    expected_token = f"APPLY-{proposal_id}"
    if not isinstance(proposal_id, str) or proposal.get("approval_token") != expected_token:
        raise ValueError("La propuesta no tiene un identificador de aprobación válido")
    if approval_token != expected_token:
        raise PermissionError("Hace falta el token exacto de confirmación de esta propuesta")
    if proposal.get("state") != "pending_approval" or proposal.get("requires_user_confirmation") is not True:
        raise PermissionError("La propuesta no está pendiente de aprobación explícita")

    inbox_root = Path(str(proposal.get("inbox_root", ""))).expanduser().resolve(strict=False)
    organized_root = Path(str(proposal.get("organized_root", ""))).expanduser().resolve(strict=False)
    source = _resolve_inside(str(proposal.get("source", "")), inbox_root, "El origen")
    destination = _resolve_inside(str(proposal.get("destination", "")), organized_root, "El destino")
    category = proposal.get("category")
    action_type = proposal.get("action_type", "move")
    
    if category not in ALLOWED_CATEGORIES or destination.parent != organized_root / category:
        raise ValueError("La propuesta contiene una categoría o un destino no permitido")
    if not source.is_file():
        raise FileNotFoundError("El archivo de origen ya no está disponible")
        
    expected_hash = proposal.get("source_sha256")
    if not isinstance(expected_hash, str) or sha256_file(source) != expected_hash:
        raise ValueError("El archivo cambió desde la propuesta; revísalo de nuevo")

    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Feature 3: Smart Deduplication
    if action_type == "deduplicate" and destination.exists():
        # Ya existe en el destino y es idéntico. Solo borramos el origen para ahorrar espacio.
        # Si queremos preservarlo en el inbox visualmente pero linkeado:
        os.unlink(source)
        os.link(destination, source)
    elif category == "Papelera_Reciclaje":
        # Feature 4: Trash action
        os.unlink(source)
    else:
        if destination.exists():
            raise FileExistsError("El destino ya existe; crea una propuesta nueva")
        os.replace(source, destination)
        
        # Feature 2: Indexación Inmediata (Memoria Vectorial)
        if destination.suffix.lower() in [".md", ".txt", ".pdf"]:
            try:
                from cerebro_core.persistent_memory import PersistentMemoryStore
                with open(destination, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(5000)
                # Instanciamos la base de datos de memoria
                store = PersistentMemoryStore(Path("~/.mlx-stack/knowledge").expanduser())
                store.remember(f"Documento clasificado por Mayordomo: {destination.name}\n\n{content}", source=str(destination))
            except Exception as e:
                print(f"Aviso: No se pudo indexar vectorialmente {destination.name}: {e}")
                
    destination_hash = sha256_file(destination) if category != "Papelera_Reciclaje" else expected_hash
    if destination_hash != expected_hash:
        raise RuntimeError("La verificación de integridad posterior al movimiento falló")
        
    return {
        "id": proposal_id,
        "state": "applied",
        "source_sha256": expected_hash,
        "destination": str(destination),
        "action": action_type
    }
