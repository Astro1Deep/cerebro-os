#!/usr/bin/env python3
"""Proposal-only watcher for the Cerebro Central inbox.

This intentionally replaces the legacy autonomous mover. Files remain in the
inbox until the owner reviews a JSON proposal and supplies its exact approval
token. The watcher never touches external volumes or removes duplicates.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from cerebro_core.mayordomo_review import apply_proposal, build_proposal

try:
    from cerebro_core.m4_optimizations import tag_as_background_utility
    tag_as_background_utility()
except Exception:
    pass


CENTRAL_ROOT = Path.home() / "Desktop" / "Cerebro_Central"
INBOX_DIR = CENTRAL_ROOT / "Bandeja_IA"
ORGANIZED_DIR = CENTRAL_ROOT / "Organizado_IA"
PROPOSALS_DIR = CENTRAL_ROOT / "Propuestas_IA"


def _proposal_file(proposal: dict[str, object]) -> Path:
    proposal_id = proposal.get("id")
    if not isinstance(proposal_id, str):
        raise ValueError("La propuesta no tiene identificador")
    return PROPOSALS_DIR / f"{proposal_id}.json"


def write_proposal(proposal: dict[str, object]) -> Path:
    """Persist a review item atomically without modifying the source file."""

    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    target = _proposal_file(proposal)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(proposal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def propose_file(path: Path | str) -> Path:
    proposal = build_proposal(path, inbox_root=INBOX_DIR, organized_root=ORGANIZED_DIR)
    if proposal.get("state") != "pending_approval":
        raise ValueError("Mayordomo sólo puede registrar propuestas pendientes de aprobación")
    target = write_proposal(proposal)
    print(f"[?] Propuesta creada: {target}")
    print(f"    Archivo: {Path(str(proposal['source'])).name}")
    print(f"    Categoría sugerida: {proposal['category']}")
    print("    No se ha movido ningún archivo. Revísala y confirma sólo si estás de acuerdo.")
    return target


def load_proposal(path: Path | str) -> dict[str, object]:
    candidate = Path(path).expanduser().resolve(strict=True)
    if candidate.suffix != ".json" or candidate.parent != PROPOSALS_DIR.resolve(strict=False):
        raise ValueError("La propuesta debe pertenecer a Cerebro_Central/Propuestas_IA")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("El archivo de propuesta no contiene un objeto")
    return data


def approve_file(path: Path | str, token: str) -> dict[str, object]:
    proposal_path = Path(path).expanduser().resolve(strict=True)
    proposal = load_proposal(proposal_path)
    result = apply_proposal(proposal, approval_token=token)
    proposal["state"] = "applied"
    proposal["applied_result"] = result
    proposal_path.write_text(
        json.dumps(proposal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


class ProposalHandler(FileSystemEventHandler):
    def _handle(self, path: str) -> None:
        candidate = Path(path)
        if candidate.name.startswith(".") or candidate.suffix.lower() in {".crdownload", ".download", ".part"}:
            return
        try:
            # Gives browsers a short time to close a newly downloaded file.
            time.sleep(1)
            if candidate.is_file():
                propose_file(candidate)
        except (OSError, ValueError) as error:
            print(f"[!] No se pudo crear una propuesta para {candidate.name}: {error}", file=sys.stderr)

    def on_created(self, event):  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):  # type: ignore[no-untyped-def]
        if not event.is_directory:
            self._handle(event.dest_path)


def watch() -> int:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(ProposalHandler(), str(INBOX_DIR), recursive=False)
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False
        observer.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    observer.start()
    print("[*] Mayordomo en modo revisión: vigila sólo la Bandeja_IA.")
    print("[*] Generará propuestas; no moverá ni borrará archivos automáticamente.")
    try:
        while running:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()
    return 0


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description="Mayordomo de Cerebro Central: propuesta y aprobación")
    commands = command_parser.add_subparsers(dest="command")
    commands.add_parser("watch", help="Vigilar Bandeja_IA y crear propuestas sin mover archivos")
    propose = commands.add_parser("propose", help="Crear una propuesta para un archivo de Bandeja_IA")
    propose.add_argument("file", type=Path)
    approve = commands.add_parser("approve", help="Aplicar una propuesta previamente revisada")
    approve.add_argument("proposal", type=Path)
    approve.add_argument("--approve", required=True, help="Token APPLY-... visible en la propuesta")
    return command_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command in {None, "watch"}:
        return watch()
    if args.command == "propose":
        propose_file(args.file)
        return 0
    if args.command == "approve":
        result = approve_file(args.proposal, args.approve)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
