#!/usr/bin/env python3
"""CLI entrypoint for Cerebro beta payload installer."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from cerebro_release.installer import install_payload

def main() -> int:
    parser = argparse.ArgumentParser(description="Instalador del paquete beta de Cerebro")
    parser.add_argument("--target", required=True, help="Directorio destino para la instalación")
    parser.add_argument("--system", default=platform.system(), help="Sistema operativo objetivo")
    parser.add_argument("--machine", default=platform.machine(), help="Arquitectura del hardware objetivo")
    parser.add_argument("--dry-run", action="store_true", help="Simula la instalación sin escribir")
    parser.add_argument("--no-venv", action="store_true", help="No crea el entorno virtual venv")
    parser.add_argument("--no-deps", action="store_true", help="No instala dependencias")
    parser.add_argument("--python", default=None, help="Ruta al intérprete de Python")
    parser.add_argument("--model", default=None, help="Modelo seleccionado")

    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    target_dir = Path(args.target).expanduser().resolve()

    if source_dir == target_dir:
        sys.stderr.write("El origen y el destino deben ser distintos.\n")
        return 1

    try:
        result = install_payload(
            source_dir=source_dir,
            target_dir=target_dir,
            system=args.system,
            machine=args.machine,
            dry_run=args.dry_run,
            create_venv=not args.no_venv,
            install_dependencies=not args.no_deps and not args.no_venv,
            python_executable=Path(args.python) if args.python else None,
            model_name=args.model,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        sys.stderr.write(f"Error durante la instalación: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
