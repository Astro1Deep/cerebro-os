#!/usr/bin/env python3
"""CLI entrypoint for Cerebro beta patcher."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cerebro_release.patcher import patch_payload

def main() -> int:
    parser = argparse.ArgumentParser(description="Actualizador (patcher) de Cerebro")
    parser.add_argument("--target", required=True, help="Directorio de la instalación existente a actualizar")
    parser.add_argument("--dry-run", action="store_true", help="Simula el parche sin modificar")
    parser.add_argument("--backup-root", default=None, help="Directorio raíz para copias de seguridad")

    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    target_dir = Path(args.target).expanduser().resolve()

    try:
        result = patch_payload(
            source_dir=source_dir,
            target_dir=target_dir,
            backup_root=Path(args.backup_root) if args.backup_root else None,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        sys.stderr.write(f"{e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
