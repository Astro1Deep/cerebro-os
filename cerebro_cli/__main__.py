"""Entrada de la CLI local de Cerebro."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from auto_doctor import build_report as build_doctor_report
from auto_doctor import exit_code_for_report, print_human as print_doctor_report
from cerebro_core.access_control import RemoteAccessPolicy, save_remote_access_policy
from cerebro_core.branding import play_welcome_motion, render_welcome, render_welcome_ansi, terminal_supports_ansi
from cerebro_core.capabilities import build_capability_report
from cerebro_core.commercial_research import EvidenceProposalBoard
from cerebro_core.hermes import inspect_hermes
from cerebro_core.profiles import list_profiles
from cerebro_core.remote_capabilities import (
    CapabilityPolicyError,
    SENSITIVE_CAPABILITIES,
    grant_capability,
    load_remote_capability_grants,
    revoke_capability,
    save_remote_capability_grants,
)
from cerebro_core.remote_capture import RemoteCaptureError, capture_camera, capture_screen, discard_capture
from cerebro_core.skills import list_skill_manifests
from cerebro_core.status import build_status, render_status
from cerebro_core.supervisor import recovery_decision
from cerebro_core.web_knowledge import CrawlLimits, WebKnowledgeError, WebKnowledgeCrawler, as_json, default_store


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cerebro", description="Control local de Cerebro")
    subcommands = parser.add_subparsers(dest="command", required=True)
    status = subcommands.add_parser("status", help="Mostrar estado sin modificar servicios")
    status.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    status.add_argument("--offline", action="store_true", help="No consultar servicios locales")
    status.add_argument("--json", action="store_true", help="Emitir JSON estable")
    capabilities = subcommands.add_parser("capabilities", help="Detectar capacidades sin modificar el equipo")
    capabilities.add_argument("--json", action="store_true", help="Emitir JSON estable")
    skill = subcommands.add_parser("skill", help="Inspeccionar skills declarativas")
    skill_commands = skill.add_subparsers(dest="skill_command", required=True)
    skill_list = skill_commands.add_parser("list", help="Listar skills sin ejecutarlas")
    skill_list.add_argument("--skills-dir", type=Path, default=Path(__file__).resolve().parents[1] / "skills")
    skill_list.add_argument("--json", action="store_true", help="Emitir JSON estable")
    profile = subcommands.add_parser("profile", help="Consultar perfiles de operación")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    profile_list = profile_commands.add_parser("list", help="Listar perfiles y sus límites")
    profile_list.add_argument("--json", action="store_true", help="Emitir JSON estable")
    hermes = subcommands.add_parser("hermes", help="Inspeccionar el adaptador opcional Hermes")
    hermes_commands = hermes.add_subparsers(dest="hermes_command", required=True)
    hermes_status = hermes_commands.add_parser("status", help="Comprobar Hermes sin leer credenciales")
    hermes_status.add_argument("--hermes-home", type=Path, default=Path.home() / ".hermes")
    hermes_status.add_argument("--json", action="store_true", help="Emitir JSON estable")
    web = subcommands.add_parser(
        "web", help="Adquirir fuentes web sólo mediante una petición explícita y trazable"
    )
    web_commands = web.add_subparsers(dest="web_command", required=True)
    web_crawl = web_commands.add_parser("crawl", help="Recorrer un sitio público y crear fuentes pendientes")
    web_crawl.add_argument("url", help="URL pública inicial; se respeta robots.txt y el mismo host")
    web_crawl.add_argument("--max-pages", type=int, default=10, help="Máximo de páginas (1–25; por defecto 10)")
    web_crawl.add_argument("--max-total-mb", type=int, default=5, help="Máximo total descargado en MB (1–20)")
    web_crawl.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    web_crawl.add_argument("--json", action="store_true", help="Emitir JSON estable")
    web_pending = web_commands.add_parser("pending", help="Listar fuentes todavía no incorporadas al RAG")
    web_pending.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    web_pending.add_argument("--json", action="store_true", help="Emitir JSON estable")
    web_approve = web_commands.add_parser("approve", help="Aprobar una fuente pendiente para el RAG local")
    web_approve.add_argument("document_id", help="Identificador de fuente pendiente")
    web_approve.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    web_approve.add_argument("--json", action="store_true", help="Emitir JSON estable")
    web_search = web_commands.add_parser("search", help="Buscar sólo entre fuentes locales aprobadas")
    web_search.add_argument("query", help="Consulta local")
    web_search.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    web_search.add_argument("--json", action="store_true", help="Emitir JSON estable")
    commercial = subcommands.add_parser(
        "commercial", help="Preparar propuestas comerciales sólo con fuentes web aprobadas"
    )
    commercial_commands = commercial.add_subparsers(dest="commercial_command", required=True)
    commercial_proposal = commercial_commands.add_parser(
        "proposal", help="Crear o aprobar una propuesta; nunca ejecuta ventas, pagos ni envíos"
    )
    proposal_commands = commercial_proposal.add_subparsers(dest="proposal_command", required=True)
    proposal_create = proposal_commands.add_parser(
        "create", help="Crear una propuesta pendiente a partir de dos fuentes aprobadas"
    )
    proposal_create.add_argument("--title", required=True, help="Título de la propuesta")
    proposal_create.add_argument("--hypothesis", required=True, help="Hipótesis a validar")
    proposal_create.add_argument(
        "--source-id", action="append", required=True, help="ID de una fuente ya aprobada; repetir dos o más veces"
    )
    proposal_create.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    proposal_create.add_argument("--json", action="store_true", help="Emitir JSON estable")
    proposal_approve = proposal_commands.add_parser(
        "approve", help="Aprobar una propuesta preparada; no envía ni cobra nada"
    )
    proposal_approve.add_argument("proposal_id", help="Identificador de propuesta")
    proposal_approve.add_argument("--approver", required=True, help="Alias de quien aprueba")
    proposal_approve.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    proposal_approve.add_argument("--json", action="store_true", help="Emitir JSON estable")
    telegram = subcommands.add_parser(
        "telegram", help="Preparar localmente la política privada de Telegram"
    )
    telegram_commands = telegram.add_subparsers(dest="telegram_command", required=True)
    telegram_init = telegram_commands.add_parser(
        "init", help="Crear una política de propietario sin abrir acceso remoto"
    )
    telegram_init.add_argument("--owner-id", type=int, required=True, help="ID numérico del propietario")
    telegram_init.add_argument("--operator-id", type=int, action="append", default=[], help="Administrador adicional")
    telegram_init.add_argument("--root-alias", default="Astro One", help="Alias local no sensible")
    telegram_init.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    telegram_init.add_argument("--json", action="store_true", help="Emitir JSON estable")
    telegram_grant = telegram_commands.add_parser(
        "grant", help="Autorizar temporalmente una captura sensible desde este Mac"
    )
    telegram_grant.add_argument("--capability", choices=sorted(SENSITIVE_CAPABILITIES), required=True)
    telegram_grant.add_argument("--minutes", type=int, required=True, help="Duración entre 1 y 1440 minutos")
    telegram_grant.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    telegram_grant.add_argument("--json", action="store_true", help="Emitir JSON estable")
    telegram_revoke = telegram_commands.add_parser(
        "revoke", help="Revocar localmente una capacidad de captura remota"
    )
    telegram_revoke.add_argument("--capability", choices=sorted(SENSITIVE_CAPABILITIES), required=True)
    telegram_revoke.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    telegram_revoke.add_argument("--json", action="store_true", help="Emitir JSON estable")
    telegram_capability_status = telegram_commands.add_parser(
        "status", help="Consultar las capacidades de captura autorizadas localmente"
    )
    telegram_capability_status.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    telegram_capability_status.add_argument("--json", action="store_true", help="Emitir JSON estable")
    telegram_capture_test = telegram_commands.add_parser(
        "capture-test", help="Probar localmente los permisos de cámara o pantalla sin enviar nada a Telegram"
    )
    telegram_capture_test.add_argument("--capability", choices=sorted(SENSITIVE_CAPABILITIES), required=True)
    telegram_capture_test.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    doctor = subcommands.add_parser("doctor", help="Ejecutar diagnóstico local no destructivo")
    doctor.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    doctor.add_argument("--offline", action="store_true", help="No consultar puertos locales")
    doctor.add_argument(
        "--repair",
        action="store_true",
        help="Reiniciar sólo servicios locales supervisados que estén caídos",
    )
    doctor.add_argument("--json", action="store_true", help="Emitir JSON estable")
    supervisor = subcommands.add_parser("supervisor", help="Planificar recuperación sin actuar sobre procesos")
    supervisor_commands = supervisor.add_subparsers(dest="supervisor_command", required=True)
    supervisor_plan = supervisor_commands.add_parser("plan", help="Evaluar el presupuesto de reinicios")
    supervisor_plan.add_argument("--failures", type=int, required=True, help="Fallos en la ventana actual")
    supervisor_plan.add_argument("--max-failures", type=int, default=2, help="Máximo de fallos permitidos")
    supervisor_plan.add_argument("--json", action="store_true", help="Emitir JSON estable")
    welcome = subcommands.add_parser("welcome", help="Mostrar la entrada ASCII de Astro One Deep")
    welcome_visual = welcome.add_mutually_exclusive_group()
    welcome_visual.add_argument("--motion", action="store_true", help="Reproducir un pulso visual local y breve en una terminal")
    welcome_visual.add_argument("--ansi", action="store_true", help="Aplicar color violeta y azul en una terminal interactiva")
    eval_cmd = subcommands.add_parser("eval", help="Medir inferencia MLX y telemetría de memoria y disco")
    eval_cmd.add_argument("model", help="Identificador del modelo en HuggingFace o ruta local")
    eval_cmd.add_argument(
        "--prompt",
        default="Explain quantum computing in 50 words.",
        help="Prompt de prueba para evaluar TTFT y throughput",
    )
    eval_cmd.add_argument("--max-tokens", type=int, default=64, help="Número máximo de tokens a generar")
    eval_cmd.add_argument("--temp", type=float, default=0.0, help="Temperatura de muestreo")
    eval_cmd.add_argument("--jang", action="store_true", help="Cargar un modelo JANG preconvertido")
    eval_cmd.add_argument("--json", action="store_true", help="Emitir telemetría en formato JSON")
    
    return parser


def _print_operational_welcome() -> None:
    print(render_welcome())
    print("Comandos principales:")
    print("  cerebro status        Estado de los servicios locales")
    print("  cerebro doctor        Diagnóstico no destructivo")
    print("  cerebro --help        Todos los comandos disponibles")


def _format_capability_expiry(expiry: object) -> str:
    return expiry.isoformat().replace("+00:00", "Z") if expiry is not None else "desactivada"


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        _print_operational_welcome()
        return 0
    args = _build_parser().parse_args(arguments)
    if args.command == "status":
        report = build_status(args.base_dir, probe_network=not args.offline)
        print(render_status(report, as_json=args.json))
        return 0
    if args.command == "capabilities":
        report = build_capability_report()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"Plataforma: {report['platform']} ({report['architecture']})")
            print(f"Motor preferente: {report['engine_policy']['primary']}")
        return 0
    if args.command == "skill" and args.skill_command == "list":
        report = {"skills": list_skill_manifests(args.skills_dir)}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print("Skills registradas:" if report["skills"] else "No hay skills registradas.")
            for skill in report["skills"]:
                print(f"- {skill['name']} {skill['version']} ({skill['state']})")
        return 0
    if args.command == "profile" and args.profile_command == "list":
        report = {"profiles": list_profiles()}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            for profile in report["profiles"]:
                print(f"- {profile['name']}: {profile['purpose']} [{profile['external_actions']}]")
        return 0
    if args.command == "hermes" and args.hermes_command == "status":
        report = inspect_hermes(args.hermes_home)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"Hermes: {report['hermes']['state']}")
            print(f"Conector local: {report['connector']['state']}")
        return 0
    if args.command == "web":
        store = default_store(args.base_dir)
        try:
            if args.web_command == "crawl":
                limits = CrawlLimits(
                    max_pages=args.max_pages,
                    max_total_bytes=args.max_total_mb * 1024 * 1024,
                    max_page_bytes=min(1024 * 1024, args.max_total_mb * 1024 * 1024),
                )
                report = WebKnowledgeCrawler(store, limits=limits).crawl(args.url)
                if args.json:
                    print(as_json(report))
                else:
                    print(
                        f"Fuentes pendientes: {report['pages_pending']} · "
                        f"bloqueadas por robots.txt: {report['pages_blocked_robots']}"
                    )
                    if report["pending_ids"]:
                        print("Revisa y aprueba con: cerebro web approve <ID>")
                        print("IDs: " + ", ".join(report["pending_ids"]))
                    if report["errors"]:
                        print("Avisos: " + "; ".join(report["errors"]))
                return 0
            if args.web_command == "pending":
                report = {"documents": store.list_documents(status="pending")}
                print(as_json(report) if args.json else "\n".join(
                    f"{item['id']} · {item['title'] or 'sin título'} · {item['url']}" for item in report["documents"]
                ) or "No hay fuentes pendientes.")
                return 0
            if args.web_command == "approve":
                document = store.approve(args.document_id)
                report = {"document": {key: value for key, value in document.items() if key != "text"}}
                print(as_json(report) if args.json else f"Fuente aprobada para el RAG local: {document['id']}")
                return 0
            if args.web_command == "search":
                report = {"documents": store.search(args.query)}
                print(as_json(report) if args.json else "\n".join(
                    f"{item['title'] or 'sin título'}\n{item['url']}\n{item['excerpt']}" for item in report["documents"]
                ) or "No hay fuentes locales aprobadas que coincidan.")
                return 0
        except WebKnowledgeError as error:
            print(f"No se adquirió conocimiento web: {error}", file=sys.stderr)
            return 1
    if args.command == "commercial" and args.commercial_command == "proposal":
        knowledge = default_store(args.base_dir)
        board = EvidenceProposalBoard(args.base_dir.resolve() / "commercial" / "commercial.sqlite", knowledge)
        try:
            if args.proposal_command == "create":
                proposal = board.create_pending(
                    title=args.title,
                    hypothesis=args.hypothesis,
                    source_ids=args.source_id,
                )
            else:
                proposal = board.approve(args.proposal_id, approver=args.approver)
        except (ValueError, WebKnowledgeError) as error:
            print(f"No se modificó la propuesta comercial: {error}", file=sys.stderr)
            return 1
        report = {"proposal": proposal}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"Propuesta {proposal['id']}: {proposal['status']}")
        return 0
    if args.command == "telegram":
        base_dir = args.base_dir.resolve()
        if args.telegram_command == "init":
            policy_path = base_dir / "telegram_roles.json"
            if policy_path.exists():
                print("La política de Telegram ya existe; no se ha sustituido.", file=sys.stderr)
                return 1
            policy = RemoteAccessPolicy(
                owner_id=args.owner_id,
                operators=frozenset(args.operator_id),
                root_alias=args.root_alias,
            )
            save_remote_access_policy(policy_path, policy)
            report = {
                "owner_id": policy.owner_id,
                "operators": sorted(policy.operators),
                "policy_path": str(policy_path),
                "remote_operations": ["ask", "status", "doctor"],
            }
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print("Política privada de Telegram creada. Operaciones: ask, status, doctor.")
            return 0
        capability_path = base_dir / "telegram_capabilities.json"
        try:
            grants = load_remote_capability_grants(capability_path)
            if args.telegram_command == "grant":
                grants = grant_capability(grants, args.capability, args.minutes)
                save_remote_capability_grants(capability_path, grants)
            elif args.telegram_command == "revoke":
                grants = revoke_capability(grants, args.capability)
                save_remote_capability_grants(capability_path, grants)
            elif args.telegram_command == "capture-test":
                capture_dir = base_dir / "runtime" / "telegram-captures"
                captured = capture_screen(capture_dir) if args.capability == "screen" else capture_camera(capture_dir)
                discard_capture(captured, capture_dir)
                print(f"Permiso local de {args.capability} comprobado; no se envió ninguna imagen.")
                return 0
        except (CapabilityPolicyError, RemoteCaptureError) as error:
            print(f"No se modificó ni envió nada: {error}", file=sys.stderr)
            return 1
        report = {
            "capabilities": {
                capability: _format_capability_expiry(grants.expiry_for(capability))
                for capability in sorted(SENSITIVE_CAPABILITIES)
            },
            "policy_path": str(capability_path),
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        elif args.telegram_command == "grant":
            print(f"Capacidad temporal autorizada: {args.capability} hasta {report['capabilities'][args.capability]}.")
        elif args.telegram_command == "revoke":
            print(f"Capacidad revocada: {args.capability}.")
        else:
            print("Capacidades de captura remota:")
            for capability, expiry in report["capabilities"].items():
                print(f"- {capability}: {expiry}")
        return 0
    if args.command == "doctor":
        from auto_doctor import repair_local_services

        if args.offline and args.repair:
            print("--offline y --repair no se pueden combinar.", file=sys.stderr)
            return 2
        report = build_doctor_report(args.base_dir, probe_network=not args.offline)
        
        if args.repair:
            if not args.json:
                print("⚡ Detectada degradación de servicios. Aplicando protocolos de auto-recuperación (Auto-Repair)...")
            repair = repair_local_services(report)
            
            report = build_doctor_report(args.base_dir, probe_network=not args.offline)
            report["repair"] = repair
            
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print_doctor_report(report)
        return exit_code_for_report(report)
    if args.command == "supervisor" and args.supervisor_command == "plan":
        report = recovery_decision(failures_in_window=args.failures, max_failures=args.max_failures)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"Estado: {report['state']} ({report['reason']})")
        return 0
    if args.command == "welcome":
        if args.motion:
            play_welcome_motion(sys.stdout)
            return 0
        if args.ansi and terminal_supports_ansi(sys.stdout):
            print(render_welcome_ansi())
            return 0
        print(render_welcome())
        return 0
    if args.command == "eval":
        from cerebro_core.evaluation import run_cli_eval

        return run_cli_eval(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
            as_json=args.json,
            use_jang=args.jang,
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
