# Guía de usuario de Cerebro OS

[English](GUIDE.md) · [Inicio](../README_ES.md) · [Notas de versión](RELEASE_NOTES.md)

## 1. Instalar y arrancar

Usa Python 3.12 para esta beta. El instalador crea un entorno virtual, instala las dependencias del perfil elegido y escribe un `config.env` privado. Una instalación nueva requiere acceso al registro de paquetes. Elige un destino nuevo o vacío. La instalación no arranca servicios ni descarga pesos de modelos.

| Plataforma | Motor elegido | Comando desde el repositorio |
|---|---|---|
| macOS Apple Silicon | MLX-LM | `python3 install.py --target ~/cerebro-beta` |
| macOS Intel o Linux | Ollama | `python3 install.py --target ~/cerebro-beta` |
| Windows x64, PowerShell | Ollama | `py -3.12 install.py --target "$env:USERPROFILE/cerebro-beta"` |

Windows, Linux e Intel son destinos del instalador. La inferencia y la compatibilidad con dispositivos dependen del equipo y del modelo. Usa `--dry-run` para ver el destino, motor y endpoint antes de escribir. Puedes elegir el modelo con `--model ID_MODELO`. Para copiar solo los archivos usa `--no-venv` y gestiona el entorno Python por separado.

Desde la **carpeta instalada**, usa terminales separadas si quieres abrir ambas interfaces.

```sh
venv/bin/python cerebro_runtime.py composer
venv/bin/python cerebro_runtime.py jarvis
venv/bin/python -m cerebro_cli doctor --offline --json
```

En Windows sustituye `venv/bin/python` por `.\venv\Scripts\python.exe`. Abre Composer en <http://127.0.0.1:8082/> y Jarvis en <http://127.0.0.1:8083/>. Pulsa Ctrl+C para cerrar un servicio en primer plano. Abre Composer por HTTP para acceder a los recursos de la interfaz, MIDI, detección tonal y asesoría.

## 2. Motores y opciones de conectividad

| Servicio | Dirección local por defecto | Uso |
|---|---|---|
| MLX-LM | `http://127.0.0.1:8080/v1` | API de modelos para Apple Silicon |
| Ollama | `http://127.0.0.1:11434/v1` | API de modelos locales compatible con chat OpenAI |
| Composer | `http://127.0.0.1:8082` | Armonía, MIDI y asesoría opcional |
| Jarvis | `http://127.0.0.1:8083` | Chat de texto y diagnóstico |
| BeatOS | `http://127.0.0.1:8888` o puerto anunciado | Catálogo externo y mantenimiento opcional |
| Gateway de desarrollo | `http://127.0.0.1:8081/v1` si se instaló por separado | Proxy local compatible con OpenAI |

En Apple Silicon, arranca un servidor de modelos compatible.

```sh
venv/bin/mlx_lm.server --host 127.0.0.1 --port 8080 --model ID_MODELO
```

Sustituye `ID_MODELO` por la carpeta de un modelo MLX descargado o por su identificador. El identificador puede provocar una consulta al registro. En otra terminal ejecuta `curl http://127.0.0.1:8080/v1/models`. La ruta local permite usar los pesos ya descargados. Si usas Ollama, instálalo y ejecútalo aparte, y prepara un modelo local apropiado, por ejemplo `ollama pull llama3.2`. Consulta los nombres instalados con `ollama list`.

Especifica en el `config.env` de la instalación el endpoint completo de chat y el identificador que devuelve el servidor. Elige un par a la vez.

```ini
CEREBRO_AI_URL=http://127.0.0.1:11434/v1/chat/completions
CEREBRO_AI_MODEL=llama3.2
```

Para MLX-LM, usa el puerto 8080 y el ID correspondiente. La configuración admite HTTP en direcciones de bucle local y puertos 8000–11999 para `/v1/chat/completions`. En macOS/Linux, protege `config.env` con `chmod 600 config.env`. Para otro servidor local compatible, como oMLX o llama.cpp, configura su puerto e identificador manualmente y verifica `/v1/models` y una petición de chat. La beta no realiza detección ni conmutación automática de motores. Empieza con un solo modelo, un contexto moderado y memoria unificada disponible. La respuesta de salud comprueba el transporte. Prueba una generación corta para verificar la inferencia.

Hermes se ejecuta por separado. Configura su proveedor compatible con OpenAI con la URL base del mismo motor, terminada en `/v1`, y el mismo identificador. `python -m cerebro_cli hermes status` comprueba su configuración local sin arrancarlo. Consulta la documentación de [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py) y [Ollama](https://docs.ollama.com/api/openai-compatibility).

El lanzador portátil restringe las interfaces web al propio equipo. Telegram es la opción móvil de la beta y cursa mensajes a través de sus servidores. Mantén Jarvis y el motor de inferencia en bucle local. El código conserva un modo LAN de Composer para desarrollo supervisado. Esta guía cubre el uso en bucle local.

## 3. Composer, MIDI y hardware

En Composer puedes seleccionar notas o usar el detector tonal del micrófono del navegador. Elige tónica y escala y solicita un análisis o una progresión. El análisis armónico, el piano, la cuadrícula 8×8 tipo Push y la importación/exportación MIDI funcionan sin el motor. La detección tonal ocurre en el navegador y requiere permiso y una entrada utilizable. La asesoría generativa sí necesita el motor.

Exporta un archivo MIDI estándar e impórtalo en un DAW como Ableton Live. La cuadrícula Push sirve para componer visualmente. El control físico de Push, Ableton Link, reloj MIDI, OSC, MIDI inalámbrico y ruteo de interfaces dependen de programas y equipos externos. Usa BeatOS para los flujos MPC/SP-404 y consulta las instrucciones de exportación por modelo. Verifica la carga en el equipo y firmware de destino antes de depender de ella en una sesión.

## 4. BeatOS, discos externos y dashboard

Inicia [BeatOS](https://github.com/Astro1Deep/BeatOS-TheMakerLicious) por separado con un servidor compatible. Monta los medios y genera o actualiza el catálogo. Cerebro lee el JSON del puente v2 y toma de él el puerto y las rutas SQLite. La ruta por defecto es `~/.mlx-stack/beatos_bridge.json`. Si guardas el archivo en otro lugar, define `CEREBRO_BEATOS_BRIDGE` en la terminal. Usa una versión de BeatOS que publique el puente, o crea un JSON local con esta estructura y rutas absolutas que existan en tu equipo.

```json
{
  "schema": "cerebro-beatos-bridge-v2",
  "api_base": "http://127.0.0.1:8888",
  "root_dir": "/ruta/absoluta/a/biblioteca",
  "database_path": "/ruta/absoluta/a/BeatOS/data/mpc_sample.db",
  "mounted_libraries": [
    {
      "id": "biblioteca-secundaria",
      "name": "Segunda biblioteca",
      "mount_path": "/ruta/absoluta/a/segunda-biblioteca",
      "db_path": "/ruta/absoluta/a/segunda-biblioteca/samples.db",
      "table_name": "samples"
    }
  ]
}
```

Consulta `GET /api/system/status` en Jarvis para ver el inventario anunciado. Selecciona RAG y busca por nombre de muestra. Jarvis solo consulta índices SQLite anunciados y legibles. La tabla necesita columnas compatibles de nombre y ruta, por ejemplo `filename` y `full_path`. Los índices se abren en solo lectura. Una biblioteca desmontada desaparece de los resultados activos. Usa el dashboard de BeatOS para reindexar, limpiar auxiliares AppleDouble, exportar kits y mantener tarjetas según el equipo y firmware. Para compartir el tema, apunta `CEREBRO_THEME_FILE` al archivo que lee tu versión de BeatOS. Las versiones que usan la ruta de Antigravity leen `~/.gemini/antigravity/scratch/ecosystem_theme.json`. Por defecto Jarvis guarda su tema dentro de la instalación.

## 5. Conocimiento, Telegram y diagnóstico

- `python -m cerebro_cli status --offline --json` muestra los endpoints configurados sin sondearlos. Quita `--offline` para comprobar los servicios. `doctor --offline` diagnostica la instalación. `doctor --repair` es una acción explícita para servicios supervisados que ya existan en macOS.
- `python -m cerebro_cli web crawl https://example.org` pone una fuente pública en revisión. Consulta `web pending` y usa `web approve ID` antes de buscar con `web search CONSULTA`. La descarga web requiere Internet. La búsqueda semántica utiliza un motor local de embeddings opcional y puede recurrir a búsqueda textual cuando el módulo lo admite. Para usar el conector Ollama, instala y arranca el modelo de embeddings correspondiente. Conserva el modelo original de cada índice vectorial existente.
- Para activar Telegram, ejecuta `python -m cerebro_cli telegram init --owner-id ID_NUMERICO`. Añade tu propio `TELEGRAM_TOKEN` al `config.env` y ejecuta `python cerebro_runtime.py telegram` con el Python de la instalación. Usa el chat privado del bot y consulta `/start` para ver sus funciones. La captura de pantalla y cámara requiere permisos locales temporales y permisos del sistema. Mantén esos permisos desactivados para el chat habitual.
- Ejecuta `python mlx-mayordomo.py propose ARCHIVO` con un archivo de la bandeja local `Cerebro_Central/Bandeja_IA`. Revisa la propuesta antes de usar `approve PROPUESTA --approve TOKEN`. El vigilante local genera propuestas y exige un token de aprobación explícito para mover un archivo.
- En Apple Silicon, `python -m cerebro_cli eval ID_MODELO --max-tokens 64 --json` informa de la latencia del primer token, el rendimiento de generación y las métricas de memoria disponibles. Compara con el mismo modelo, prompt, contexto y estado de caché. La optimización específica por hardware, los dashboards de desarrollo, los puentes móviles antiguos y los prototipos Mix/DJ siguen en desarrollo.
- Ante un puerto ocupado, identifica el proceso que ya escucha. Si el modelo carga pero no genera, revisa su identificador, reduce el contexto y utiliza un solo motor. En Windows usa un destino con permiso de escritura y ejecuta directamente el Python del entorno. Si faltan bibliotecas, revisa el esquema del puente, las rutas absolutas, el montaje y los nombres de las tablas.

## 6. Actualizar o volver a la versión anterior

Instala cada beta en su propia carpeta y copia solo los ajustes que necesites. Detén los servicios antiguos antes de arrancar la versión nueva. Conserva la instalación y sus datos hasta probar ambas interfaces, una generación corta y tus flujos MIDI y de bibliotecas. Para volver atrás, cierra los servicios nuevos y arranca la instalación anterior.

### Preparar una candidata de memoria vectorial

Para examinar una base sin modificarla ni crear una candidata:

```text
python -m cerebro_core.memory_migration --source RUTA_SQLITE --destination DIRECTORIO_NUEVO --kind memory
```

Para documentos web, usa `--kind web-approved`: solo selecciona los aprobados. Para construir la candidata separada añade `--apply`. Necesita Ollama local y bge-m3 instalados con digest identificable. El origen se lee en una transacción SQLite de solo lectura; el destino no puede contener el origen ni estar dentro de su carpeta. Si cambian el contenido de origen o el digest durante la reanudación, se rechaza y hay que utilizar una candidata nueva. Se guardan el hash de la instantánea y el contrato del modelo, sin activar ni reemplazar índices anteriores. Una ejecución interrumpida puede repetirse con la misma fuente y modelo. La candidata debe revisarse antes de cualquier activación, que esta herramienta no realiza.
