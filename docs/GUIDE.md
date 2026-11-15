# Cerebro OS user guide

[Español](GUIA.md) · [Repository overview](../README.md) · [Release notes](RELEASE_NOTES.md)

## 1. Install and run

Use Python 3.12 for this beta. The installer creates a virtual environment, installs the selected dependency set and writes a private `config.env`. It requires registry access for a fresh installation. Choose a new or empty destination. The runtime does not start background services or download model weights during installation.

| Platform | Engine selected by the installer | Launch from the source checkout |
|---|---|---|
| macOS Apple Silicon | MLX-LM | `python3 install.py --target ~/cerebro-beta` |
| macOS Intel or Linux | Ollama | `python3 install.py --target ~/cerebro-beta` |
| Windows x64, PowerShell | Ollama | `py -3.12 install.py --target "$env:USERPROFILE/cerebro-beta"` |

The Windows, Linux and Intel paths are installer targets. Live inference and device compatibility depend on each host and model. Use `--dry-run` to see the target, engine and endpoint before writing. Use `--model MODEL_ID` to set the desired model during installation. For a script-only copy use `--no-venv`; it requires a separately managed Python environment.

From the **installed directory**, run these commands in separate terminals to use both interfaces.

```sh
venv/bin/python cerebro_runtime.py composer
venv/bin/python cerebro_runtime.py jarvis
venv/bin/python -m cerebro_cli doctor --offline --json
```

On Windows use `.\venv\Scripts\python.exe` in place of `venv/bin/python`. Open Composer at <http://127.0.0.1:8082/> and Jarvis at <http://127.0.0.1:8083/>. Close a foreground service with Ctrl+C. Composer serves its asset and API URLs over HTTP. Open the served page for MIDI, live pitch detection and model advice.

## 2. Model servers and connection options

| Service | Default local address | Purpose |
|---|---|---|
| MLX-LM | `http://127.0.0.1:8080/v1` | Apple Silicon model API |
| Ollama | `http://127.0.0.1:11434/v1` | Local model API with OpenAI-compatible chat |
| Composer | `http://127.0.0.1:8082` | Harmony, MIDI and optional advice |
| Jarvis | `http://127.0.0.1:8083` | Text chat and diagnostics |
| BeatOS | `http://127.0.0.1:8888` or its advertised port | Optional external catalog and maintenance |
| Development gateway | `http://127.0.0.1:8081/v1` if separately installed | Optional local OpenAI-compatible proxy |

On Apple Silicon, start a compatible model server with:

```sh
venv/bin/mlx_lm.server --host 127.0.0.1 --port 8080 --model MODEL_ID
```

Replace `MODEL_ID` with a downloaded MLX model directory or a model ID. Using a model ID may contact its registry. In another terminal check `curl http://127.0.0.1:8080/v1/models`. A local directory avoids a fresh weight download. On machines using Ollama, install and start Ollama separately and pull an appropriate local model, for example `ollama pull llama3.2`. Inspect the installed model names with `ollama list`.

In the installation's `config.env`, set the full chat endpoint and the model advertised by the server. Use one pair at a time.

```ini
CEREBRO_AI_URL=http://127.0.0.1:11434/v1/chat/completions
CEREBRO_AI_MODEL=llama3.2
```

For MLX-LM use port 8080 and its model ID. The configuration accepts HTTP loopback hosts and ports 8000–11999 for `/v1/chat/completions`. On macOS/Linux keep `config.env` owner-only with `chmod 600 config.env`. For a separately installed local OpenAI-compatible server such as oMLX or llama.cpp, configure its actual port and model identifier manually and verify both `/v1/models` and a chat request. Automatic discovery and switching of model engines are outside this beta. Start with one model, a modest context and enough free unified memory. A health response confirms API transport. Run a short generation to check model inference.

Hermes runs separately. Point its OpenAI-compatible provider to the same model server base URL ending in `/v1` and use the server's model identifier. `python -m cerebro_cli hermes status` inspects its local configuration without starting it. See the upstream [MLX-LM server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py) and [Ollama API](https://docs.ollama.com/api/openai-compatibility) documentation.

The portable launcher binds the web interfaces to loopback. Telegram is the beta's optional phone workflow and sends messages through Telegram's servers. Keep Jarvis and the inference service local. A legacy Composer LAN mode exists in the source for supervised development. This guide covers loopback use only.

## 3. Composer, MIDI and hardware

Open Composer, select notes or use the browser microphone's pitch detector, choose a tonic and scale, then request chord analysis or build a progression. The analysis, progression builder, piano, Push-style 8×8 grid and MIDI import/export work independently of model availability. The microphone detector operates in the browser and depends on browser permission and a usable input device. Model advice needs the local server.

Export a Standard MIDI File and import it into a DAW such as Ableton Live. The Push grid is a visual composition tool. Physical Push control, Ableton Link, MIDI clock, OSC, wireless MIDI and audio-interface routing depend on external software and hardware. Use BeatOS for the MPC/SP-404 workflow and consult its device-specific export instructions. Verify loading on the target hardware and firmware before relying on an export in a session.

## 4. BeatOS, external drives and the dashboard

Start [BeatOS](https://github.com/Astro1Deep/BeatOS-TheMakerLicious) separately with a compatible server build. Mount your media and create or refresh its catalog. Cerebro reads the v2 bridge JSON and uses the port and SQLite paths it advertises. Set `CEREBRO_BEATOS_BRIDGE` in the shell if the file is stored elsewhere. The default is `~/.mlx-stack/beatos_bridge.json`. The release of BeatOS you use must export the bridge, or you can create a local bridge manually with this shape. Use absolute paths that exist on your computer.

```json
{
  "schema": "cerebro-beatos-bridge-v2",
  "api_base": "http://127.0.0.1:8888",
  "root_dir": "/absolute/path/to/library",
  "database_path": "/absolute/path/to/BeatOS/data/mpc_sample.db",
  "mounted_libraries": [
    {
      "id": "second-library",
      "name": "Second sample library",
      "mount_path": "/absolute/path/to/second-library",
      "db_path": "/absolute/path/to/second-library/samples.db",
      "table_name": "samples"
    }
  ]
}
```

Use `GET /api/system/status` in Jarvis for the advertised library inventory. Select RAG mode and search by sample name. Only advertised, readable SQLite indexes are queried. The table must have compatible name and path columns, such as `filename` and `full_path`. Jarvis opens the indexes read-only. Unmounted libraries disappear from its active results. Open the BeatOS dashboard to reindex, perform AppleDouble maintenance, export kits or manage cards using the drive and firmware appropriate to your setup. For optional shared themes, set `CEREBRO_THEME_FILE` to the file your BeatOS build reads. BeatOS builds that use the Antigravity path read `~/.gemini/antigravity/scratch/ecosystem_theme.json`. Jarvis otherwise stores its theme in the installed directory.

## 5. Knowledge, Telegram and diagnostics

- `python -m cerebro_cli status --offline --json` lists configured endpoints without probing them. Omit `--offline` for health checks. Use `doctor --offline` for installation checks. `doctor --repair` is an explicit macOS action for existing supervised services.
- `python -m cerebro_cli web crawl https://example.org` stages a public source. Use `web pending` to review it and `web approve ID` before using `web search QUERY`. Crawling requires Internet access. Semantic search works with an optional local embedding engine and may fall back to text search where supported. Install and start Ollama with the matching embedding model if you want that connector. Keep existing vector indexes with their original embedding model.
- To enable Telegram, use `python -m cerebro_cli telegram init --owner-id NUMERIC_ID`. Put your own `TELEGRAM_TOKEN` in the installed `config.env`, then run `python cerebro_runtime.py telegram` using the installation's Python. Use a private bot chat and review `/start` for active functions. Screen and camera capture require explicit, time-limited local permissions plus operating-system permissions. Leave those grants disabled for routine chat.
- Run `python mlx-mayordomo.py propose FILE` for a file in the local `Cerebro_Central/Bandeja_IA` inbox. Review its proposal before using `approve PROPOSAL --approve TOKEN`. The local watcher creates proposals and requires an explicit approval token to move a file.
- On Apple Silicon, `python -m cerebro_cli eval MODEL_ID --max-tokens 64 --json` reports first-token latency, generation throughput and available memory metrics. Compare runs with the same model, prompt, context and cache state. Hardware-specific optimization, developer dashboards, legacy mobile bridges and the old mix/DJ placeholders remain development surfaces.
- For a port conflict, identify the process already using that port. If a model loads but generation fails, check the model identifier, reduce the context and run one engine at a time. For Windows path or permission issues, use a user-writable folder and call the environment's Python directly. For missing library results, verify the bridge schema, absolute database paths, mounted media and table names.

## 6. Update or restore

Install each beta in its own directory, then copy only the deliberate settings you need. Stop the old foreground services before launching the new version. Keep the old directory and its local state until you have opened both services, tested a short generation and checked your MIDI and sample library workflow. To roll back, stop the new services and restart the previous installation.
