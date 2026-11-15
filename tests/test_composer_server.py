import asyncio
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

STACK_DIR = str(Path(__file__).resolve().parent.parent)
if STACK_DIR not in sys.path:
    sys.path.insert(0, STACK_DIR)

from composer_server import analyze_notes, composer_bind_host, create_app, load_lan_token


class ComposerServerTests(unittest.TestCase):
    def test_composer_allows_a_cold_local_model_load_before_reporting_timeout(self):
        from composer_server import MODEL_REQUEST_TIMEOUT_SECONDS

        self.assertGreaterEqual(MODEL_REQUEST_TIMEOUT_SECONDS, 180)

    def setUp(self):
        async def fake_advisor(notes, emotion, theory):
            return f"Consejo IA para {', '.join(notes)}: {emotion}"

        self.client = TestClient(create_app(advisor=fake_advisor), base_url="http://127.0.0.1", client=("127.0.0.1", 50000))

    def test_root_serves_composer_interface(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Cerebro - Asistente Musical", response.text)
        self.assertIn('id="keyboard"', response.text)

    def test_composer_alias_serves_same_interface(self):
        response = self.client.get("/composer")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="ask-btn"', response.text)

    def test_composer_applies_security_headers_to_its_browser_interface(self):
        response = self.client.get("/")

        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        self.assertIn("default-src 'self'", response.headers["content-security-policy"])
        self.assertIn("connect-src 'self'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["permissions-policy"], "camera=(), geolocation=(), microphone=(self)")

    def test_composer_health_uses_the_configured_local_model_endpoint(self):
        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

        class FakeSession:
            def __init__(self):
                self.requested_urls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def get(self, url, *, allow_redirects):
                assert not allow_redirects
                self.requested_urls.append(url)
                return FakeResponse()

        session = FakeSession()
        with (
            mock.patch("composer_server.aiohttp.ClientSession", return_value=session),
            mock.patch(
                "composer_server.models_endpoint",
                return_value="http://127.0.0.1:11434/v1/models",
            ),
        ):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ai"], "online")
        self.assertEqual(session.requested_urls, ["http://127.0.0.1:11434/v1/models"])

    def test_local_ai_advisor_includes_the_configured_ollama_model(self):
        from composer_server import local_ai_advisor

        class FakeResponse:
            status = 200
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def raise_for_status(self):
                return None

            async def json(self):
                return {"choices": [{"message": {"content": "Consejo"}}]}

        class FakeSession:
            def __init__(self):
                self.payloads = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def post(self, _url, *, json, allow_redirects):
                assert not allow_redirects
                self.payloads.append(json)
                return FakeResponse()

        session = FakeSession()
        with (
            mock.patch("composer_server.aiohttp.ClientSession", return_value=session),
            mock.patch("composer_server.AI_MODEL", "llama3.2", create=True),
        ):
            result = asyncio.run(
                local_ai_advisor(["C", "E", "G"], "calma", analyze_notes(["C", "E", "G"]))
            )

        self.assertEqual(result, "Consejo")
        self.assertEqual(session.payloads[0]["model"], "llama3.2")

    def test_composer_serves_a_local_favicon(self):
        favicon = self.client.get("/favicon.svg")
        self.assertEqual(favicon.status_code, 200)
        self.assertEqual(favicon.headers["content-type"], "image/svg+xml")
        self.assertIn("Cerebro Composer", favicon.text)
        self.assertIn('rel="icon"', self.client.get("/").text)

        legacy_favicon = self.client.get("/favicon.ico", follow_redirects=False)
        self.assertEqual(legacy_favicon.status_code, 307)
        self.assertEqual(legacy_favicon.headers["location"], "/favicon.svg")



    def test_mobile_header_reserves_space_for_the_astro_credit(self):
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        mobile_css = root.text.split("@media (max-width: 650px) {", 1)[1]
        self.assertIn("body { padding: 60px 4px 30px; }", mobile_css)

    def test_recognizes_c_major_triad(self):
        theory = analyze_notes(["C", "E", "G"])
        self.assertEqual(theory["chord"], "C mayor")
        self.assertIn("C mayor", theory["scales"])
        self.assertIn("A menor", theory["scales"])

    def test_recognizes_dominant_flat_five_with_sharp_spelling(self):
        theory = analyze_notes(["C", "D", "F#", "G#"])
        self.assertEqual(theory["chord"], "D séptima dominante ♭5")
        self.assertEqual(theory["root"], "D")
        self.assertEqual(theory["quality"], "séptima dominante ♭5")
        self.assertEqual(theory["next_roots"], ["G", "A", "B"])
        self.assertIn("D lidia dominante", theory["scales"])

    def test_recognizes_complete_suspended_add9_voicing(self):
        theory = analyze_notes(["D", "E", "A", "B"])
        self.assertEqual(theory["chord"], "A sus4(add9)")
        self.assertEqual(theory["root"], "A")
        self.assertEqual(theory["quality"], "sus4(add9)")
        self.assertEqual(theory["missing_notes"], [])

    def test_completes_a_unique_partial_chord_in_the_selected_scale(self):
        theory = analyze_notes(
            ["C", "E"],
            tonic="C",
            scale_intervals=(0, 2, 4, 5, 7, 9, 11),
        )
        self.assertEqual(theory["chord"], "C mayor (incompleto)")
        self.assertEqual(theory["root"], "C")
        self.assertEqual(theory["quality"], "mayor")
        self.assertEqual(theory["missing_notes"], ["G"])
        self.assertEqual(theory["next_roots"], ["F", "G", "A"])

    def test_unknown_note_collection_does_not_invent_a_key_or_next_root(self):
        theory = analyze_notes(["D", "F", "G", "A", "A#", "B"])
        self.assertEqual(theory["quality"], "no determinada")
        self.assertEqual(theory["scales"], [])
        self.assertEqual(theory["next_roots"], [])

    def test_unresolved_collection_exposes_explicit_transition_routes(self):
        theory = analyze_notes(
            ["C", "C#", "D", "E"],
            tonic="C",
            scale_intervals=(0, 2, 4, 5, 7, 9, 11),
        )
        self.assertEqual(theory["quality"], "no determinada")
        self.assertEqual(
            [option["id"] for option in theory["transition_options"]],
            ["cadence", "diminished-passing", "augmented-dominant", "secondary-dominant"],
        )

    def test_compose_returns_theory_and_ai_advice(self):
        response = self.client.post(
            "/api/compose",
            json={"notes": ["C", "E", "G"], "emotion": "épico"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["theory"]["chord"], "C mayor")
        self.assertIn("Consejo IA", data["response"])

    def test_compose_passes_selected_scale_context_to_the_advisor(self):
        response = self.client.post(
            "/api/compose",
            json={
                "notes": ["D", "F", "A"],
                "emotion": "tensión",
                "tonic": "D",
                "scale": "phrygian-dominant",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["theory"]["performance_context"]["root"], "D")
        self.assertEqual(data["theory"]["performance_context"]["scale"], "phrygian-dominant")
        self.assertIn("Contexto de interpretación: D Frigia dominante", data["response"])

    def test_compose_returns_missing_notes_for_a_unique_partial_chord(self):
        response = self.client.post(
            "/api/compose",
            json={
                "notes": ["C", "E"],
                "emotion": "resolución",
                "tonic": "C",
                "scale": "major",
            },
        )

        self.assertEqual(response.status_code, 200)
        theory = response.json()["theory"]
        self.assertEqual(theory["chord"], "C mayor (incompleto)")
        self.assertEqual(theory["missing_notes"], ["G"])
        self.assertEqual(theory["completion_options"], [{
            "root": "C",
            "quality": "mayor",
            "chord": "C mayor",
            "missing_notes": ["G"],
        }])

    def test_chord_analysis_exposes_ambiguous_completion_options_without_guessing(self):
        response = self.client.post(
            "/api/chord-analysis",
            json={"notes": ["C", "G"], "tonic": "C", "scale": "major"},
        )
        self.assertEqual(response.status_code, 200)
        theory = response.json()["theory"]
        self.assertEqual(theory["quality"], "no determinada")
        self.assertGreater(len(theory["completion_options"]), 1)
        self.assertTrue(all(option["missing_notes"] for option in theory["completion_options"]))
        self.assertTrue(theory["transition_options"])

    def test_rejects_invalid_or_empty_notes(self):
        empty = self.client.post("/api/compose", json={"notes": [], "emotion": ""})
        invalid = self.client.post(
            "/api/compose", json={"notes": ["H"], "emotion": "prueba"}
        )
        self.assertEqual(empty.status_code, 422)
        self.assertEqual(invalid.status_code, 422)

    def test_exports_a_standard_midi_file_for_a_confirmed_chord_chain(self):
        response = self.client.post(
            "/api/export/midi",
            json={
                "tempo": 120,
                "events": [
                    {
                        "notes": [
                            {"note": "C", "octave": 4, "velocity": 96},
                            {"note": "E", "octave": 4, "velocity": 90},
                            {"note": "G", "octave": 4, "velocity": 88},
                        ],
                        "duration_beats": 4,
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/midi")
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertEqual(response.content[:4], b"MThd")
        self.assertEqual(int.from_bytes(response.content[4:8], "big"), 6)
        self.assertIn(b"MTrk", response.content)
        self.assertIn(bytes((0x90, 60, 96)), response.content)
        self.assertIn(bytes((0x80, 60, 0)), response.content)

    def test_rejects_an_invalid_midi_note(self):
        response = self.client.post(
            "/api/export/midi",
            json={
                "events": [
                    {
                        "notes": [{"note": "H", "octave": 4}],
                        "duration_beats": 4,
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_imports_a_standard_midi_file_with_doubled_octaves_and_velocity(self):
        exported = self.client.post(
            "/api/export/midi",
            json={
                "tempo": 118,
                "beats_per_bar": 4,
                "events": [
                    {
                        "notes": [
                            {"note": "C", "octave": 3, "velocity": 64},
                            {"note": "C", "octave": 4, "velocity": 101},
                            {"note": "G", "octave": 4, "velocity": 87},
                        ],
                        "duration_beats": 2,
                    },
                    {
                        "notes": [{"note": "F", "octave": 3, "velocity": 76}],
                        "offset_beats": 1,
                        "duration_beats": 3,
                    },
                ],
            },
        )
        self.assertEqual(exported.status_code, 200)

        imported = self.client.post(
            "/api/import/midi",
            content=exported.content,
            headers={"Content-Type": "audio/midi"},
        )
        self.assertEqual(imported.status_code, 200)
        data = imported.json()
        self.assertEqual(data["tempo"], 118)
        self.assertEqual(data["beats_per_bar"], 4)
        self.assertEqual(len(data["events"]), 2)
        self.assertEqual(data["events"][0]["duration_beats"], 2)
        self.assertEqual(
            data["events"][0]["notes"],
            [
                {"note": "C", "octave": 3, "velocity": 64},
                {"note": "C", "octave": 4, "velocity": 101},
                {"note": "G", "octave": 4, "velocity": 87},
            ],
        )
        self.assertEqual(data["events"][1]["offset_beats"], 1)

    def test_rejects_an_invalid_midi_import(self):
        response = self.client.post(
            "/api/import/midi",
            content=b"esto no es MIDI",
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(response.status_code, 422)

    def test_builds_a_complete_diatonic_progression_from_the_selected_chord(self):
        response = self.client.post(
            "/api/progression",
            json={
                "notes": [
                    {"note": "C", "octave": 3, "velocity": 76},
                    {"note": "E", "octave": 3, "velocity": 84},
                    {"note": "G", "octave": 3, "velocity": 92},
                ],
                "tonic": "C",
                "scale": "major",
                "style": "diatonic",
                "length": 4,
                "duration_beats": 4,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["events"]), 4)
        self.assertEqual(data["events"][0]["label"], "C mayor · origen")
        self.assertEqual(data["events"][0]["notes"][0], {"note": "C", "octave": 3, "velocity": 76})
        self.assertIn("A menor", data["events"][1]["label"])
        self.assertIn("D menor", data["events"][2]["label"])
        self.assertIn("G séptima dominante", data["events"][3]["label"])

    def test_builds_an_explicit_resolution_from_an_unresolved_colour_cluster(self):
        response = self.client.post(
            "/api/progression",
            json={
                "notes": [
                    {"note": "C", "octave": 3},
                    {"note": "C#", "octave": 3},
                    {"note": "D", "octave": 3},
                    {"note": "E", "octave": 3},
                ],
                "tonic": "C",
                "scale": "major",
                "style": "diminished-passing",
                "length": 4,
                "allow_unresolved_source": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("color de origen", data["events"][0]["label"])
        self.assertIn("disminuido séptima", data["events"][1]["label"])
        self.assertEqual(data["events"][-1]["label"], "C mayor")

    def test_builds_a_diminished_link_without_guessing_an_incomplete_source_chord(self):
        response = self.client.post(
            "/api/progression",
            json={
                "notes": [
                    {"note": "C", "octave": 3},
                    {"note": "E", "octave": 3},
                    {"note": "G", "octave": 3},
                ],
                "tonic": "C",
                "scale": "major",
                "style": "diminished-passing",
                "length": 4,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("disminuido séptima", data["events"][1]["label"])
        self.assertEqual(
            [note["note"] for note in data["events"][1]["notes"]],
            ["F#", "A", "C", "D#"],
        )
        self.assertEqual(data["events"][-1]["label"], "C mayor")

        augmented = self.client.post(
            "/api/progression",
            json={
                "notes": [
                    {"note": "C", "octave": 3},
                    {"note": "E", "octave": 3},
                    {"note": "G", "octave": 3},
                ],
                "tonic": "C",
                "scale": "major",
                "style": "augmented-dominant",
                "length": 4,
            },
        )
        self.assertEqual(augmented.status_code, 200)
        self.assertEqual(augmented.json()["events"][-1]["label"], "C mayor")
        self.assertIn("séptima dominante aumentada", augmented.json()["events"][-2]["label"])

        incomplete = self.client.post(
            "/api/progression",
            json={
                "notes": [{"note": "C", "octave": 3}, {"note": "E", "octave": 3}],
                "tonic": "C",
                "scale": "major",
                "style": "diatonic",
                "length": 4,
            },
        )
        self.assertEqual(incomplete.status_code, 422)

    def test_ui_has_health_status_audio_and_safe_response_rendering(self):
        html = self.client.get("/").text
        self.assertIn("/api/health", html)
        self.assertIn('id="play-btn"', html)
        self.assertIn("function playVoices", html)
        self.assertIn('fetch("/api/chord-analysis"', html)
        self.assertIn("responseBox.textContent = data.response", html)
        self.assertNotIn("responseBox.innerHTML = data.response", html)

    def test_ui_offers_a_local_microphone_chord_capture_without_uploading_audio(self):
        html = self.client.get("/").text
        self.assertIn('id="listen-chord-btn"', html)
        self.assertIn('id="microphone-status"', html)
        self.assertIn("captureChordFromMicrophone", html)

        detector = self.client.get("/assets/composer_audio.js")
        self.assertEqual(detector.status_code, 200)
        self.assertIn("navigator.mediaDevices.getUserMedia", detector.text)
        self.assertIn("window.isSecureContext", detector.text)
        self.assertNotIn("fetch(", detector.text)

    def test_ui_defers_the_microphone_module_so_file_mode_can_bootstrap(self):
        html = self.client.get("/").text
        self.assertIn('<script>\n    const notes', html)
        self.assertNotIn('<script type="module">', html)
        self.assertIn('import("/assets/composer_audio.js")', html)
        self.assertIn('location.protocol === "file:"', html)

    def test_lan_client_needs_a_key_then_receives_a_session(self):
        token = "test-lan-key-which-is-long-enough"
        app = create_app(lan_token=token)
        remote = TestClient(
            app,
            follow_redirects=False,
            client=("192.168.8.45", 53000),
        )

        denied = remote.get("/")
        self.assertEqual(denied.status_code, 403)

        connect_page = remote.get("/connect")
        self.assertEqual(connect_page.status_code, 200)

        authenticated = remote.post(
            "/api/lan-session",
            headers={"X-Composer-Access-Key": token},
        )
        self.assertEqual(authenticated.status_code, 204)
        self.assertIn("composer_lan_session", authenticated.headers["set-cookie"])

        granted = remote.get("/")
        self.assertEqual(granted.status_code, 200)

    def test_lan_token_file_enables_only_a_protected_lan_bind(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "composer-token"
            token_path.write_text("x" * 32 + "\n", encoding="utf-8")
            token = load_lan_token(str(token_path))

        self.assertEqual(token, "x" * 32)
        self.assertEqual(composer_bind_host(token), "0.0.0.0")
        self.assertEqual(composer_bind_host(None), "127.0.0.1")

    def test_push_grid_in_key_folds_c_major_into_an_eight_by_eight_pad_layout(self):
        response = self.client.get(
            "/api/push-grid",
            params={"root": "C", "scale": "major", "mode": "in-key", "octave": 3},
        )

        self.assertEqual(response.status_code, 200)
        grid = response.json()["grid"]
        self.assertEqual(len(grid), 8)
        self.assertTrue(all(len(row) == 8 for row in grid))
        self.assertEqual(
            [pad["note"] for pad in grid[-1]],
            ["C", "D", "E", "F", "G", "A", "B", "C"],
        )
        self.assertTrue(grid[-1][0]["is_root"])
        self.assertTrue(all(pad["in_scale"] for row in grid for pad in row))

    def test_push_grid_chromatic_marks_notes_outside_the_selected_scale(self):
        response = self.client.get(
            "/api/push-grid",
            params={"root": "C", "scale": "major", "mode": "chromatic", "octave": 3},
        )

        self.assertEqual(response.status_code, 200)
        bottom_row = response.json()["grid"][-1]
        self.assertEqual([pad["note"] for pad in bottom_row[:3]], ["C", "C#", "D"])
        self.assertTrue(bottom_row[0]["is_root"])
        self.assertFalse(bottom_row[1]["in_scale"])
        self.assertTrue(bottom_row[2]["in_scale"])

    def test_scale_catalog_includes_extended_composition_families(self):
        response = self.client.get("/api/scales")
        self.assertEqual(response.status_code, 200)
        scales = response.json()["scales"]
        identifiers = {scale["id"] for scale in scales}
        self.assertGreaterEqual(len(scales), 40)
        self.assertTrue(
            {
                "phrygian-dominant",
                "lydian-dominant",
                "altered",
                "diminished-half-whole",
                "double-harmonic",
                "chromatic",
            }.issubset(identifiers)
        )

    def test_push_grid_accepts_a_custom_pitch_class_set(self):
        response = self.client.get(
            "/api/push-grid",
            params={
                "root": "C",
                "scale": "custom",
                "intervals": "0,1,4",
                "mode": "in-key",
                "octave": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        grid = response.json()["grid"]
        self.assertEqual([pad["note"] for pad in grid[-1]], ["C", "C#", "E", "C", "C#", "E", "C", "C#"])
        self.assertTrue(all(pad["in_scale"] for row in grid for pad in row))

    def test_ui_exposes_push_surface_and_scale_controls(self):
        html = self.client.get("/").text
        self.assertIn('id="push-grid"', html)
        self.assertIn('id="root-select"', html)
        self.assertIn('id="scale-select"', html)
        self.assertIn('id="push-mode"', html)

    def test_ui_exposes_shared_composition_controls_for_the_piano(self):
        html = self.client.get("/").text
        self.assertIn('id="piano-root-select"', html)
        self.assertIn('id="piano-scale-select"', html)
        self.assertIn('id="piano-mode-select"', html)
        self.assertIn('id="piano-octave-down"', html)
        self.assertIn('id="piano-octave-up"', html)
        self.assertIn("function renderKeyboard()", html)
        self.assertIn('control.classList.contains("push-pad") || pushMode.value === "in-key"', html)

    def test_ui_distinguishes_source_chord_from_system_transition(self):
        html = self.client.get("/").text
        self.assertIn('id="source-chord-display"', html)
        self.assertIn('id="completion-notes-display"', html)
        self.assertIn('id="transition-roots-display"', html)
        self.assertIn('id="completion-action"', html)
        self.assertIn('id="apply-completion-btn"', html)
        self.assertIn('id="transition-action"', html)
        self.assertIn('id="transition-options"', html)
        self.assertIn("function renderTransitionActions", html)
        self.assertIn("--completion", html)
        self.assertIn("--transition", html)
        self.assertIn('responseBox.className = "analysis-response"', html)

    def test_ui_loads_the_scale_catalog_and_supports_custom_intervals(self):
        html = self.client.get("/").text
        self.assertIn('id="custom-intervals"', html)
        self.assertIn('fetch("/api/scales"', html)
        self.assertIn("syncPianoControls(true)", html)
        self.assertIn("currentScaleNotes", html)
        self.assertIn(".custom-intervals[hidden] { display: none; }", html)
        self.assertIn(".push-grid { min-width: 0; width: 100%; }", html)

    def test_ui_exposes_a_confirmed_chord_chain_and_midi_export(self):
        html = self.client.get("/").text
        self.assertIn('id="add-chord-btn"', html)
        self.assertIn('id="sequence-list"', html)
        self.assertIn('id="play-sequence-btn"', html)
        self.assertIn('id="export-midi-btn"', html)
        self.assertIn("selectedVoices", html)
        self.assertIn("function playChordSequence", html)
        self.assertIn("secondsPerBeat", html)
        self.assertIn('fetch("/api/export/midi"', html)

    def test_ui_supports_voice_velocity_persistence_and_midi_import(self):
        html = self.client.get("/").text
        self.assertIn('id="new-note-velocity"', html)
        self.assertIn('id="midi-import-input"', html)
        self.assertIn('id="import-midi-btn"', html)
        self.assertIn("function voiceKey", html)
        self.assertIn("localStorage", html)
        self.assertIn('fetch("/api/import/midi"', html)

    def test_ui_exposes_complete_midi_progression_controls(self):
        html = self.client.get("/").text
        self.assertIn('id="progression-style"', html)
        self.assertIn('id="progression-length"', html)
        self.assertIn('id="build-progression-btn"', html)
        self.assertIn('value="adaptive-link"', html)
        self.assertIn('value="secondary-dominant"', html)
        self.assertIn('value="augmented-dominant"', html)
        self.assertIn('fetch("/api/progression"', html)

    def test_ui_explains_that_the_file_copy_cannot_load_the_full_local_app(self):
        html = self.client.get("/").text
        self.assertIn('location.protocol === "file:"', html)
        self.assertIn("http://127.0.0.1:8082/", html)

    def test_ui_discards_an_analysis_response_for_a_changed_note_selection(self):
        html = self.client.get("/").text
        self.assertIn("let analysisRequest = 0", html)
        self.assertIn("const requestId = ++analysisRequest", html)
        self.assertIn("if (requestId !== analysisRequest) return", html)

    def test_ui_exposes_an_honest_local_model_wait_indicator(self):
        html = self.client.get("/").text

        self.assertIn('id="analysis-progress"', html)
        self.assertIn('role="progressbar"', html)
        self.assertIn("Solicitud enviada a Cerebro local", html)
        self.assertIn("startAnalysisProgress", html)
        self.assertIn("stopAnalysisProgress", html)

    def test_composer_can_be_forced_to_loopback_for_a_private_reverse_proxy(self):
        self.assertEqual(
            composer_bind_host("a" * 24, requested_host="127.0.0.1"),
            "127.0.0.1",
        )

    def test_composer_refuses_an_arbitrary_forced_bind_address(self):
        with self.assertRaises(ValueError):
            composer_bind_host("a" * 24, requested_host="0.0.0.0")


if __name__ == "__main__":
    unittest.main()
