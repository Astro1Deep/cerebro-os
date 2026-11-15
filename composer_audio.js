const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function midiFrequency(midi) {
    return 440 * Math.pow(2, (midi - 69) / 12);
}

function clampInteger(value, minimum, maximum, fallback) {
    return Number.isInteger(value) && value >= minimum && value <= maximum ? value : fallback;
}

function magnitudeFromDb(value) {
    return Number.isFinite(value) ? Math.pow(10, Math.max(-120, value) / 20) : 0;
}

function peakMagnitudeNearBin(spectrum, bin) {
    const center = Math.round(bin);
    let strongest = 0;
    for (let offset = -1; offset <= 1; offset += 1) {
        const index = center + offset;
        if (index >= 0 && index < spectrum.length) {
            strongest = Math.max(strongest, magnitudeFromDb(spectrum[index]));
        }
    }
    return strongest;
}

function closeToHarmonic(interval) {
    return [12, 19, 24, 28, 31].some(harmonic => Math.abs(interval - harmonic) <= 1);
}

export function midiToVoice(midi) {
    if (!Number.isInteger(midi) || midi < 0 || midi > 127) return null;
    return {
        note: NOTE_NAMES[midi % 12],
        octave: Math.floor(midi / 12) - 1,
    };
}

/**
 * Returns likely fundamentals from one local analyser spectrum.
 * It is deliberately conservative: a silent/noisy frame returns no note.
 */
export function findFundamentalMidiPitches(spectrum, sampleRate, fftSize, options = {}) {
    if (!(spectrum instanceof Float32Array) || !Number.isFinite(sampleRate) || sampleRate <= 0
        || !Number.isInteger(fftSize) || fftSize <= 0) {
        return [];
    }

    const minimumDb = Number.isFinite(options.minimumDb) ? options.minimumDb : -58;
    const minimumMidi = clampInteger(options.minimumMidi, 21, 108, 36);
    const maximumMidi = clampInteger(options.maximumMidi, minimumMidi, 108, 96);
    const maximumNotes = clampInteger(options.maximumNotes, 1, 8, 6);
    const candidates = [];

    for (let midi = minimumMidi; midi <= maximumMidi; midi += 1) {
        const fundamentalBin = (midiFrequency(midi) * fftSize) / sampleRate;
        const fundamentalDb = spectrum[Math.round(fundamentalBin)];
        if (!Number.isFinite(fundamentalDb) || fundamentalDb < minimumDb) continue;

        let score = peakMagnitudeNearBin(spectrum, fundamentalBin);
        for (const [harmonic, weight] of [[2, 0.35], [3, 0.18], [4, 0.08]]) {
            const harmonicBin = fundamentalBin * harmonic;
            if (harmonicBin < spectrum.length - 1) {
                score += peakMagnitudeNearBin(spectrum, harmonicBin) * weight;
            }
        }
        candidates.push({midi, score});
    }

    if (!candidates.length) return [];
    candidates.sort((first, second) => second.score - first.score || first.midi - second.midi);
    const strongest = candidates[0].score;
    const selected = [];
    for (const candidate of candidates) {
        if (candidate.score < strongest * 0.28) continue;
        const relatedFundamental = selected.find(existing => (
            candidate.midi > existing.midi
            && closeToHarmonic(candidate.midi - existing.midi)
            && candidate.score < existing.score * 0.72
        ));
        if (relatedFundamental || selected.some(existing => Math.abs(existing.midi - candidate.midi) <= 1)) continue;
        selected.push(candidate);
        if (selected.length >= maximumNotes) break;
    }
    return selected.map(candidate => candidate.midi).sort((first, second) => first - second);
}

/** Merge analyser frames. A note must recur in a clear majority of frames. */
export function mergeStableDetections(frameDetections, options = {}) {
    if (!Array.isArray(frameDetections) || !frameDetections.length) {
        return {midi: [], confidence: 0, frameCount: 0};
    }
    const minimumFrames = clampInteger(options.minimumFrames, 1, 120, 5);
    const minimumAgreement = Number.isFinite(options.minimumAgreement)
        ? Math.min(1, Math.max(0.5, options.minimumAgreement))
        : 0.68;
    const validFrames = frameDetections.map(frame => (
        Array.isArray(frame)
            ? new Set(frame.filter(midi => Number.isInteger(midi) && midi >= 21 && midi <= 108))
            : new Set()
    ));
    const counts = new Map();
    for (const frame of validFrames) {
        for (const midi of frame) counts.set(midi, (counts.get(midi) || 0) + 1);
    }
    const threshold = Math.max(minimumFrames, Math.ceil(validFrames.length * minimumAgreement));
    const stable = [...counts.entries()]
        .filter(([, count]) => count >= threshold)
        .sort((first, second) => first[0] - second[0]);
    const confidence = stable.length
        ? stable.reduce((total, [, count]) => total + count / validFrames.length, 0) / stable.length
        : 0;
    return {
        midi: stable.map(([midi]) => midi),
        confidence: Math.round(confidence * 100) / 100,
        frameCount: validFrames.length,
    };
}

/**
 * Captures a short audio window entirely inside the browser.
 * No MediaRecorder, upload, or network request is used here.
 */
export async function captureChordFromMicrophone(options = {}) {
    if (typeof window === "undefined" || !window.isSecureContext) {
        throw new Error("El micrófono requiere una página segura (HTTPS o localhost).");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Este navegador no permite acceder al micrófono.");
    }
    const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextConstructor) throw new Error("Este navegador no dispone de análisis de audio.");

    const durationMs = Number.isFinite(options.durationMs)
        ? Math.min(6000, Math.max(1000, options.durationMs))
        : 2600;
    const signal = options.signal;
    if (signal?.aborted) throw new DOMException("Captura cancelada", "AbortError");

    const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
        },
        video: false,
    });
    const context = new AudioContextConstructor();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 8192;
    analyser.smoothingTimeConstant = 0.2;
    source.connect(analyser);
    const spectrum = new Float32Array(analyser.frequencyBinCount);
    const frames = [];
    let animationFrame = null;
    let lastSampleAt = 0;
    let settled = false;

    return new Promise((resolve, reject) => {
        const cleanup = async () => {
            if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
            signal?.removeEventListener("abort", cancelCapture);
            try { source.disconnect(); } catch (_) {}
            try { analyser.disconnect(); } catch (_) {}
            stream.getTracks().forEach(track => track.stop());
            try { await context.close(); } catch (_) {}
        };
        const finish = async () => {
            if (settled) return;
            settled = true;
            await cleanup();
            resolve(mergeStableDetections(frames));
        };
        const fail = async error => {
            if (settled) return;
            settled = true;
            await cleanup();
            reject(error);
        };
        const cancelCapture = () => {
            void fail(new DOMException("Captura cancelada", "AbortError"));
        };
        signal?.addEventListener("abort", cancelCapture, {once: true});
        const startedAt = performance.now();
        const sample = now => {
            if (settled) return;
            try {
                if (now - lastSampleAt >= 80) {
                    analyser.getFloatFrequencyData(spectrum);
                    frames.push(findFundamentalMidiPitches(spectrum, context.sampleRate, analyser.fftSize));
                    lastSampleAt = now;
                    options.onProgress?.({frameCount: frames.length, elapsedMs: Math.round(now - startedAt)});
                }
                if (now - startedAt >= durationMs) {
                    void finish();
                    return;
                }
                animationFrame = window.requestAnimationFrame(sample);
            } catch (error) {
                void fail(error);
            }
        };
        animationFrame = window.requestAnimationFrame(sample);
    });
}
