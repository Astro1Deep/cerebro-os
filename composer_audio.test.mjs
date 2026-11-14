import assert from "node:assert/strict";
import test from "node:test";

import {
    findFundamentalMidiPitches,
    mergeStableDetections,
    midiToVoice,
} from "./composer_audio.js";

test("midiToVoice preserves Composer's sharp note spelling and octaves", () => {
    assert.deepEqual(midiToVoice(60), {note: "C", octave: 4});
    assert.deepEqual(midiToVoice(61), {note: "C#", octave: 4});
    assert.equal(midiToVoice(60.5), null);
});

test("mergeStableDetections keeps a stable C major triad and rejects a transient", () => {
    const result = mergeStableDetections([
        [60, 64, 67], [60, 64, 67], [60, 64, 67], [60, 64, 67],
        [60, 64, 67], [60, 64, 67], [60, 65, 67], [60, 64, 67],
    ], {minimumFrames: 5, minimumAgreement: 0.7});

    assert.deepEqual(result.midi, [60, 64, 67]);
    assert.ok(result.confidence >= 0.9);
    assert.equal(result.frameCount, 8);
});

test("findFundamentalMidiPitches identifies synthetic C4 E4 G4 fundamentals", () => {
    const sampleRate = 44100;
    const fftSize = 8192;
    const spectrum = new Float32Array(fftSize / 2).fill(-120);
    for (const frequency of [261.63, 329.63, 392.0]) {
        for (const harmonic of [1, 2, 3]) {
            const bin = Math.round((frequency * harmonic * fftSize) / sampleRate);
            spectrum[bin] = harmonic === 1 ? -16 : -26;
        }
    }

    assert.deepEqual(
        findFundamentalMidiPitches(spectrum, sampleRate, fftSize, {minimumDb: -70}),
        [60, 64, 67],
    );
});
