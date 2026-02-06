# Plan: Phase 7 — Audio Processing Integration

## Overview

Integrate all previously ported crates (webrtc-common-audio, webrtc-agc2, webrtc-ns, webrtc-aec3) into the unified `webrtc-apm` crate, creating a complete Rust AudioProcessing API matching the C++ implementation.

**Scope:** ~9,300 lines of C++ → ~12,000 lines of Rust, 18 steps, ~200 new tests

**What's already ported (reuse):**
- `ChannelBuffer<T>` — `crates/webrtc-common-audio/src/channel_buffer.rs`
- `PushSincResampler` — `crates/webrtc-common-audio/src/push_sinc_resampler.rs`
- `PushResampler` — `crates/webrtc-common-audio/src/push_resampler.rs`
- `audio_util` (S16↔float) — `crates/webrtc-common-audio/src/audio_util.rs`
- `CascadedBiquadFilter` — `crates/webrtc-aec3/src/cascaded_biquad_filter.rs` (pub(crate))
- All AGC2 internals — `crates/webrtc-agc2/src/` (19 modules, pub(crate))
- All AEC3 internals — `crates/webrtc-aec3/src/` (61 modules, pub(crate))
- `NoiseSuppressor` — `crates/webrtc-ns/src/noise_suppressor.rs` (pub)

**What to skip (not ported):**
- `AgcManagerDirect` / `GainControlImpl` — AGC1, deprecated
- `EchoControlMobileImpl` — AECM, removed upstream M146
- `AecDump` — debug logging
- `ApmDataDumper` — debug logging
- `PostFilter` — optional post-processing hook
- `CustomProcessing` / `CustomAudioAnalyzer` — user-provided callbacks (can add later)
- `NeuralResidualEchoEstimator` — optional neural estimator
- `EchoControlFactory` — factory pattern (use direct construction)
- `RaceChecker` — Rust's type system provides Send/Sync
- `FieldTrialsView` / `Environment` — use default values throughout
- `metrics.h` / `InputVolumeStatsReporter` — WebRTC metrics system

## Architecture Decisions

1. **CascadedBiquadFilter**: Move from webrtc-aec3 to webrtc-common-audio (generic DSP primitive used by both AEC3 and HighPassFilter). webrtc-aec3 re-exports it.

2. **SwapQueue**: Port C++ lock-free SPSC queue directly using `AtomicUsize` + `UnsafeCell`. Only 249 lines, and lock-free behavior is critical for real-time audio.

3. **Threading**: Maintain C++ two-thread model (render + capture). `AudioProcessing` is `Send + Sync` with internal `Mutex` for shared state.

4. **Visibility**: Make necessary modules `pub` in webrtc-agc2 and webrtc-aec3 so webrtc-apm can use them. Specifically:
   - webrtc-agc2: make all modules pub (currently pub(crate))
   - webrtc-aec3: make block_processor, frame_blocker, block_framer, block_delay_buffer, config_selector, multi_channel_content_detector, api_call_jitter_metrics, block, config pub

5. **AudioBuffer**: Port in webrtc-apm (not webrtc-common-audio) since it depends on SplittingFilter which depends on ThreeBandFilterBank — both APM-specific.

6. **AudioFrame**: Simple interleaved buffer type. Port as a lightweight struct in webrtc-apm.

## Steps

### Step 0: Move CascadedBiquadFilter to webrtc-common-audio

Move `cascaded_biquad_filter.rs` from webrtc-aec3 to webrtc-common-audio. Update webrtc-aec3 to depend on webrtc-common-audio and re-import. Add webrtc-common-audio dependency to webrtc-aec3's Cargo.toml.

**Files:** `crates/webrtc-common-audio/src/cascaded_biquad_filter.rs` (new), `crates/webrtc-common-audio/src/lib.rs`, `crates/webrtc-aec3/src/lib.rs`, `crates/webrtc-aec3/Cargo.toml`
**C++ source:** `webrtc/modules/audio_processing/utility/cascaded_biquad_filter.h/cc` (149 lines)
**Tests:** Existing tests move with the module (~5 tests)
**Commit:** `refactor(rust): move CascadedBiquadFilter to webrtc-common-audio`

### Step 1: ThreeBandFilterBank

Port the 3-band QMF analysis/synthesis filter bank used for splitting 48kHz audio into three 16kHz bands.

**File:** `crates/webrtc-apm/src/three_band_filter_bank.rs`
**C++ source:** `webrtc/modules/audio_processing/three_band_filter_bank.h/cc` (355 lines)
**Tests:** ~4 (basic analysis, synthesis, roundtrip, multi-frame)
**Commit:** `feat(rust): port three-band filter bank`

### Step 2: SplittingFilter

Port the multi-channel splitting filter that wraps ThreeBandFilterBank. Handles 2-band (32kHz) and 3-band (48kHz) splitting.

**File:** `crates/webrtc-apm/src/splitting_filter.rs`
**C++ source:** `webrtc/modules/audio_processing/splitting_filter.h/cc` (216 lines)
**Tests:** ~3 (from `tests/unit/splitting_filter_unittest.cc`, 1 TEST_P)
**Commit:** `feat(rust): port splitting filter`

### Step 3: AudioBuffer

Port the central multi-band, multi-channel audio buffer. This is the core data type that all APM components operate on. Handles resampling (input/output rates can differ), channel conversion (downmixing), and band splitting via SplittingFilter.

**File:** `crates/webrtc-apm/src/audio_buffer.rs`
**C++ source:** `webrtc/modules/audio_processing/audio_buffer.h/cc` (597 lines)
**Dependencies:** ChannelBuffer (webrtc-common-audio), PushSincResampler (webrtc-common-audio), SplittingFilter (step 2)
**Tests:** ~8 (from C++ tests + custom: construction, copy_from_i16, copy_to_i16, copy_from_f32, split/merge bands, downmixing, resampling, set_num_channels)
**Commit:** `feat(rust): port AudioBuffer`

### Step 4: StreamConfig + ProcessingConfig + AudioProcessing::Config

Port the configuration types from `api/audio/audio_processing.h`:
- `StreamConfig` — sample rate, num_channels, num_frames
- `ProcessingConfig` — 4 StreamConfigs (input, output, reverse_input, reverse_output)
- `AudioProcessing::Config` — all nested config structs (Pipeline, PreAmplifier, CaptureLevelAdjustment, HighPassFilter, EchoCanceller, NoiseSuppression, GainController1, GainController2)
- `RuntimeSetting` — dynamic configuration changes

**File:** `crates/webrtc-apm/src/config.rs`
**C++ source:** `webrtc/api/audio/audio_processing.h` (883 lines, config portion ~400 lines)
**Tests:** ~5 (config defaults, validation, stream_config properties)
**Commit:** `feat(rust): port AudioProcessing config types`

### Step 5: Echo detector utilities

Port the 4 small echo detector utility components used by ResidualEchoDetector:
- `circular_buffer.rs` — fixed-size float circular buffer
- `mean_variance_estimator.rs` — online mean/variance
- `moving_max.rs` — sliding window maximum
- `normalized_covariance_estimator.rs` — normalized covariance

**Files:** 4 files in `crates/webrtc-apm/src/echo_detector/`
**C++ sources:** `webrtc/modules/audio_processing/echo_detector/` (350 lines total)
**Tests:** ~14 (4+4+4+2 from C++ tests)
**Commit:** `feat(rust): port echo detector utilities`

### Step 6: HighPassFilter + ResidualEchoDetector + AudioSamplesScaler

Port three simpler component wrappers:
- `high_pass_filter.rs` — cascaded biquad HPF per channel (uses CascadedBiquadFilter from webrtc-common-audio)
- `residual_echo_detector.rs` — echo detection using covariance analysis (uses echo detector utilities)
- `audio_samples_scaler.rs` — scales audio samples with linear ramping

**Files:** `crates/webrtc-apm/src/high_pass_filter.rs`, `crates/webrtc-apm/src/residual_echo_detector.rs`, `crates/webrtc-apm/src/audio_samples_scaler.rs`
**C++ sources:** `high_pass_filter.h/cc` (182 lines), `residual_echo_detector.h/cc` (302 lines), `audio_samples_scaler.h/cc` (139 lines)
**Tests:** ~16 (7 HPF + 4 RED + 5 scaler)
**Commit:** `feat(rust): port high-pass filter, residual echo detector, and audio samples scaler`

### Step 7: CaptureLevelsAdjuster + RmsLevel

Port capture level adjustment:
- `capture_levels_adjuster.rs` — pre/post gain adjustment + optional analog mic emulation
- `rms_level.rs` — RMS level metering for statistics

**Files:** `crates/webrtc-apm/src/capture_levels_adjuster.rs`, `crates/webrtc-apm/src/rms_level.rs`
**C++ sources:** `capture_levels_adjuster.h/cc` (184 lines), `rms_level.h/cc` (222 lines)
**Tests:** ~6 (3 CLA + 3 RMS)
**Commit:** `feat(rust): port capture levels adjuster and RMS level`

### Step 8: Visibility changes in webrtc-agc2 + webrtc-aec3

Make necessary modules public in both crates so webrtc-apm can use them:
- webrtc-agc2: change pub(crate) → pub for modules consumed by GainController2
- webrtc-aec3: change pub(crate) → pub for modules consumed by EchoCanceller3

**Files:** `crates/webrtc-agc2/src/lib.rs`, `crates/webrtc-aec3/src/lib.rs`
**Tests:** 0 new (existing tests still pass)
**Commit:** `refactor(rust): make AGC2 and AEC3 modules public for integration`

### Step 9: InputVolumeController (deferred from Phase 4)

Port the mic volume recommendation controller. Complex state machine with clipping detection, volume ramping, and speech-based adjustment.

**File:** `crates/webrtc-agc2/src/input_volume_controller.rs`
**C++ source:** `webrtc/modules/audio_processing/agc2/input_volume_controller.h/cc` (884 lines)
**Tests:** ~20 (subset of 55 C++ tests — many depend on field trials and AudioBuffer which we simplify)
**Commit:** `feat(rust): port AGC2 input volume controller`

### Step 10: GainController2 (deferred from Phase 4)

Port the top-level AGC2 orchestrator that combines:
- Fixed digital gain
- Adaptive digital gain (VAD + level estimation + saturation protection)
- Input volume controller
- Limiter

**File:** `crates/webrtc-apm/src/gain_controller2.rs` (or `crates/webrtc-agc2/src/gain_controller2.rs`)
**C++ source:** `webrtc/modules/audio_processing/gain_controller2.h/cc` (394 lines)
**Dependencies:** All AGC2 modules (webrtc-agc2), AudioBuffer (step 3)
**Tests:** ~10 (subset of 17 C++ tests)
**Commit:** `feat(rust): port GainController2 wrapper`

### Step 11: ConfigSelector (deferred from Phase 6)

Port AEC3 config selection — selects between mono and multichannel configs.

**File:** `crates/webrtc-aec3/src/config_selector.rs` (already exists as stub? check)
**C++ source:** `webrtc/modules/audio_processing/aec3/config_selector.h/cc` (115 lines)
**Tests:** ~3 (from `tests/unit/aec3/config_selector_unittest.cc`, 5 TEST_P but some use FieldTrials)
**Commit:** `feat(rust): port AEC3 config selector`

### Step 12: SwapQueue

Port the lock-free SPSC queue used for render→capture thread communication. Uses `AtomicUsize` for `num_elements_` with Acquire/Release ordering, `UnsafeCell` for queue storage.

**File:** `crates/webrtc-apm/src/swap_queue.rs`
**C++ source:** `webrtc/rtc_base/swap_queue.h` (249 lines)
**Tests:** ~8 (single-thread insert/remove, full queue, empty queue, clear, concurrent producer/consumer)
**Commit:** `feat(rust): port lock-free SwapQueue`

### Step 13: EchoCanceller3 (deferred from Phase 6)

Port the top-level AEC3 wrapper. This is the interface between AudioProcessingImpl and the AEC3 block-level processing. Handles:
- Frame↔block conversion (160 samples ↔ 64 samples) via FrameBlocker/BlockFramer
- Render queue (SwapQueue) for concurrent render/capture paths
- Multichannel content detection and config switching
- RenderWriter inner class for thread-safe render analysis

**File:** `crates/webrtc-aec3/src/echo_canceller3.rs` (or `crates/webrtc-apm/src/echo_canceller3.rs`)
**C++ source:** `webrtc/modules/audio_processing/aec3/echo_canceller3.h/cc` (1,238 lines)
**Dependencies:** BlockProcessor, FrameBlocker, BlockFramer, BlockDelayBuffer, ConfigSelector, MultiChannelContentDetector, ApiCallJitterMetrics, SwapQueue, AudioBuffer
**Tests:** ~10 (subset of 23 C++ tests — many use FRIEND_TEST and mocks)
**Commit:** `feat(rust): port EchoCanceller3 top-level wrapper`

### Step 14: AudioConverter

Port audio format conversion (channel mixing + resampling). Used by AudioProcessingImpl for render path format conversion.

**File:** `crates/webrtc-apm/src/audio_converter.rs`
**C++ source:** `webrtc/common_audio/audio_converter.h/cc` (290 lines)
**Dependencies:** PushSincResampler, audio_util
**Tests:** ~4 (mono↔stereo, resampling, passthrough, combined)
**Commit:** `feat(rust): port AudioConverter`

### Step 15: SubmoduleStates + processing pipeline helpers

Port the submodule state tracking and processing pipeline helper types:
- `SubmoduleStates` — tracks which submodules are active
- `RenderQueueItemVerifier` — verifier for render signal queues
- AudioProcessingStats reporting

**File:** `crates/webrtc-apm/src/submodule_states.rs`, `crates/webrtc-apm/src/stats.rs`
**C++ source:** portions of `audio_processing_impl.h/cc` (~200 lines)
**Tests:** ~3
**Commit:** `feat(rust): port APM submodule states and stats`

### Step 16: AudioProcessingImpl — initialization + config

Port the core AudioProcessingImpl struct with construction, initialization, and configuration:
- `new()`, `initialize()`, `apply_config()`, `get_config()`
- All `Initialize*` methods for submodules
- `MaybeInitializeCapture`, `MaybeInitializeRender`
- Format state management

**File:** `crates/webrtc-apm/src/audio_processing_impl.rs`
**C++ source:** `webrtc/modules/audio_processing/audio_processing_impl.h/cc` (init portion ~800 lines)
**Tests:** ~5 (construction, config changes, reinitialization)
**Commit:** `feat(rust): port AudioProcessingImpl initialization`

### Step 17: AudioProcessingImpl — ProcessStream + ProcessReverseStream

Port the capture and render processing paths:
- `process_stream()` (i16 and f32 variants) — the main capture processing pipeline:
  Input → AudioBuffer → PreGain → HPF → AEC3 → NS → AGC2 → PostGain → Output
- `process_reverse_stream()` — render path: feeds AEC3 and echo detector
- Runtime settings handling
- Stream delay, analog level, key pressed setters

**File:** `crates/webrtc-apm/src/audio_processing_impl.rs` (continued)
**C++ source:** `webrtc/modules/audio_processing/audio_processing_impl.cc` (processing portion ~1,500 lines)
**Tests:** ~15 (basic processing, different sample rates, multi-channel, config changes mid-stream, runtime settings)
**Commit:** `feat(rust): port AudioProcessingImpl processing pipeline`

### Step 18: Public API + Builder

Create the public API surface:
- `AudioProcessing` trait (or struct) with the complete public interface
- `AudioProcessingBuilder` — builder pattern for construction
- Re-export all public types from `crates/webrtc-apm/src/lib.rs`

**Files:** `crates/webrtc-apm/src/lib.rs`, `crates/webrtc-apm/src/builder.rs`
**C++ source:** `webrtc/api/audio/audio_processing.h` (interface), `audio_processing_builder_impl.cc` (34 lines)
**Tests:** ~5 (builder, default config, full pipeline smoke test)
**Commit:** `feat(rust): add AudioProcessing public API and builder`

## Crate Dependencies (final)

```toml
# crates/webrtc-apm/Cargo.toml
[dependencies]
webrtc-common-audio = { workspace = true }
webrtc-aec3 = { workspace = true }
webrtc-agc2 = { workspace = true }
webrtc-ns = { workspace = true }
webrtc-simd = { workspace = true }
tracing = { workspace = true }
```

```toml
# crates/webrtc-aec3/Cargo.toml (add dependency)
webrtc-common-audio = { workspace = true }
```

## Processing Pipeline (Capture Path)

```
ProcessStream(input)
  │
  ├─ Create AudioBuffer (resample input → processing rate)
  ├─ PreAmplifier / CaptureLevelsAdjuster (pre-gain)
  ├─ SplitIntoFrequencyBands (if > 16kHz)
  ├─ HighPassFilter (band 0)
  ├─ EchoCanceller3.ProcessCapture (band 0, uses render from queue)
  ├─ NoiseSuppressor.Analyze + Process (band 0)
  ├─ GainController2.Analyze (clipping detection)
  ├─ GainController2.Process (adaptive + fixed gain + limiter)
  ├─ MergeFrequencyBands (if > 16kHz)
  ├─ CaptureLevelsAdjuster (post-gain)
  ├─ ResidualEchoDetector (fullband)
  └─ CopyTo output (resample processing rate → output)
```

## Risk Areas

1. **AudioProcessingImpl complexity (HIGH)** — 2,327 lines C++ with complex state management, two mutexes, multiple initialization paths. Strategy: port incrementally in steps 16-17, skip legacy components.

2. **SwapQueue thread safety (HIGH)** — Lock-free code requires careful `unsafe` Rust. Strategy: minimal unsafe scope, extensive concurrent tests, match C++ memory ordering exactly.

3. **EchoCanceller3 threading (MEDIUM)** — RenderWriter runs on render thread, ProcessCapture on capture thread. Strategy: use SwapQueue exactly as C++, RaceChecker replaced by Rust's Send/Sync.

4. **AudioBuffer resampling (MEDIUM)** — Multiple resamplers per channel, input/output rates. Strategy: reuse existing PushSincResampler, test with all standard rates (8k/16k/32k/48k).

5. **Legacy component interactions (LOW)** — AudioProcessingImpl references AGC1, AECM, PostFilter. Strategy: skip entirely, ifdef-equivalent in Rust (feature flags or just omit).

## Verification

```bash
cargo nextest run -p webrtc-apm         # All APM tests pass
cargo nextest run -p webrtc-aec3        # AEC3 still passes (172 tests)
cargo nextest run -p webrtc-agc2        # AGC2 still passes (165 tests)
cargo nextest run -p webrtc-common-audio # Common audio still passes
cargo nextest run --workspace           # All workspace tests pass
cargo clippy --workspace --all-targets  # Zero warnings
```
