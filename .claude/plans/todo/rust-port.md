# Plan: Port WebRTC Audio Processing to Rust

## Overview

Create a modular, fully-tested Rust port of the WebRTC Audio Processing library that exposes a C-compatible API. The port maintains bit-exact compatibility with the original C++ implementation through property-based testing.

**Goals:**
- Port the modern audio processing pipeline (AEC3, AGC2, NS)
- Modular multi-crate architecture
- Full test coverage via property tests comparing Rust output against C++ reference
- All SIMD optimizations (SSE2, AVX2, NEON) preserved
- C-compatible API for drop-in replacement
- Incremental port with build always passing

**Non-Goals:**
- Rewriting tests in Rust (tests remain in C++)
- Changing algorithms or behavior
- Adding new features
- Porting legacy/deprecated modules (AECM, AGC1) — see rationale below

**Excluded Modules (rationale):**
- **AECM (Echo Control Mobile):** Disabled by default (`mobile_mode = false`). Deleted
  upstream on M145 main (Jan 2026) — will be absent in M146. AEC3 is the recommended
  replacement. Ring buffer was its only consumer.
- **AGC1 (GainController1 / legacy AGC):** Disabled by default. Deeply entangled with
  the fixed-point SPL library (30+ C files). AGC2 is the recommended replacement.
  TODO `bugs.webrtc.org/7494` tracks eventual removal. The `kAdaptiveAnalog` mode
  (HAL mic gain control) has no AGC2 equivalent yet, but applications are expected
  to migrate to AGC2's `InputVolumeController`.
- **SPL (Signal Processing Library):** 30+ files of fixed-point `int16_t`/`int32_t`
  arithmetic used only by AECM, AGC1, and VAD filterbank. The modern pipeline
  (AEC3, AGC2, NS) uses none of it. Skipping saves ~3 weeks of porting effort.

## Current Codebase Analysis

**Source Version:** WebRTC M145 (branch-heads/7632), version 3.0

### Public API Surface (`webrtc/api/audio/audio_processing.h`)

**Core Types (M145 API):**
- `AudioProcessing` - Main interface (refcounted)
- `BuiltinAudioProcessingBuilder` - New builder pattern (M145+)
- `Environment` - New configuration context (M145+)
- `StreamConfig` - Sample rate, channels configuration
- `ProcessingConfig` - Input/output stream configurations  
- `AudioProcessing::Config` - Nested configuration struct with:
  - `Pipeline` - Processing rate, multi-channel settings
  - `PreAmplifier` - Pre-gain settings
  - `CaptureLevelAdjustment` - Pre/post gain, analog mic emulation
  - `HighPassFilter` - HPF settings
  - `EchoCanceller` - AEC3/AECM settings
  - `NoiseSuppression` - NS settings with level enum
  - `GainController1` - AGC1 settings (analog/digital modes)
  - `GainController2` - AGC2 settings (adaptive digital, fixed digital)
- `RuntimeSetting` - Runtime parameter changes
- `EchoDetector` - Echo detection interface
- `CustomProcessing` - Custom processing callback interface

**Core Methods:**
- `Initialize()` / `Initialize(ProcessingConfig)`
- `ApplyConfig(Config)`
- `ProcessStream(int16_t*/float*, StreamConfig, StreamConfig, int16_t*/float*)`
- `ProcessReverseStream(int16_t*/float*, StreamConfig, StreamConfig, int16_t*/float*)`
- `AnalyzeReverseStream(float**, StreamConfig)`
- `set_stream_analog_level(int)` / `recommended_stream_analog_level()`
- `set_stream_delay_ms(int)` / `stream_delay_ms()`
- `GetStatistics()` / `GetConfig()`
- AEC dump methods

**M145 API Changes (from M131):**
- New `BuiltinAudioProcessingBuilder` replaces `AudioProcessingBuilder`
- New `Environment` object required for creation via `CreateEnvironment()`
- New `FieldTrials` configuration system
- Namespace consolidation: `rtc::ArrayView` → `webrtc::ArrayView`, etc.
- C++20 required (was C++17)

### Module Structure

```
webrtc/
├── api/audio/                    # Public API headers
├── modules/audio_processing/
│   ├── aec3/                     # Echo Canceller 3 (65+ files)
│   │   ├── adaptive_fir_filter.cc/h
│   │   ├── echo_canceller3.cc/h
│   │   ├── matched_filter.cc/h
│   │   ├── suppression_gain.cc/h
│   │   └── ... (60+ more)
│   ├── aecm/                     # Mobile Echo Control (5 files)
│   ├── agc/                      # AGC1 (8 files)
│   ├── agc2/                     # AGC2 (25+ files)
│   │   └── rnn_vad/              # RNN-based VAD (15 files)
│   ├── ns/                       # Noise Suppression (15 files)
│   ├── vad/                      # Voice Activity Detection (10 files)
│   ├── capture_levels_adjuster/  # Level adjustment (2 files)
│   ├── echo_detector/            # Echo detection (4 files)
│   └── utility/                  # Utilities (delay estimator, etc.)
├── common_audio/                 # DSP primitives
│   ├── resampler/                # Audio resampling (6 files)
│   ├── signal_processing/        # SPL functions (30+ files)
│   ├── vad/                      # Core VAD (6 files)
│   └── third_party/ooura/        # Ooura FFT
├── rtc_base/                     # Base utilities
├── system_wrappers/              # Platform abstractions
└── third_party/
    ├── pffft/                    # PFFFT library
    └── rnnoise/                  # RNNoise weights/activations
```

### SIMD Optimizations Inventory

**SSE2 (x86/x86_64 inline):**
- `common_audio/fir_filter_sse.cc` - FIR filter convolution
- `common_audio/resampler/sinc_resampler_sse.cc` - Sinc interpolation
- `common_audio/third_party/ooura/fft_size_128/ooura_fft_sse2.cc` - FFT
- `modules/audio_processing/aec3/adaptive_fir_filter.cc` - Filter adaptation (SSE2)
- `modules/audio_processing/aec3/matched_filter.cc` - Matched filter (SSE2)
- `modules/audio_processing/aec3/vector_math.h` - Vector ops (SSE2 inline)

**AVX2 (x86/x86_64 separate compilation units):**
- `common_audio/fir_filter_avx2.cc` - FIR filter (FMA)
- `common_audio/resampler/sinc_resampler_avx2.cc` - Sinc (FMA)
- `modules/audio_processing/aec3/adaptive_fir_filter_avx2.cc` - Filter adaptation
- `modules/audio_processing/aec3/adaptive_fir_filter_erl_avx2.cc` - ERL computation
- `modules/audio_processing/aec3/fft_data_avx2.cc` - FFT data operations
- `modules/audio_processing/aec3/matched_filter_avx2.cc` - Matched filter
- `modules/audio_processing/aec3/vector_math_avx2.cc` - Vector ops
- `modules/audio_processing/agc2/rnn_vad/vector_math_avx2.cc` - RNN vector ops

**NEON (ARM/ARM64):**
- `common_audio/fir_filter_neon.cc` - FIR filter
- `common_audio/resampler/sinc_resampler_neon.cc` - Sinc interpolation
- `common_audio/third_party/ooura/fft_size_128/ooura_fft_neon.cc` - FFT
- `common_audio/signal_processing/cross_correlation_neon.c` - Cross-correlation
- `common_audio/signal_processing/downsample_fast_neon.c` - Downsampling
- `common_audio/signal_processing/min_max_operations_neon.c` - Min/max ops
- `modules/audio_processing/aecm/aecm_core_neon.cc` - AECM operations
- `modules/audio_processing/aec3/adaptive_fir_filter.cc` - Filter adaptation (NEON inline)

**ARM Assembly (ARMv7):**
- `common_audio/signal_processing/complex_bit_reverse_arm.S`
- `common_audio/signal_processing/filter_ar_fast_q12_armv7.S`
- `common_audio/third_party/spl_sqrt_floor/spl_sqrt_floor_arm.S`

**MIPS Assembly:**
- `common_audio/signal_processing/*_mips.c` - Various DSP operations (8 files)
- `modules/audio_processing/aecm/aecm_core_mips.cc` - AECM

**Inline Assembly (various platforms):**
- `common_audio/signal_processing/include/spl_inl_armv7.h` - ARM intrinsics
- `rtc_base/zero_memory.cc` - Secure memory zeroing
- `system_wrappers/source/denormal_disabler.cc` - FPU denormal handling
- `system_wrappers/source/cpu_features.cc` - CPU feature detection

### Third-Party Dependencies

1. **PFFFT** (`webrtc/third_party/pffft/`)
   - Single C file FFT library (scalar path ~3000 lines)
   - Supports SSE, Altivec, NEON in C; composite sizes (2^a * 3^b * 5^c)
   - **Rust:** Ported as pure Rust scalar in `webrtc-fft` crate (SIMD deferred to Phase 4)

2. **RNNoise** (`webrtc/third_party/rnnoise/`)
   - Neural network weights (`rnn_vad_weights.cc/h`)
   - Activation functions (`rnn_activations.h`)
   - **Rust strategy:** Port as data tables, port activation functions

3. **Ooura FFT** (`webrtc/common_audio/third_party/ooura/`)
   - 128-point and 256-point FFT implementations
   - Has SSE2 and NEON variants for 128-point
   - **Rust:** Ported as pure Rust in `webrtc-fft` crate (SSE2/NEON SIMD pending)

### Test Infrastructure (142 test files)

- **Framework:** Google Test + Google Mock
- **Test utilities:** `tests/test_utils/`
  - Audio file reading/writing
  - Bitexactness verification
  - Echo canceller test tools
  - Protobuf support for reference data
- **Mocks:** `tests/test_utils/mock/`
  - MockEchoRemover, MockRenderDelayBuffer, MockRenderDelayController, MockBlockProcessor
- **Resources:** WAV files, protobuf reference data
- **Current count:** 2432 tests passing (M145)

## Rust Workspace (Implemented)

```toml
# Cargo.toml (workspace root)
[workspace]
resolver = "3"
edition = "2024"
rust-version = "1.91"
# 10 crates in crates/ directory
```

### Crate Dependencies

```
webrtc-apm (main crate, C API)
  +-- webrtc-common-audio + tracing
  +-- webrtc-aec3 + webrtc-simd + webrtc-fft + tracing
  +-- webrtc-agc2 + webrtc-simd + webrtc-fft + tracing
  +-- webrtc-ns + webrtc-fft + tracing
  +-- webrtc-vad + tracing
  +-- webrtc-simd
  +-- webrtc-fft
  +-- webrtc-ring-buffer

Testing crates (publish = false):
  webrtc-apm-sys      -- cxx FFI to C++ (feature: cxx-bridge)
  webrtc-apm-proptest -- proptest + test-strategy generators
```

Note: `webrtc-aecm` and `webrtc-agc` (AGC1) crates removed — see Excluded Modules.

## Phased Implementation Plan

### Phase 1: Foundation Infrastructure -- COMPLETE

- [x] Workspace: 10 crates, edition 2024, resolver 3, MSRV 1.91
- [x] Lints: unexpected_cfgs=deny, unreachable_pub, absolute_paths, mod_module_files=deny, etc.
- [x] SIMD: `SimdBackend` enum (Scalar/SSE2/AVX2/NEON), 4 operations, 16 tests
- [x] FFI: `webrtc-apm-sys` with cxx shim (feature-gated `cxx-bridge`)
- [x] Testing: `proptest` + `test-strategy` derive macros, 18 tests
- [x] Tracing: direct dependency (not feature-gated)
- [x] Docs: `docs.rs` metadata with `--cfg docsrs` for feature-gated items
- [x] 11 commits, 34 Rust tests passing

### Phase 2: Common Audio Primitives -- COMPLETE (~3 weeks, 10 commits, 128 Rust tests)

**2.1 Ring Buffer** -- COMPLETE
- [x] Port `ring_buffer.c` as standalone `webrtc-ring-buffer` crate (16 tests)

**2.2 Audio Utilities** -- COMPLETE
- [x] Port `audio_util.h` conversions (int16/float, scaling)
- [x] Port `channel_buffer.cc` (multi-channel, multi-band buffer)
- Deferred: `smoothing_filter.cc` (no downstream consumer in modern pipeline)

**2.3 FIR Filter** -- DEFERRED to Phase 6 (AEC3)
- Only consumed by AEC3, will be ported alongside it

**2.4 Resampler** -- COMPLETE
- [x] Port `SincResampler` with SIMD convolution via `webrtc-simd`
- [x] Port `PushSincResampler` and `PushResampler`

**2.5 FFT** -- COMPLETE (scalar), SSE2/NEON SIMD pending
- [x] Pure Rust `webrtc-fft` crate (no `cc` crate, no C code)
- [x] Ooura 128-point FFT (scalar) + 6 unit + 3 proptests
- [x] Ooura fft4g variable-size FFT (scalar only in C++) + 3 unit + 3 proptests
- [x] PFFFT scalar (pure Rust port of pffft.c) + 10 unit + 4 proptests
- [ ] Ooura 128 SSE2 SIMD (4 inner functions)
- [ ] Ooura 128 NEON SIMD (4 inner functions)

### Phase 3: Voice Activity Detection (2 weeks)

**3.1 Core VAD (`webrtc-vad`)**
- [ ] Port `vad_core.c`
- [ ] Port `vad_filterbank.c`
- [ ] Port `vad_gmm.c`
- [ ] Port `vad_sp.c`
- [ ] Port `webrtc_vad.c` API
- [ ] Proptest: Frame-by-frame VAD decision comparison

**3.2 Standalone VAD (`webrtc-vad`)**
- [ ] Port `voice_activity_detector.cc`
- [ ] Port pitch-based VAD components
- [ ] Proptest: Multi-frame comparison

### Phase 4: Automatic Gain Control — AGC2 only (3-4 weeks)

Note: AGC1 (`webrtc-agc`) is **not ported** — see Excluded Modules rationale.

**4.1 AGC2 Core**
- [ ] Port `limiter_db_gain_curve.cc`
- [ ] Port `interpolated_gain_curve.cc`
- [ ] Port `limiter.cc`
- [ ] Port `gain_applier.cc`
- [ ] Port `biquad_filter.cc`
- [ ] Proptest: Each component

**4.2 AGC2 Adaptive Digital**
- [ ] Port `fixed_digital_level_estimator.cc`
- [ ] Port `noise_level_estimator.cc`
- [ ] Port `speech_level_estimator.cc`
- [ ] Port `saturation_protector.cc`
- [ ] Port `adaptive_digital_gain_controller.cc`
- [ ] Proptest: Level estimation comparison

**4.3 AGC2 Input Volume Controller**
- [ ] Port `clipping_predictor.cc`
- [ ] Port `clipping_predictor_level_buffer.cc`
- [ ] Port `input_volume_controller.cc`
- [ ] Port `input_volume_stats_reporter.cc`
- [ ] Proptest: Clipping detection, volume recommendations

**4.4 RNN VAD**
- [ ] Port `rnn_vad_weights.cc` (weight tables)
- [ ] Port `rnn_activations.h` (tanh approximation, etc.)
- [ ] Port `rnn_fc.cc` / `rnn_gru.cc` (neural network layers)
- [ ] Port `rnn.cc` (RNN model)
- [ ] Port feature extraction (`spectral_features.cc`, `lp_residual.cc`, etc.)
- [ ] Port pitch search (`auto_correlation.cc`, `pitch_search.cc`)
- [ ] Port `vad_wrapper.cc`
- [ ] Proptest: Speech probability comparison

### Phase 5: Noise Suppression -- COMPLETE (7 commits, 70 tests)

**5.1 NS Core (`webrtc-ns`)**
- [x] Port `ns_fft.cc` → `ns_fft.rs` (FFT wrapper using webrtc-fft Fft4g)
- [x] Port `fast_math.cc` → `fast_math.rs` (SqrtFastStartingAtLog, LogApproximation)
- [x] Port `histograms.cc` → `histograms.rs` (LRT, flatness, diff histograms)
- [x] Port `suppression_params.cc` → `suppression_params.rs` (const params per level)
- [x] Port `prior_signal_model.cc` / `signal_model.cc` → Rust modules
- [x] Port `noise_estimator.cc` / `quantile_noise_estimator.cc` → Rust modules
- [x] Port `prior_signal_model_estimator.cc` / `signal_model_estimator.cc` → Rust modules
- [x] Port `speech_probability_estimator.cc` → Rust module
- [x] Port `wiener_filter.cc` → Rust module (directed-decision SNR, gain computation)
- [x] Port `noise_suppressor.cc` → `noise_suppressor.rs` (single-channel pipeline)
- Deferred to Phase 7: multi-channel support (requires AudioBuffer), upper band processing

### Phase 6: Echo Cancellation (6-8 weeks)

**6.1 AEC3 Foundation**
- [ ] Port `aec3_common.cc` (constants, enums)
- [ ] Port `aec3_fft.cc` (FFT wrapper)
- [ ] Port `fft_data.cc` (complex FFT data)
- [ ] Port `block.h` / `block_buffer.cc`
- [ ] Port `spectrum_buffer.cc` / `fft_buffer.cc`

**6.2 AEC3 Framing**
- [ ] Port `block_delay_buffer.cc`
- [ ] Port `frame_blocker.cc`
- [ ] Port `block_framer.cc`
- [ ] Port `decimator.cc`
- [ ] Proptest: Frame blocking/deblocking

**6.3 AEC3 Delay Estimation**
- [ ] Port `downsampled_render_buffer.cc`
- [ ] Port `matched_filter.cc` (with AVX2/SSE2)
- [ ] Port `matched_filter_lag_aggregator.cc`
- [ ] Port `echo_path_delay_estimator.cc`
- [ ] Port `render_delay_controller.cc`
- [ ] Port `render_delay_buffer.cc`
- [ ] Proptest: Delay estimation accuracy

**6.4 AEC3 Adaptive Filter**
- [ ] Port `adaptive_fir_filter.cc` (with SSE2/AVX2/NEON)
- [ ] Port `adaptive_fir_filter_erl.cc`
- [ ] Port `coarse_filter_update_gain.cc`
- [ ] Port `refined_filter_update_gain.cc`
- [ ] Port `filter_analyzer.cc`
- [ ] Port `subtractor.cc`
- [ ] Proptest: Filter coefficient comparison

**6.5 AEC3 Echo Estimation**
- [ ] Port `erl_estimator.cc`
- [ ] Port `erle_estimator.cc` / `fullband_erle_estimator.cc`
- [ ] Port `signal_dependent_erle_estimator.cc`
- [ ] Port `residual_echo_estimator.cc`
- [ ] Port `reverb_model.cc` / `reverb_model_estimator.cc`
- [ ] Proptest: ERLE/ERL estimation

**6.6 AEC3 Suppression**
- [ ] Port `render_signal_analyzer.cc`
- [ ] Port `comfort_noise_generator.cc`
- [ ] Port `suppression_gain.cc`
- [ ] Port `suppression_filter.cc`
- [ ] Port `dominant_nearend_detector.cc` / `subband_nearend_detector.cc`
- [ ] Proptest: Suppression mask comparison

**6.7 AEC3 State & Integration**
- [ ] Port `aec_state.cc`
- [ ] Port `echo_audibility.cc`
- [ ] Port `echo_path_variability.cc`
- [ ] Port `echo_remover.cc`
- [ ] Port `block_processor.cc`
- [ ] Port `echo_canceller3.cc`
- [ ] Port `config_selector.cc`
- [ ] Proptest: Full AEC3 pipeline

### Phase 7: Audio Processing Integration (3-4 weeks)

**7.1 Audio Buffer & Utilities**
- [ ] Port `audio_buffer.cc`
- [ ] Port `AudioFrame` handling

**7.2 Component Wrappers**
- [ ] Port `high_pass_filter.cc`
- [ ] Port `gain_controller2.cc`
- [ ] Port `residual_echo_detector.cc`
- [ ] Port `splitting_filter.cc` / `three_band_filter_bank.cc`
- [ ] Port level adjustment components

**7.3 Main Implementation**
- [ ] Port `audio_processing_impl.cc`
- [ ] Port `audio_processing_builder_impl.cc`
- [ ] Proptest: Full ProcessStream comparison

### Phase 8: C API & Final Integration (2-3 weeks)

**8.1 C API Design**
```c
// Example C API
typedef struct WapAudioProcessing WapAudioProcessing;
typedef struct WapConfig WapConfig;

WapAudioProcessing* wap_audio_processing_create(const WapConfig* config);
void wap_audio_processing_destroy(WapAudioProcessing* apm);

int wap_process_stream_i16(
    WapAudioProcessing* apm,
    const int16_t* src,
    int input_sample_rate,
    size_t input_channels,
    int output_sample_rate,
    size_t output_channels,
    int16_t* dest
);

int wap_process_stream_f32(
    WapAudioProcessing* apm,
    const float* const* src,
    int input_sample_rate,
    size_t input_channels,
    int output_sample_rate,
    size_t output_channels,
    float* const* dest
);

// Similar for ProcessReverseStream, config getters/setters, etc.
```

**8.2 Implementation**
- [ ] Define C API in `crates/webrtc-apm/src/ffi.rs`
- [ ] Generate C headers with cbindgen
- [ ] Implement all C API functions
- [ ] Create pkg-config file

**8.3 Integration Testing**
- [ ] Modify C++ tests to optionally use Rust library via C API
- [ ] Verify all 2458 tests pass with Rust implementation
- [ ] Performance benchmarking vs C++ implementation

### Phase 9: Documentation & Release (1-2 weeks)

**9.1 Documentation**
- [ ] API documentation (rustdoc)
- [ ] C API documentation
- [ ] Migration guide from C++ library
- [ ] Architecture documentation

**9.2 Release Preparation**
- [ ] Version 0.1.0 release
- [ ] Publish to crates.io
- [ ] Create release binaries for major platforms

## Testing Strategy (Implemented)

### Property-Based Testing with test-strategy

Uses `proptest` + `test-strategy` derive macros for clean, readable tests:

```rust
use webrtc_apm_proptest::generators::*;
use test_strategy::proptest;

#[proptest]
fn fir_filter_preserves_length(frame: MonoFrameF32) {
    let output = fir_filter(&coeffs, &frame.samples);
    assert_eq!(output.len(), frame.samples.len());
}
```

Available `#[derive(Arbitrary)]` types: `SampleRate`, `ChannelCount`, `MonoFrameF32`, `MonoFrameI16`, `MultiChannelFrameF32`.

Comparison utilities: `assert_f32_near`, `assert_f32_relative`, `assert_i16_exact`, `compare_f32` (detailed stats).

### Test Runner

**Always use `cargo nextest run`** (not `cargo test`). Each test runs in its own process with parallel execution.

### Bitexactness Verification

For algorithms requiring exact output (e.g., integer processing):
- Compare byte-for-byte output
- Use existing C++ test reference data

### Verification Commands

```bash
cargo build --workspace                    # Build
cargo nextest run --workspace              # Test (128 tests currently)
cargo clippy --workspace --all-targets     # Lint (zero warnings required)
meson test -C builddir                     # C++ tests still pass
```

## SIMD Implementation Strategy (Implemented)

### Approach

`SimdBackend` enum with `std::arch` intrinsics. No trait objects — the enum is `Copy + Eq` and dispatches via `match`. Backend modules export `pub(crate)` free functions.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

// Current operations: dot_product, dual_dot_product, multiply_accumulate, sum
// Operations added incrementally as porting phases require them
pub fn detect_backend() -> SimdBackend { /* runtime feature detection */ }
```

### ARM Assembly

For ARM32 assembly files (`.S`), two options:
1. Use `cc` crate to compile original assembly
2. Port to Rust inline assembly or NEON intrinsics

Recommended: Port to NEON intrinsics where possible, keep assembly for ARMv7-specific optimizations via `cc` crate.

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Floating-point differences | Use tolerance-based comparison; match FPU flags (denormals, rounding) |
| SIMD instruction differences | Exact intrinsic mapping; verify intermediate values |
| Memory layout differences | Match C++ struct layouts with `#[repr(C)]` |
| Thread safety changes | Mirror C++ threading model with explicit synchronization |
| Build complexity | Comprehensive CI; clear feature flags |
| Performance regression | Continuous benchmarking; optimization pass before release |

## Success Criteria

- [ ] All 2432 C++ tests pass when using Rust implementation
- [ ] Property tests demonstrate equivalence for all components
- [ ] Performance within 10% of C++ implementation
- [ ] Builds on Linux x86_64, ARM64; macOS x86_64, ARM64; Windows x86_64
- [ ] All SIMD paths (SSE2, AVX2, NEON) functional and verified
- [ ] C API documented and usable from C/C++ projects
- [ ] Published to crates.io with complete documentation
- [ ] `cargo clippy --all-targets` zero warnings
- [ ] `cargo nextest run` all tests green

## Reference

- **C++ Source:** WebRTC M145 (branch-heads/7632)
- **Library Version:** 3.0
- **Test Count:** 2432 passing (C++), 199 passing (Rust, Phases 1-2 + 5)
- **Build System:** Meson (C++), Cargo (Rust)
- **C++ Standard:** C++20
- **Rust Edition:** 2024, MSRV 1.91, resolver 3
