# Plan: Update Docs, FFT SIMD, Noise Suppression

## Part 0: Update Plan Docs to Current State

The following docs are stale and need updating before proceeding:

### MEMORY.md
- Test count: 118 → **128** (29 webrtc-fft, 16 webrtc-ring-buffer, 16 webrtc-simd, 18 webrtc-apm-proptest, 33 webrtc-common-audio, 16 webrtc-ring-buffer proptest — actually 128 total)
- Phase status: "Phase 2 In Progress" → note FFT and resampler complete
- Tech: remove "pffft via cc crate" → "pffft ported to pure Rust in webrtc-fft crate"

### phase-02-common-audio.md
- Status: "In Progress (ring buffer complete)" → "In Progress (ring buffer, audio utils, channel buffer, resampler, FFT complete)"
- Section 2.2 Audio Utilities: mark audio_util and channel_buffer as **DONE** (commits `b7c105c`, `0ab5248`). Note: smoothing_filter not yet ported but no downstream consumer needs it yet
- Section 2.3 FIR Filter: note "deferred — only consumed by AEC3 (Phase 6)"
- Section 2.4 Resampler: mark as **DONE** (commits `9d49769`, `fdb098c`). Note: SincResampler SIMD ported (uses webrtc-simd dispatch), PushSincResampler and PushResampler done
- Section 2.5 FFT: mark as **DONE** — but note implementation differs from plan:
  - FFT ported as standalone `webrtc-fft` crate (not inside `webrtc-common-audio`)
  - PFFFT ported to pure Rust (not via cc crate as planned)
  - Ooura 128 scalar done, SSE2/NEON SIMD still pending
  - Ooura fft4g (256-point) scalar done (no SIMD variants exist in C++)
  - 29 tests (19 unit + 10 proptest)
- Completion checklist: update checkmarks
- Commit summary: update with actual commits

### README.md (rust-port)
- Phase 2 status: "In Progress" with more detail
- Test count in technology section: update
- Dependency graph: note Phase 3 VAD depends on SPL subset (6 functions), not blocked on full SPL port
- Note that Phase 3 (VAD) core depends on SPL but modern pipeline (AGC2, NS, AEC3) doesn't use core VAD at all — Phase 4 AGC2 has its own RNN VAD

### rust-port.md (master plan)
- Phase 2 subtasks: mark completed items
- Test count: "50 passing (Rust, Phase 1 + ring buffer)" → "128 passing"
- Phase 2.5 FFT: note pure Rust implementation (not cc crate)

---

## Part A: Ooura 128-point FFT SIMD (webrtc-fft)

Add SSE2 and NEON variants of the 128-point Ooura FFT. Currently only the scalar path is ported.

### SIMD Architecture

The C++ `OouraFft` class dispatches 4 inner functions to SSE2 or scalar at runtime:
- `cft1st_128` — first butterfly pass
- `cftmdl_128` — middle butterfly pass
- `rftfsub_128` — forward real-to-complex post-processing
- `rftbsub_128` — backward complex-to-real pre-processing

NEON variants exist as separate free functions with the same signatures.

**C++ sources (~776 lines total):**
- `webrtc/common_audio/third_party/ooura/fft_size_128/ooura_fft_sse2.cc` (425 lines)
- `webrtc/common_audio/third_party/ooura/fft_size_128/ooura_fft_neon.cc` (351 lines)
- `webrtc/common_audio/third_party/ooura/fft_size_128/ooura_fft_tables_neon_sse2.h` (shared twiddle tables)

### Implementation

Add SIMD modules to `webrtc-fft`:
```
crates/webrtc-fft/src/
├── ooura_fft.rs           # existing scalar (becomes fallback)
├── ooura_fft_sse2.rs      # SSE2 intrinsics (x86/x86_64)
└── ooura_fft_neon.rs      # NEON intrinsics (aarch64)
```

The public API stays the same (`forward`/`inverse` free functions). Add runtime dispatch:
```rust
pub fn forward(a: &mut [f32; 128]) {
    // dispatch to SSE2, NEON, or scalar based on platform
}
```

This requires `unsafe` for `std::arch` intrinsics — relax `#![deny(unsafe_code)]` to `#![cfg_attr(not(...), deny(unsafe_code))]` or use a per-module `#![allow(unsafe_code)]`.

### Tests

Existing tests already cover correctness. Add:
- Test that SIMD path produces identical output to scalar path (proptest)
- Explicit backend selection for testing each path independently

### Consumers

- **AEC3** (Phase 6) — only consumer of 128-point FFT

### Commits

1. `feat(rust): add SSE2 SIMD for Ooura 128-point FFT`
2. `feat(rust): add NEON SIMD for Ooura 128-point FFT`

### Deferred: PFFFT SIMD

PFFFT SIMD (~2000+ lines of macro-heavy SIMD code) is used only by AGC2 RNN VAD. Deferred to Phase 4 when AGC2 is ported. The scalar PFFFT already works correctly.

---

## Part B: Noise Suppression (Phase 5)

### Overview

Port the Noise Suppression (NS) module to the existing `webrtc-ns` crate. NS is self-contained — no dependency on VAD, AGC, or SPL. It only needs `Fft4g` (256-point, already in `webrtc-fft`, scalar-only in C++ too) and basic math.

**~2,460 lines across 28 files (15 headers + 13 implementations) → ~13 Rust modules.**

## Architecture

```
crates/webrtc-ns/
├── Cargo.toml          # deps: webrtc-fft, webrtc-common-audio, tracing
└── src/
    ├── lib.rs                          # pub API: NoiseSuppressor, SuppressionLevel
    ├── common.rs                       # constants (FFT_SIZE, NS_FRAME_SIZE, etc.)
    ├── config.rs                       # NsConfig, SuppressionLevel
    ├── suppression_params.rs           # SuppressionParams (per-level tuning)
    ├── fast_math.rs                    # fast log2/exp/pow/sqrt approximations
    ├── ns_fft.rs                       # NrFft wrapper around Fft4g (256-point)
    ├── signal_model.rs                 # SignalModel struct
    ├── prior_signal_model.rs           # PriorSignalModel struct
    ├── histograms.rs                   # Histograms for feature tracking
    ├── prior_signal_model_estimator.rs # PriorSignalModelEstimator
    ├── signal_model_estimator.rs       # SignalModelEstimator
    ├── quantile_noise_estimator.rs     # QuantileNoiseEstimator
    ├── noise_estimator.rs              # NoiseEstimator
    ├── speech_probability_estimator.rs # SpeechProbabilityEstimator
    ├── wiener_filter.rs                # WienerFilter
    └── noise_suppressor.rs             # NoiseSuppressor (main pipeline)
```

### Dependencies

- `webrtc-fft` — `Fft4g` for 256-point real FFT
- `webrtc-common-audio` — `ChannelBuffer<f32>` for audio I/O
- `tracing` — instrumentation (direct dep, not feature-gated)
- `proptest` + `test-strategy` — dev-deps for property tests

### Interface Decision

C++ uses `AudioBuffer` which combines `ChannelBuffer` with splitting filter and other concerns. For the Rust port, `NoiseSuppressor` takes `&ChannelBuffer<f32>` / `&mut ChannelBuffer<f32>` directly — the higher-level `AudioBuffer` will be assembled in Phase 7 (Integration).

## C++ Source → Rust Module Mapping

| C++ Source | Rust Module | Lines | Notes |
|------------|-------------|-------|-------|
| `ns_common.h` | `common.rs` | 34 | Constants only |
| `ns_config.h` | `config.rs` | 24 | Struct + enum, no .cc |
| `suppression_params.h/cc` | `suppression_params.rs` | 80 | Per-level params |
| `fast_math.h/cc` | `fast_math.rs` | 123 | log2/exp/pow via bit tricks |
| `ns_fft.h/cc` | `ns_fft.rs` | 115 | Thin wrapper around Fft4g |
| `signal_model.h/cc` | `signal_model.rs` | 74 | Data struct |
| `prior_signal_model.h/cc` | `prior_signal_model.rs` | 50 | Data struct |
| `histograms.h/cc` | `histograms.rs` | 104 | Feature histograms |
| `prior_signal_model_estimator.h/cc` | `prior_signal_model_estimator.rs` | 211 | Histogram → thresholds |
| `signal_model_estimator.h/cc` | `signal_model_estimator.rs` | 239 | LRT, flatness, diff |
| `quantile_noise_estimator.h/cc` | `quantile_noise_estimator.rs` | 139 | 3-quantile tracking |
| `noise_estimator.h/cc` | `noise_estimator.rs` | 283 | White/pink noise model |
| `speech_probability_estimator.h/cc` | `speech_probability_estimator.rs` | 161 | Bayesian speech prob |
| `wiener_filter.h/cc` | `wiener_filter.rs` | 181 | Decision-directed SNR |
| `noise_suppressor.h/cc` | `noise_suppressor.rs` | 657 | Full pipeline |

**Not ported** (don't exist in source):
- `noise_suppressor_init.cc` — init is in constructor
- `analyze_frame.cc`, `quantization_util.cc`, `suppression_filter.cc` — don't exist; overlap-add is inline in `noise_suppressor.cc`

## Implementation Steps

### Step 1: Foundation (constants, config, fast math, suppression params)

**Files:** `common.rs`, `config.rs`, `suppression_params.rs`, `fast_math.rs`

Key types:
```rust
// common.rs
pub const FFT_SIZE: usize = 256;
pub const FFT_SIZE_BY_2_PLUS_1: usize = 129;
pub const NS_FRAME_SIZE: usize = 160;
pub const OVERLAP_SIZE: usize = 96;

// config.rs
pub enum SuppressionLevel { Low, Moderate, High, VeryHigh }

// fast_math.rs — bit-manipulation approximations, must be exact
pub fn log_approximation(x: f32) -> f32;
pub fn exp_approximation(x: f32) -> f32;
pub fn sqrt_fast_approximation(x: f32) -> f32;
```

**Critical:** `fast_math.rs` uses `f32::to_bits()`/`from_bits()` for the fast log2 approximation (union trick in C). Must match C exactly.

**Tests:** Unit tests for fast_math edge cases; proptest comparing ranges.

**Commit:** `feat(rust): add NS foundation (config, fast math, suppression params)`

### Step 2: FFT wrapper + data models

**Files:** `ns_fft.rs`, `signal_model.rs`, `prior_signal_model.rs`, `histograms.rs`

```rust
// ns_fft.rs — wraps webrtc_fft::fft4g::Fft4g
pub struct NrFft { fft: Fft4g }
impl NrFft {
    pub fn fft(&self, time_data: &mut [f32; FFT_SIZE],
               real: &mut [f32; FFT_SIZE], imag: &mut [f32; FFT_SIZE]);
    pub fn ifft(&self, real: &[f32], imag: &[f32], time_data: &mut [f32]);
}
```

**Tests:** FFT roundtrip; histogram clear/update.

**Commit:** `feat(rust): port NS FFT wrapper and signal models`

### Step 3: Noise estimation

**Files:** `quantile_noise_estimator.rs`, `noise_estimator.rs`

The quantile estimator tracks 3 simultaneous quantiles per frequency bin. The noise estimator combines quantile-based and parametric (white/pink) noise models.

**Tests:** Multi-frame convergence tests; proptest with random spectra.

**Commit:** `feat(rust): port NS noise estimator`

### Step 4: Speech probability estimation

**Files:** `prior_signal_model_estimator.rs`, `signal_model_estimator.rs`, `speech_probability_estimator.rs`

Computes per-bin speech probability using LRT, spectral flatness, and spectral difference features.

**Tests:** Feature extraction correctness; probability bounds [0, 1].

**Commit:** `feat(rust): port NS speech probability estimator`

### Step 5: Wiener filter

**Files:** `wiener_filter.rs`

Decision-directed SNR estimation → Wiener filter gains. Startup phase blending during first 50 blocks.

**Tests:** Gain bounds [min_attenuation, 1.0]; proptest with random spectra.

**Commit:** `feat(rust): port NS Wiener filter`

### Step 6: Main pipeline

**Files:** `noise_suppressor.rs`

Private helpers (ported from `noise_suppressor.cc` anonymous namespace):
- `form_extended_frame()` — prepend overlap memory
- `apply_filter_bank_window()` — hybrid Hanning window (`FILTER_BANK_WINDOW` const table, 96 elements)
- `compute_magnitude_spectrum()` — uses `sqrt_fast_approximation`
- `compute_snr()` — prior/post SNR
- `compute_upper_bands_gain()` — gain for bands > 16kHz
- `overlap_and_add()` — synthesis
- `delay_signal()` — upper band delay compensation

Public API:
```rust
pub struct NoiseSuppressor { /* per-channel state, shared FFT, params */ }
impl NoiseSuppressor {
    pub fn new(level: SuppressionLevel, sample_rate_hz: usize, num_channels: usize) -> Self;
    pub fn analyze(&mut self, audio: &ChannelBuffer<f32>);
    pub fn process(&mut self, audio: &mut ChannelBuffer<f32>);
    pub fn set_capture_output_used(&mut self, used: bool);
}
```

**Tests:** Multi-channel identical-output test (matches C++ `IdenticalChannelEffects` unittest); multi-frame processing; proptest with random audio.

**Commit:** `feat(rust): port NoiseSuppressor pipeline`

## Verification

```bash
cargo nextest run -p webrtc-ns         # All NS tests pass
cargo nextest run --workspace          # All workspace tests pass
cargo clippy --workspace --all-targets # Zero warnings
```

## Full Commit Plan

0. `docs: update Rust port plans to reflect Phase 2 completion`
1. `feat(rust): add SSE2 SIMD for Ooura 128-point FFT`
2. `feat(rust): add NEON SIMD for Ooura 128-point FFT`
3. `feat(rust): add NS foundation (config, fast math, suppression params)`
4. `feat(rust): port NS FFT wrapper and signal models`
5. `feat(rust): port NS noise estimator`
6. `feat(rust): port NS speech probability estimator`
7. `feat(rust): port NS Wiener filter`
8. `feat(rust): port NoiseSuppressor pipeline`
