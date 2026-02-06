# Phase 4: Automatic Gain Control (AGC2 only)

**Status:** Complete (all 9 steps done)
**Estimated Duration:** 3-4 weeks
**Dependencies:** Phase 2 (Common Audio)
**Outcome:** The `webrtc-agc2` crate contains working AGC2 and RNN VAD implementations. All gain computations match the C++ reference within specified tolerance.

**AGC1 is excluded** (deprecated, SPL-dependent — see master plan rationale).

## Progress

| Step | Description | Status | Commit | Tests |
|------|-------------|--------|--------|-------|
| 1 | Crate scaffold + constants | Done | `34917f3` | 0 |
| 2 | RNN VAD foundation (weights, activations, data structures) | Done | `c6eb9cc` | 20 |
| 3 | RNN VAD neural network layers (FC, GRU) | Done | `6e1163d` | 24 |
| 4a | RNN VAD LP residual + auto-correlation | Done | `77b85a2` | 29 |
| 4b | RNN VAD spectral features | Done | `0e3f213` | 36 |
| 4c | RNN VAD pitch search | Done | `ca27b41` | 42 |
| 4d | RNN VAD feature extraction pipeline | Done | `7a08835` | 43 |
| 5a | RNN VAD model (rnn.rs) | Done | `9126077` | 46 |
| 5b | VAD wrapper + FeatureVector refactor | Done | `c08e95a` | 50 |
| 6a | Biquad filter | Done | `ac6ad17` | 55 |
| 6b | Gain curves (limiter_db + interpolated) | Done | `15dd985` | 69 |
| 6c | Gain applier | Done | `c8c4cfc` | 73 |
| 7 | Level estimators + saturation protector | Done | `87622d0` | 124 |
| 8 | Limiter + adaptive digital gain controller | Done | `e26be90` | 141 |
| 9 | Clipping predictor | Done | `6295021` | 165 |

**Current:** 165 tests in webrtc-agc2, 364 across workspace.

---

## Overview

Port the AGC2 module and its RNN VAD to Rust. AGC2 is the modern gain controller used by the default audio processing pipeline. It includes an RNN-based voice activity detector, adaptive digital gain, limiter, and input volume controller.

**Scope:** ~5,500 lines (AGC2 core) + ~3,200 lines (RNN VAD) + ~540 lines (rnnoise weights/activations) = **~9,200 lines total** across 38 source files → new `webrtc-agc2` crate.

---

## Key Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| `webrtc-fft` (PFFFT) | RNN VAD spectral features, auto-correlation | Already ported (scalar) |
| `webrtc-simd` (dot_product) | RNN VAD VectorMath | Already ported (SSE2/AVX2/NEON) |
| `webrtc-common-audio` (PushResampler) | VAD wrapper resamples to 24kHz | Already ported |
| `webrtc-common-audio` (audio_util) | `FloatS16ToFloat` etc. | Already ported |
| `AudioBuffer` | `input_volume_controller`, `gain_controller2` top-level | **Not ported — defer to Phase 7** |
| `AudioFrameView` | `gain_applier`, `clipping_predictor` | Simple view type, port inline |
| `ApmDataDumper` | Debug logging in many AGC2 files | No-op (WEBRTC_APM_DEBUG_DUMP=0), skip entirely |
| `FieldTrialsView` | `input_volume_controller`, `speech_level_estimator` | Use default values (no field trial system in Rust port) |
| `metrics.h` | `input_volume_stats_reporter` | Skip metrics reporting |

## What to Defer to Phase 7

These depend on `AudioBuffer` (not yet ported):
- `input_volume_controller.cc` — uses `AudioBuffer` for multi-channel analysis
- `gain_controller2.cc` — top-level wrapper that orchestrates everything with `AudioBuffer`
- `input_volume_stats_reporter.cc` — uses `metrics.h` for WebRTC stats

Everything else is self-contained and can be ported now.

---

## Architecture

```
crates/webrtc-agc2/
├── Cargo.toml          # deps: webrtc-fft, webrtc-simd, webrtc-common-audio, tracing
└── src/
    ├── lib.rs
    ├── common.rs                    # agc2_common.h constants
    ├── biquad_filter.rs             # IIR biquad filter
    ├── limiter_db_gain_curve.rs     # dB gain curve math
    ├── interpolated_gain_curve.rs   # lookup-table gain curve
    ├── gain_applier.rs              # per-sample gain with ramping
    ├── limiter.rs                   # output limiter
    ├── fixed_digital_level_estimator.rs  # peak level estimation
    ├── noise_level_estimator.rs     # noise floor estimation
    ├── speech_level_estimator.rs    # speech level estimation
    ├── saturation_protector.rs      # headroom management
    ├── saturation_protector_buffer.rs
    ├── speech_probability_buffer.rs
    ├── adaptive_digital_gain_controller.rs  # main adaptive gain logic
    ├── clipping_predictor.rs        # clipping prediction
    ├── clipping_predictor_level_buffer.rs
    ├── vad_wrapper.rs               # VAD wrapper (resamples → RNN VAD)
    └── rnn_vad/
        ├── mod.rs
        ├── common.rs                # constants (sample rates, pitch, feature dims)
        ├── activations.rs           # TansigApproximated, sigmoid, ReLU (from rnnoise)
        ├── weights.rs               # NN weight tables (from rnnoise)
        ├── vector_math.rs           # dot product dispatch (uses webrtc-simd)
        ├── ring_buffer.rs           # fixed-size ring buffer (header-only in C++)
        ├── sequence_buffer.rs       # fixed-size sequence buffer (header-only)
        ├── symmetric_matrix_buffer.rs  # symmetric matrix buffer (header-only)
        ├── fc_layer.rs              # fully-connected NN layer
        ├── gru_layer.rs             # GRU recurrent layer
        ├── rnn.rs                   # full RNN model (FC → GRU → FC → output)
        ├── lp_residual.rs           # linear prediction residual
        ├── auto_correlation.rs      # auto-correlation (uses PFFFT)
        ├── spectral_features_internal.rs
        ├── spectral_features.rs     # spectral feature extraction (uses PFFFT)
        ├── pitch_search_internal.rs # pitch period search helpers
        ├── pitch_search.rs          # pitch search
        └── features_extraction.rs   # complete feature extraction pipeline
```

## SIMD

The only SIMD in AGC2 is `VectorMath::DotProduct` in the RNN VAD:
- **SSE2**: inline in `vector_math.h` — 4-wide multiply-add
- **NEON**: inline in `vector_math.h` — 4-wide FMA
- **AVX2**: separate `vector_math_avx2.cc` — 8-wide FMA

This maps directly to `webrtc_simd::dot_product()` which already has all three backends. No new SIMD code needed — just call `webrtc_simd::dot_product`.

---

## Implementation Steps

### Step 1: Crate scaffold + constants

Create `crates/webrtc-agc2/` with `Cargo.toml` and `common.rs` (port `agc2_common.h` constants).

**Commit:** `feat(rust): add webrtc-agc2 crate scaffold`

### Step 2: RNN VAD — weights, activations, data structures

Port the foundation pieces that have no internal dependencies:
- `activations.rs` — `TansigApproximated` (201-entry lookup table + interpolation), `SigmoidApproximated`, `RectifiedLinearUnit` from `rnnoise/src/rnn_activations.h`
- `weights.rs` — const arrays from `rnnoise/src/rnn_vad_weights.cc` (401 lines of float data)
- `common.rs` (rnn_vad) — constants from `rnn_vad/common.h`
- `ring_buffer.rs` — header-only fixed-size ring buffer
- `sequence_buffer.rs` — header-only sequence buffer
- `symmetric_matrix_buffer.rs` — header-only symmetric matrix buffer
- `vector_math.rs` — delegates to `webrtc_simd::dot_product`

**Critical:** The tanh approximation (`TansigApproximated`) uses a 201-entry lookup table with linear interpolation. Must match C++ exactly for bit-exact NN inference.

**Tests:** Port from 5 C++ test files: `ring_buffer_unittest`, `sequence_buffer_unittest`, `symmetric_matrix_buffer_unittest`, `vector_math_unittest` + activation function unit tests.

**Commit:** `feat(rust): port RNN VAD foundation (weights, activations, data structures)`

### Step 3: RNN VAD — neural network layers

- `fc_layer.rs` — fully-connected layer: `output = activation(weights × input + bias)`. Uses `vector_math::dot_product` for each output neuron.
- `gru_layer.rs` — GRU layer: update gate, reset gate, candidate state. Three weight matrices + recurrent weights. Uses `vector_math::dot_product`.

**C++ sources:** `rnn_fc.cc` (109 lines), `rnn_gru.cc` (208 lines)

**Tests:** Port `rnn_fc_unittest`, `rnn_gru_unittest`.

**Commit:** `feat(rust): port RNN VAD neural network layers (FC, GRU)`

### Step 4: RNN VAD — feature extraction

Port bottom-up:
1. `lp_residual.rs` — LP residual via Levinson-Durbin (139 lines)
2. `auto_correlation.rs` — uses PFFFT for FFT-based auto-correlation (94 lines)
3. `spectral_features_internal.rs` — band energy computation, cepstral coefficients (191 lines)
4. `spectral_features.rs` — full spectral feature extractor using PFFFT (221 lines)
5. `pitch_search_internal.rs` — pitch search helpers (515 lines — largest file)
6. `pitch_search.rs` — pitch period search at 12kHz/24kHz
7. `features_extraction.rs` — ties everything together into 42-element feature vector (94 lines)

**Tests:** Port `lp_residual_unittest`, `auto_correlation_unittest`, `spectral_features_internal_unittest`, `spectral_features_unittest`, `pitch_search_internal_unittest`, `pitch_search_unittest`, `features_extraction_unittest`.

**Commits:**
1. `feat(rust): port RNN VAD LP residual and auto-correlation`
2. `feat(rust): port RNN VAD spectral features`
3. `feat(rust): port RNN VAD pitch search`
4. `feat(rust): port RNN VAD feature extraction pipeline`

### Step 5: RNN VAD — model + wrapper

- `rnn.rs` — full RNN model: input FC (42→24) → GRU (24→24) → output FC (24→1) → sigmoid (96 lines)
- `vad_wrapper.rs` — resamples input to 24kHz via `PushResampler`, extracts features, runs RNN, outputs speech probability

**Tests:** Port `rnn_unittest`, `rnn_vad_unittest`, `vad_wrapper_unittest`.

**Commit:** `feat(rust): port RNN VAD model and wrapper`

### Step 6: AGC2 building blocks

- `biquad_filter.rs` — simple IIR biquad (cascadable)
- `limiter_db_gain_curve.rs` — piecewise dB gain curve
- `interpolated_gain_curve.rs` — lookup-table approximation of gain curve
- `gain_applier.rs` — applies gain per-sample with linear ramping

**Tests:** Port `biquad_filter_unittest`, `limiter_db_gain_curve_unittest`, `interpolated_gain_curve_unittest`, `gain_applier_unittest`.

**Commits:**
1. `feat(rust): port AGC2 biquad filter`
2. `feat(rust): port AGC2 gain curves and gain applier`

### Step 7: AGC2 level estimators

- `fixed_digital_level_estimator.rs` — peak envelope tracking
- `noise_level_estimator.rs` — minimum statistics noise floor
- `speech_level_estimator.rs` — speech level with confidence tracking
- `saturation_protector.rs` + `saturation_protector_buffer.rs` — headroom management
- `speech_probability_buffer.rs` — ring buffer of speech probabilities

**Tests:** Port `fixed_digital_level_estimator_unittest`, `noise_level_estimator_unittest`, `speech_level_estimator_unittest`, `saturation_protector_unittest`, `saturation_protector_buffer_unittest`, `speech_probability_buffer_unittest`.

**Commits:**
1. `feat(rust): port AGC2 level estimators`
2. `feat(rust): port AGC2 saturation protector`

### Step 8: AGC2 adaptive digital controller

- `adaptive_digital_gain_controller.rs` — main adaptive gain logic combining VAD + level estimators + limiter

**Tests:** Port `adaptive_digital_gain_controller_unittest`.

**Commit:** `feat(rust): port AGC2 adaptive digital gain controller`

### Step 9: AGC2 clipping predictor

- `clipping_predictor.rs` — clipping prediction (uses `AudioFrameView` — port as simple `&[&[f32]]` view)
- `clipping_predictor_level_buffer.rs` — level tracking buffer

**Tests:** Port `clipping_predictor_unittest`, `clipping_predictor_level_buffer_unittest`.

**Commit:** `feat(rust): port AGC2 clipping predictor`

### Deferred to Phase 7

- `input_volume_controller.rs` — depends on `AudioBuffer`
- `gain_controller2.rs` — top-level orchestrator, depends on `AudioBuffer`
- `input_volume_stats_reporter.rs` — depends on `metrics.h`

---

## C++ Reference Files

### AGC2 Core (in `webrtc/modules/audio_processing/agc2/`)
| File | Lines | Rust Target |
|------|-------|-------------|
| `agc2_common.h` | 50 | `common.rs` |
| `biquad_filter.cc/h` | 60 | `biquad_filter.rs` |
| `limiter_db_gain_curve.cc/h` | 140 | `limiter_db_gain_curve.rs` |
| `interpolated_gain_curve.cc/h` | 160 | `interpolated_gain_curve.rs` |
| `compute_interpolated_gain_curve.cc/h` | 200 | inline in `interpolated_gain_curve.rs` |
| `gain_applier.cc/h` | 80 | `gain_applier.rs` |
| `limiter.cc/h` | 120 | `limiter.rs` |
| `fixed_digital_level_estimator.cc/h` | 100 | `fixed_digital_level_estimator.rs` |
| `noise_level_estimator.cc/h` | 140 | `noise_level_estimator.rs` |
| `speech_level_estimator.cc/h` | 80 | `speech_level_estimator.rs` |
| `speech_level_estimator_impl.cc/h` | 100 | inline in `speech_level_estimator.rs` |
| `speech_level_estimator_experimental_impl.cc/h` | 100 | inline in `speech_level_estimator.rs` |
| `saturation_protector.cc/h` | 120 | `saturation_protector.rs` |
| `saturation_protector_buffer.cc/h` | 80 | `saturation_protector_buffer.rs` |
| `speech_probability_buffer.cc/h` | 70 | `speech_probability_buffer.rs` |
| `adaptive_digital_gain_controller.cc/h` | 200 | `adaptive_digital_gain_controller.rs` |
| `clipping_predictor.cc/h` | 200 | `clipping_predictor.rs` |
| `clipping_predictor_level_buffer.cc/h` | 80 | `clipping_predictor_level_buffer.rs` |
| `vad_wrapper.cc/h` | 120 | `vad_wrapper.rs` |
| `cpu_features.cc/h` | 60 | use `webrtc_simd::detect_backend()` |
| `gain_map_internal.h` | 30 | inline in `clipping_predictor.rs` |

### RNN VAD (in `webrtc/modules/audio_processing/agc2/rnn_vad/`)
| File | Lines | Rust Target |
|------|-------|-------------|
| `common.h` | 79 | `rnn_vad/common.rs` |
| `vector_math.h` + `vector_math_avx2.cc` | 166 | `rnn_vad/vector_math.rs` (delegates to webrtc-simd) |
| `ring_buffer.h` | 55 | `rnn_vad/ring_buffer.rs` |
| `sequence_buffer.h` | 77 | `rnn_vad/sequence_buffer.rs` |
| `symmetric_matrix_buffer.h` | 96 | `rnn_vad/symmetric_matrix_buffer.rs` |
| `rnn_fc.cc/h` | 182 | `rnn_vad/fc_layer.rs` |
| `rnn_gru.cc/h` | 268 | `rnn_vad/gru_layer.rs` |
| `rnn.cc/h` | 146 | `rnn_vad/rnn.rs` |
| `lp_residual.cc/h` | 175 | `rnn_vad/lp_residual.rs` |
| `auto_correlation.cc/h` | 134 | `rnn_vad/auto_correlation.rs` |
| `spectral_features_internal.cc/h` | 291 | `rnn_vad/spectral_features_internal.rs` |
| `spectral_features.cc/h` | 299 | `rnn_vad/spectral_features.rs` |
| `pitch_search_internal.cc/h` | 626 | `rnn_vad/pitch_search_internal.rs` |
| `pitch_search.cc/h` | 95 | `rnn_vad/pitch_search.rs` |
| `features_extraction.cc/h` | 145 | `rnn_vad/features_extraction.rs` |

### RNNoise (in `webrtc/third_party/rnnoise/src/`)
| File | Lines | Rust Target |
|------|-------|-------------|
| `rnn_activations.h` | 102 | `rnn_vad/activations.rs` |
| `rnn_vad_weights.cc/h` | 438 | `rnn_vad/weights.rs` |

---

## Test Files (33 total)

### AGC2 Core (18 tests in `tests/unit/agc2/`)
- `biquad_filter_unittest.cc`
- `limiter_db_gain_curve_unittest.cc`
- `interpolated_gain_curve_unittest.cc`
- `gain_applier_unittest.cc`
- `limiter_unittest.cc`
- `fixed_digital_level_estimator_unittest.cc`
- `noise_level_estimator_unittest.cc`
- `speech_level_estimator_unittest.cc`
- `saturation_protector_unittest.cc`
- `saturation_protector_buffer_unittest.cc`
- `speech_probability_buffer_unittest.cc`
- `adaptive_digital_gain_controller_unittest.cc`
- `clipping_predictor_unittest.cc`
- `clipping_predictor_level_buffer_unittest.cc`
- `input_volume_controller_unittest.cc` → **deferred (AudioBuffer)**
- `input_volume_stats_reporter_unittest.cc` → **deferred (metrics)**
- `agc2_testing_common_unittest.cc`
- `vad_wrapper_unittest.cc`

### RNN VAD (15 tests in `tests/unit/agc2/rnn_vad/`)
- `auto_correlation_unittest.cc`
- `features_extraction_unittest.cc`
- `lp_residual_unittest.cc`
- `pitch_search_internal_unittest.cc`
- `pitch_search_unittest.cc`
- `ring_buffer_unittest.cc`
- `rnn_fc_unittest.cc`
- `rnn_gru_unittest.cc`
- `rnn_unittest.cc`
- `rnn_vad_unittest.cc`
- `sequence_buffer_unittest.cc`
- `spectral_features_internal_unittest.cc`
- `spectral_features_unittest.cc`
- `symmetric_matrix_buffer_unittest.cc`
- `vector_math_unittest.cc`

---

## Verification

```bash
cargo nextest run -p webrtc-agc2       # All AGC2 tests pass
cargo nextest run --workspace          # All workspace tests pass (199 + new)
cargo clippy --workspace --all-targets # Zero warnings
```

## Commit Summary

| # | Commit | Scope |
|---|--------|-------|
| 1 | `feat(rust): add webrtc-agc2 crate scaffold` | Scaffold + constants |
| 2 | `feat(rust): port RNN VAD foundation` | Weights, activations, data structures |
| 3 | `feat(rust): port RNN VAD neural network layers` | FC, GRU |
| 4 | `feat(rust): port RNN VAD LP residual and auto-correlation` | Signal analysis |
| 5 | `feat(rust): port RNN VAD spectral features` | PFFFT-based features |
| 6 | `feat(rust): port RNN VAD pitch search` | Pitch detection |
| 7 | `feat(rust): port RNN VAD feature extraction pipeline` | 42-dim feature vector |
| 8 | `feat(rust): port RNN VAD model and wrapper` | RNN + VAD wrapper |
| 9 | `feat(rust): port AGC2 biquad filter` | IIR filter |
| 10 | `feat(rust): port AGC2 gain curves and gain applier` | Limiter curves + gain |
| 11 | `feat(rust): port AGC2 level estimators` | Fixed, noise, speech |
| 12 | `feat(rust): port AGC2 saturation protector` | Headroom management |
| 13 | `feat(rust): port AGC2 adaptive digital gain controller` | Main adaptive logic |
| 14 | `feat(rust): port AGC2 clipping predictor` | Clipping prediction |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RNN weight precision loss | Low | Critical | Use exact f32 literals from C++ source; verify bit patterns |
| Tanh approximation divergence | Medium | High | Test with exhaustive f32 range near breakpoints |
| GRU gate computation order | Medium | High | Match C++ matrix multiply order exactly |
| RNN VAD AVX2 accumulation order | Low | High | Compare dot product results against scalar at each step |

---

## Phase 4 Completion Checklist

- [ ] AGC2 fully ported: biquad, limiter, gain curves, level estimators, adaptive controller
- [ ] RNN VAD fully ported: weights, NN layers, feature extraction, pitch search, model
- [ ] AGC2 clipping predictor ported
- [ ] All 16 AGC2 core unit tests have Rust equivalents (excluding 2 deferred)
- [ ] All 15 RNN VAD unit tests have Rust equivalents
- [ ] SIMD via webrtc-simd dot_product verified
- [ ] C++ tests still pass
