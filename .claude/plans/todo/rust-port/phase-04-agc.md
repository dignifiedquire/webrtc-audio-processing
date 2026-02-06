# Phase 4: Automatic Gain Control

**Status:** Not Started
**Estimated Duration:** 4-5 weeks
**Dependencies:** Phase 2 (Common Audio), Phase 3 (VAD - for RNN VAD dependency)
**Outcome:** The `webrtc-agc` crate contains working AGC1, AGC2, and RNN VAD implementations. All gain computations match the C++ reference within specified tolerance.

---

## Overview

Port the Automatic Gain Control modules. AGC is the second-largest module after AEC3 and has two generations:
- **AGC1** (legacy): Analog + digital gain control, used by older applications
- **AGC2** (modern): Adaptive digital gain with RNN-based VAD, input volume controller, limiter

AGC2 internally uses an RNN-based Voice Activity Detector (different from the core VAD in Phase 3). This RNN VAD uses neural network inference with SIMD-optimized vector math.

---

## Source Files to Port

### AGC1 Legacy (`webrtc/modules/audio_processing/agc/`)

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `agc.cc` | AGC interface | Low |
| `agc_manager_direct.cc` | Direct AGC manager | High |
| `loudness_histogram.cc` | Loudness tracking | Medium |
| `utility.cc` | AGC utilities | Low |
| `legacy/analog_agc.cc` | Analog gain control | High |
| `legacy/digital_agc.cc` | Digital compression | High |

### AGC2 Core (`webrtc/modules/audio_processing/agc2/`)

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `limiter_db_gain_curve.cc` | Limiter gain curve | Medium |
| `interpolated_gain_curve.cc` | Interpolated gain lookup | Medium |
| `compute_interpolated_gain_curve.cc` | Gain curve computation | Medium |
| `limiter.cc` | Audio limiter | Medium |
| `gain_applier.cc` | Gain application | Low |
| `biquad_filter.cc` | Biquad IIR filter | Low |
| `fixed_digital_level_estimator.cc` | Fixed digital level | Medium |
| `noise_level_estimator.cc` | Noise floor estimation | Medium |
| `speech_level_estimator.cc` | Speech level estimation | Medium |
| `speech_level_estimator_impl.cc` | Implementation | Medium |
| `speech_level_estimator_experimental_impl.cc` | Experimental impl | Medium |
| `saturation_protector.cc` | Saturation protection | Medium |
| `saturation_protector_buffer.cc` | Saturation buffer | Low |
| `speech_probability_buffer.cc` | Speech probability tracking | Low |
| `adaptive_digital_gain_controller.cc` | Adaptive digital AGC | High |
| `agc2_testing_common.cc` | Test utilities | Low |
| `vector_float_frame.cc` | Float frame container | Low |
| `cpu_features.cc` | CPU detection for SIMD | Low |

### AGC2 Input Volume Controller

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `clipping_predictor.cc` | Clipping prediction | High |
| `clipping_predictor_level_buffer.cc` | Level buffer | Low |
| `input_volume_controller.cc` | Volume control | High |
| `input_volume_stats_reporter.cc` | Stats reporting | Low |

### RNN VAD (`webrtc/modules/audio_processing/agc2/rnn_vad/`)

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `rnn_vad_weights.cc` | Neural network weights (data) | Low (large) |
| `rnn_activations.h` | Activation functions (tanh approx) | Medium |
| `rnn_fc.cc` | Fully-connected layer | Medium |
| `rnn_gru.cc` | GRU recurrent layer | High |
| `rnn.cc` | Full RNN model | High |
| `features_extraction.cc` | Feature extraction pipeline | High |
| `spectral_features.cc` | Spectral feature computation | Medium |
| `spectral_features_internal.cc` | Spectral internals | Medium |
| `lp_residual.cc` | LP residual computation | Medium |
| `auto_correlation.cc` | Auto-correlation for pitch | Medium |
| `pitch_search.cc` | Pitch period search | High |
| `pitch_search_internal.cc` | Pitch search helpers | Medium |
| `vector_math_avx2.cc` | AVX2 vector operations | Medium |
| `vad_wrapper.cc` | VAD wrapper for AGC2 | Medium |

---

## Tasks

### 4.1 AGC2 Building Blocks (Bottom-Up)

Port the lowest-level AGC2 components first.

**Step 1: Biquad Filter**
- `biquad_filter.cc` -> `agc2/biquad_filter.rs`
- Simple IIR filter, straightforward port

**Step 2: Gain Curves**
- `limiter_db_gain_curve.cc` -> `agc2/limiter_db_gain_curve.rs`
- `interpolated_gain_curve.cc` -> `agc2/interpolated_gain_curve.rs`
- `compute_interpolated_gain_curve.cc` -> `agc2/compute_gain_curve.rs`
- These are pure math functions with lookup tables

**Step 3: Gain Application and Limiter**
- `gain_applier.cc` -> `agc2/gain_applier.rs`
- `limiter.cc` -> `agc2/limiter.rs`

**Proptest per component:**
```rust
proptest! {
    #[test]
    fn limiter_gain_curve_matches_cpp(
        input_level_db in -100.0f32..0.0f32,
    ) {
        let rust_gain = rust_limiter_gain(input_level_db);
        let cpp_gain = cpp_limiter_gain(input_level_db);
        prop_assert!((rust_gain - cpp_gain).abs() < 1e-6);
    }
}
```

**Verification:**
- [ ] `biquad_filter_unittest` matched
- [ ] `limiter_db_gain_curve_unittest` matched
- [ ] `interpolated_gain_curve_unittest` matched
- [ ] `limiter_unittest` matched
- [ ] `gain_applier_unittest` matched

**Commits:**
1. `feat(rust): port AGC2 biquad filter`
2. `feat(rust): port AGC2 gain curves and limiter`
3. `feat(rust): port AGC2 gain applier`

---

### 4.2 AGC2 Level Estimators

Port the level estimation components.

**Files:**
- `fixed_digital_level_estimator.cc` -> `agc2/fixed_digital_level_estimator.rs`
- `noise_level_estimator.cc` -> `agc2/noise_level_estimator.rs`
- `speech_level_estimator.cc` -> `agc2/speech_level_estimator.rs`
- `speech_level_estimator_impl.cc` -> `agc2/speech_level_estimator_impl.rs`
- `speech_level_estimator_experimental_impl.cc` -> `agc2/speech_level_estimator_experimental.rs`
- `saturation_protector.cc` -> `agc2/saturation_protector.rs`
- `saturation_protector_buffer.cc` -> `agc2/saturation_protector_buffer.rs`
- `speech_probability_buffer.cc` -> `agc2/speech_probability_buffer.rs`

**Proptest:** Feed multi-frame audio sequences and compare level estimates.

**Verification:**
- [ ] `fixed_digital_level_estimator_unittest` matched
- [ ] `noise_level_estimator_unittest` matched
- [ ] `speech_level_estimator_unittest` matched
- [ ] `saturation_protector_unittest` matched
- [ ] `saturation_protector_buffer_unittest` matched
- [ ] `speech_probability_buffer_unittest` matched

**Commits:**
1. `feat(rust): port AGC2 level estimators (fixed digital, noise)`
2. `feat(rust): port AGC2 speech level estimator and saturation protector`

---

### 4.3 RNN VAD

Port the neural network-based voice activity detector. This is the most complex sub-component.

**Port order:**

**Step 1: Weight tables and activation functions**
- `rnn_vad_weights.cc` - Large const arrays of floats (just data)
- `rnn_activations.h` - Custom tanh approximation, sigmoid, ReLU

The tanh approximation is critical for bit-exactness:
```cpp
// Piecewise linear approximation of tanh
inline float TanhApproximation(float x) { ... }
```

**Step 2: Neural network layers**
- `rnn_fc.cc` - Fully-connected layer (matrix multiply + bias + activation)
- `rnn_gru.cc` - GRU layer (gated recurrent unit)
- `vector_math_avx2.cc` - AVX2-optimized vector dot product for NN inference

**Step 3: Feature extraction**
- `lp_residual.cc` - Linear prediction residual
- `auto_correlation.cc` - Auto-correlation for pitch detection
- `spectral_features_internal.cc` - Spectral feature helpers
- `spectral_features.cc` - Full spectral feature pipeline
- `pitch_search_internal.cc` - Pitch period search helpers
- `pitch_search.cc` - Full pitch search
- `features_extraction.cc` - Complete feature extraction

**Step 4: RNN model and wrapper**
- `rnn.cc` - Full RNN model (FC -> GRU -> FC -> output)
- `vad_wrapper.cc` - VAD wrapper that combines feature extraction + RNN

**Destination:**
```
webrtc-agc/src/
  rnn_vad/
    mod.rs                # Re-exports
    weights.rs            # Weight tables (generated from .cc)
    activations.rs        # Tanh approx, sigmoid, ReLU
    fc_layer.rs           # Fully-connected layer
    gru_layer.rs          # GRU layer
    vector_math.rs        # Scalar vector ops
    vector_math_avx2.rs   # AVX2-optimized ops
    lp_residual.rs        # LP residual
    auto_correlation.rs   # Auto-correlation
    spectral_features.rs  # Spectral features
    pitch_search.rs       # Pitch search
    features.rs           # Feature extraction pipeline
    rnn.rs                # Full RNN model
    vad.rs                # VAD wrapper
```

**Critical correctness concerns:**
- The tanh approximation MUST produce identical output to C++. Test with edge cases.
- Matrix multiplication order and accumulation order must match exactly for SIMD.
- Weight tables must be copied exactly (include all decimal places).
- The GRU update gate / reset gate computation order matters.

**Proptest:**
```rust
proptest! {
    #[test]
    fn rnn_vad_probability_matches_cpp(
        audio_frames in proptest::collection::vec(
            audio_frame_f32(24000),  // RNN VAD operates at 24kHz
            1..50
        ),
    ) {
        // Process frames through both implementations
        for frame in &audio_frames {
            let rust_prob = rust_rnn_vad.process(frame);
            let cpp_prob = cpp_rnn_vad.process(frame);
            prop_assert!((rust_prob - cpp_prob).abs() < 1e-5);
        }
    }
}
```

**Verification:**
- [ ] `auto_correlation_unittest` matched
- [ ] `features_extraction_unittest` matched
- [ ] `lp_residual_unittest` matched
- [ ] `pitch_search_internal_unittest` matched
- [ ] `pitch_search_unittest` matched
- [ ] `rnn_fc_unittest` matched
- [ ] `rnn_gru_unittest` matched
- [ ] `rnn_unittest` matched
- [ ] `rnn_vad_unittest` matched
- [ ] `ring_buffer_unittest` matched (RNN VAD has its own ring buffer)
- [ ] `sequence_buffer_unittest` matched
- [ ] `spectral_features_internal_unittest` matched
- [ ] `spectral_features_unittest` matched
- [ ] `symmetric_matrix_buffer_unittest` matched
- [ ] `vector_math_unittest` matched
- [ ] AVX2 vector math produces same output as scalar

**Commits:**
1. `feat(rust): port RNN VAD weight tables and activation functions`
2. `feat(rust): port RNN VAD neural network layers (FC, GRU)`
3. `feat(rust): port RNN VAD vector math with AVX2 optimization`
4. `feat(rust): port RNN VAD feature extraction pipeline`
5. `feat(rust): port RNN VAD pitch search`
6. `feat(rust): port RNN VAD model and wrapper`

---

### 4.4 AGC2 Adaptive Digital Controller

Port the main adaptive digital gain controller that ties together level estimators and RNN VAD.

**File:** `adaptive_digital_gain_controller.cc` -> `agc2/adaptive_digital.rs`

This is the central AGC2 component that:
1. Gets speech probability from RNN VAD
2. Estimates speech level and noise level
3. Computes adaptive gain
4. Applies gain with saturation protection

**Verification:**
- [ ] `adaptive_digital_gain_controller_unittest` matched
- [ ] Multi-frame gain tracking matches C++

**Commit:** `feat(rust): port AGC2 adaptive digital gain controller`

---

### 4.5 AGC2 Input Volume Controller

Port the clipping prediction and input volume control.

**Files:**
- `clipping_predictor.cc` -> `agc2/clipping_predictor.rs`
- `clipping_predictor_level_buffer.cc` -> `agc2/clipping_predictor_buffer.rs`
- `input_volume_controller.cc` -> `agc2/input_volume_controller.rs`
- `input_volume_stats_reporter.cc` -> `agc2/input_volume_stats.rs`

**Verification:**
- [ ] `clipping_predictor_unittest` matched
- [ ] `clipping_predictor_level_buffer_unittest` matched
- [ ] `input_volume_controller_unittest` matched
- [ ] `input_volume_stats_reporter_unittest` matched

**Commits:**
1. `feat(rust): port AGC2 clipping predictor`
2. `feat(rust): port AGC2 input volume controller`

---

### 4.6 AGC2 Top-Level

Port the top-level gain_controller2 and VAD wrapper.

**Files:**
- `gain_controller2.cc` -> `agc2/mod.rs` (or `gain_controller2.rs`)
- `vad_wrapper.cc` -> `agc2/vad_wrapper.rs`

**Verification:**
- [ ] `gain_controller2_unittest` matched
- [ ] `vad_wrapper_unittest` matched
- [ ] `agc2_testing_common_unittest` matched

**Commit:** `feat(rust): port AGC2 top-level gain controller`

---

### 4.7 AGC1 Legacy

Port the legacy AGC. Lower priority but needed for full compatibility.

**Files:**
- `agc.cc` -> `agc1/agc.rs`
- `agc_manager_direct.cc` -> `agc1/agc_manager.rs`
- `loudness_histogram.cc` -> `agc1/loudness_histogram.rs`
- `utility.cc` -> `agc1/utility.rs`
- `legacy/analog_agc.cc` -> `agc1/analog_agc.rs`
- `legacy/digital_agc.cc` -> `agc1/digital_agc.rs`

**Destination:**
```
webrtc-agc/src/
  agc1/
    mod.rs
    agc.rs
    agc_manager.rs
    loudness_histogram.rs
    utility.rs
    analog_agc.rs
    digital_agc.rs
```

**Verification:**
- [ ] `agc_manager_direct_unittest` matched
- [ ] `loudness_histogram_unittest` matched

**Commits:**
1. `feat(rust): port AGC1 legacy digital and analog gain control`
2. `feat(rust): port AGC1 manager and loudness histogram`

---

## Phase 4 Completion Checklist

- [ ] AGC2 fully ported: biquad, limiter, gain curves, level estimators, adaptive controller
- [ ] RNN VAD fully ported: weights, NN layers, feature extraction, pitch search, model
- [ ] AGC2 Input Volume Controller ported: clipping predictor, volume controller
- [ ] AGC1 legacy fully ported: analog, digital, manager, histogram
- [ ] All 33 AGC2 unit tests have Rust equivalents
- [ ] All 2 AGC1 unit tests have Rust equivalents
- [ ] All 15 RNN VAD unit tests have Rust equivalents
- [ ] AVX2 vector math for RNN VAD is ported and verified
- [ ] Proptest: RNN VAD speech probability matches within tolerance
- [ ] Proptest: AGC2 gain computation matches within tolerance
- [ ] C++ tests still pass

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1-3 | AGC2 building blocks | biquad, gain curves, limiter, gain applier |
| 4-5 | AGC2 level estimators | fixed digital, noise, speech, saturation |
| 6-11 | RNN VAD | weights, NN layers, features, pitch, model |
| 12 | AGC2 adaptive controller | adaptive_digital_gain_controller |
| 13-14 | AGC2 input volume | clipping predictor, volume controller |
| 15 | AGC2 top-level | gain_controller2 |
| 16-17 | AGC1 legacy | analog/digital AGC, manager |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RNN weight precision loss | Low | Critical | Use exact f32 literals from C++ source; verify bit patterns |
| Tanh approximation divergence | Medium | High | Test with exhaustive f32 range near breakpoints |
| GRU gate computation order | Medium | High | Match C++ matrix multiply order exactly |
| AGC1 integer overflow behavior | Medium | Medium | Use wrapping arithmetic; test with extreme gain values |
| RNN VAD AVX2 accumulation order | Low | High | Compare dot product results against scalar at each step |
