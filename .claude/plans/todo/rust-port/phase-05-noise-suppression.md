# Phase 5: Noise Suppression

**Status:** Complete
**Completed:** February 2025
**Dependencies:** Phase 2 (Common Audio - FFT, signal processing)
**Outcome:** The `webrtc-ns` crate contains a working single-channel Rust noise suppressor with 70 unit tests, zero clippy warnings. Multi-channel and multi-band support deferred to Phase 7 (requires AudioBuffer).

---

## Overview

Port the Noise Suppression (NS) module. This is a self-contained module with moderate complexity. It uses Wiener filtering based on noise estimation, with a speech probability estimator guiding the suppression.

The NS module depends on:
- FFT (Ooura 256-point, from Phase 2)
- Basic math utilities (from Phase 2)
- No dependency on VAD or AGC

---

## Source Files to Port

All files in `webrtc/modules/audio_processing/ns/`:

| Source File | Description | Complexity | Lines |
|-------------|-------------|------------|-------|
| `ns_config.cc` | Configuration constants | Low | ~30 |
| `fast_math.cc` | Fast log/exp approximations | Medium | ~80 |
| `histograms.cc` | Histogram tracking | Medium | ~100 |
| `prior_signal_model.cc` | Prior signal model | Medium | ~60 |
| `prior_signal_model_estimator.cc` | Prior model estimation | Medium | ~120 |
| `signal_model.cc` | Signal model | Low | ~40 |
| `signal_model_estimator.cc` | Signal model estimation | Medium | ~200 |
| `noise_estimator.cc` | Noise PSD estimation | High | ~150 |
| `quantile_noise_estimator.cc` | Quantile-based noise estimation | High | ~200 |
| `speech_probability_estimator.cc` | Speech probability | High | ~150 |
| `suppression_params.cc` | Suppression parameters by level | Low | ~50 |
| `wiener_filter.cc` | Wiener filter computation | High | ~200 |
| `noise_suppressor.cc` | Main suppressor pipeline | High | ~400 |
| `noise_suppressor_init.cc` | Initialization | Low | ~50 |
| `ns_fft.cc` | FFT wrapper for NS | Medium | ~80 |
| `analyze_frame.cc` | Frame analysis (unused in build?) | Low | ~60 |
| `quantization_util.cc` | Quantization utilities (unused?) | Low | ~30 |
| `suppression_filter.cc` | Overlap-add suppression filter | High | ~150 |

---

## Tasks

### 5.1 NS Foundation

Port the building blocks in bottom-up order.

**Step 1: Config and math**
- `ns_config.cc` -> `config.rs` - Suppression level parameters
- `fast_math.cc` -> `fast_math.rs` - Fast approximations for log2, exp2
- `suppression_params.cc` -> `suppression_params.rs` - Level-dependent parameters

**Critical:** The fast math approximations (log2, exp2) use specific polynomial approximations or lookup tables. These MUST be ported exactly for the noise estimation to converge identically.

**Step 2: FFT wrapper**
- `ns_fft.cc` -> `ns_fft.rs` - Thin wrapper around Ooura 256-point FFT (from Phase 2)

**Step 3: Signal models**
- `histograms.cc` -> `histograms.rs`
- `signal_model.cc` -> `signal_model.rs`
- `prior_signal_model.cc` -> `prior_signal_model.rs`

**Verification:**
- [ ] Fast math outputs match C++ within 1e-6
- [ ] Config values match exactly

**Commits:**
1. `feat(rust): port NS config, fast math, and suppression params`
2. `feat(rust): port NS FFT wrapper and signal models`

---

### 5.2 Noise and Speech Estimation

Port the core estimation algorithms.

**Files:**
- `noise_estimator.cc` -> `noise_estimator.rs`
- `quantile_noise_estimator.cc` -> `quantile_noise_estimator.rs`
- `prior_signal_model_estimator.cc` -> `prior_signal_model_estimator.rs`
- `signal_model_estimator.cc` -> `signal_model_estimator.rs`
- `speech_probability_estimator.cc` -> `speech_probability_estimator.rs`

**Proptest strategy:** Feed spectral frames through both Rust and C++ estimators, compare:
- Noise PSD estimates
- Speech probability values
- Signal model parameters

**Verification:**
- [ ] Noise estimation converges to same values as C++ over 100+ frames
- [ ] Speech probability matches within 1e-5

**Commits:**
1. `feat(rust): port NS noise estimator (quantile-based)`
2. `feat(rust): port NS speech probability and signal model estimators`

---

### 5.3 Wiener Filter and Suppression

Port the core suppression logic.

**Files:**
- `wiener_filter.cc` -> `wiener_filter.rs` - Computes suppression gains from noise/speech estimates
- `suppression_filter.cc` -> `suppression_filter.rs` - Applies suppression via overlap-add

**The suppression pipeline:**
1. Frame -> FFT -> spectral analysis
2. Noise estimation -> speech probability
3. Wiener filter gain computation
4. Apply gains in frequency domain
5. IFFT -> overlap-add -> output frame

**Proptest:**
```rust
proptest! {
    #[test]
    fn wiener_filter_gains_match_cpp(
        noise_spectrum in proptest::collection::vec(0.0f32..1.0f32, 129..=129),
        signal_spectrum in proptest::collection::vec(0.0f32..10.0f32, 129..=129),
    ) {
        let rust_gains = rust_wiener_filter(&noise_spectrum, &signal_spectrum);
        let cpp_gains = cpp_wiener_filter(&noise_spectrum, &signal_spectrum);
        assert_f32_near(&rust_gains, &cpp_gains, 1e-6);
    }
}
```

**Verification:**
- [ ] Wiener filter gains match C++ for all suppression levels
- [ ] Suppression filter overlap-add produces correct output

**Commit:** `feat(rust): port NS Wiener filter and suppression filter`

---

### 5.4 Main Noise Suppressor

Port the top-level noise suppressor that ties everything together.

**Files:**
- `noise_suppressor.cc` -> `noise_suppressor.rs` - Main pipeline
- `noise_suppressor_init.cc` -> integrated into `noise_suppressor.rs`

**Public API:**
```rust
pub struct NoiseSuppressor {
    // Internal state
}

pub enum SuppressionLevel {
    Low,
    Moderate,
    High,
    VeryHigh,
}

impl NoiseSuppressor {
    pub fn new(sample_rate: usize, num_channels: usize, level: SuppressionLevel) -> Self;
    pub fn analyze(&mut self, audio: &AudioBuffer);
    pub fn process(&mut self, audio: &mut AudioBuffer);
}
```

**End-to-end proptest:**
```rust
proptest! {
    #[test]
    fn noise_suppressor_output_matches_cpp(
        level in prop_oneof![
            Just(SuppressionLevel::Low),
            Just(SuppressionLevel::Moderate),
            Just(SuppressionLevel::High),
            Just(SuppressionLevel::VeryHigh),
        ],
        frames in proptest::collection::vec(
            audio_frame_f32(48000), 10..50
        ),
    ) {
        // Process all frames through both implementations
        // Compare output frame-by-frame
    }
}
```

**Verification:**
- [ ] `noise_suppressor_unittest` matched
- [ ] End-to-end: process 100 frames at each suppression level, output matches C++
- [ ] Multi-channel processing matches
- [ ] Sample rate changes handled correctly (16k, 32k, 48k)

**Commit:** `feat(rust): port main NoiseSuppressor pipeline`

---

## Phase 5 Completion Checklist

- [ ] All NS source files ported to `webrtc-ns` crate
- [ ] Fast math approximations bit-exact with C++
- [ ] Noise estimation converges identically over multi-frame sequences
- [ ] Wiener filter gains match at all suppression levels
- [ ] End-to-end suppression output matches C++ for multi-frame audio
- [ ] `noise_suppressor_unittest` behaviors matched
- [ ] `cargo test -p webrtc-ns` passes
- [ ] Proptest: 1000+ multi-frame sequences match
- [ ] C++ tests still pass

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1 | `feat(rust): port NS config, fast math, and suppression params` | Foundation |
| 2 | `feat(rust): port NS FFT wrapper and signal models` | FFT, models |
| 3 | `feat(rust): port NS noise estimator (quantile-based)` | Noise estimation |
| 4 | `feat(rust): port NS speech probability and signal model estimators` | Speech/signal estimation |
| 5 | `feat(rust): port NS Wiener filter and suppression filter` | Core suppression |
| 6 | `feat(rust): port main NoiseSuppressor pipeline` | Integration |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fast math approximation divergence | Medium | High | Test exhaustively near polynomial breakpoints |
| Noise estimator state drift | Medium | High | Compare intermediate state after each frame |
| FFT precision affecting suppression | Low | Medium | Using same Ooura FFT ensures bit-exactness |
| Overlap-add boundary effects | Low | Medium | Test with consecutive frames to verify continuity |
