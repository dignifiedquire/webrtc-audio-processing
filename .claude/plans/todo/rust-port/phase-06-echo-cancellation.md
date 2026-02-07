# Phase 6: Echo Cancellation (AEC3)

**Status:** Complete (172 tests, 65 files, ~15,800 lines; SIMD for MatchedFilter/AdaptiveFirFilter deferred to SIMD phase)
**Estimated Duration:** 6-8 weeks
**Dependencies:** Phase 2 (Common Audio - FFT, FIR filter, SIMD)
**Outcome:** The `webrtc-aec3` crate contains a complete Rust implementation of the Echo Canceller 3 that produces identical echo-cancelled output to the C++ reference. All SIMD optimizations (SSE2, AVX2, NEON) are ported.

---

## Overview

Port the Echo Canceller 3 (AEC3) - the largest and most complex module (124 source files, ~750 tests). AEC3 uses frequency-domain adaptive filtering to estimate and cancel acoustic echo.

This phase is broken into 7 sub-phases that build up the AEC3 pipeline bottom-to-top. Each sub-phase produces a testable, committable unit.

**AEC3 Signal Flow:**
```
Render (far-end) audio
  -> RenderDelayBuffer (buffering + alignment)
  -> MatchedFilter (delay estimation)
  -> RenderDelayController (delay control)
  
Capture (near-end) audio
  -> FrameBlocker (frame -> block conversion)
  -> Subtractor (adaptive filter subtraction)
    -> AdaptiveFirFilter (echo estimation)
    -> FilterAnalyzer (filter quality analysis)
  -> EchoRemover (echo removal decision)
    -> AecState (echo path state tracking)
    -> SuppressionGain (gain computation)
    -> SuppressionFilter (gain application)
    -> ComfortNoiseGenerator (comfort noise)
  -> BlockFramer (block -> frame conversion)
  -> EchoCanceller3 (top-level orchestration)
```

---

## Source Files to Port (by sub-phase)

### 6.1 Foundation Constants and FFT

| Source File | Description |
|-------------|-------------|
| `aec3_common.cc` | Constants, enums, helper functions |
| `aec3_fft.cc` | FFT wrapper using Ooura 128-point |
| `fft_data.h` | Complex FFT data (FftData struct) |
| `fft_data_avx2.cc` | AVX2-optimized FFT data operations |
| `block.h` | Audio block definition |
| `block_buffer.cc` | Block ring buffer |
| `spectrum_buffer.cc` | Spectral data ring buffer |
| `fft_buffer.cc` | FFT data ring buffer |

### 6.2 Framing (Block <-> Frame)

| Source File | Description |
|-------------|-------------|
| `block_delay_buffer.cc` | Delay compensation buffer |
| `frame_blocker.cc` | Frame to block conversion |
| `block_framer.cc` | Block to frame conversion |
| `decimator.cc` | Signal decimation for delay estimation |

### 6.3 Delay Estimation

| Source File | Description |
|-------------|-------------|
| `downsampled_render_buffer.cc` | Downsampled render storage |
| `matched_filter.cc` | Matched filter delay estimator (SSE2/NEON inline) |
| `matched_filter_avx2.cc` | AVX2 matched filter |
| `matched_filter_lag_aggregator.cc` | Lag aggregation |
| `echo_path_delay_estimator.cc` | Echo path delay estimation |
| `render_delay_controller.cc` | Delay control logic |
| `render_delay_controller_metrics.cc` | Delay metrics |
| `render_delay_buffer.cc` | Render buffer with delay management |
| `render_buffer.cc` | Render buffer interface |

### 6.4 Adaptive Filter

| Source File | Description |
|-------------|-------------|
| `adaptive_fir_filter.cc` | Frequency-domain adaptive filter (SSE2/NEON inline) |
| `adaptive_fir_filter_avx2.cc` | AVX2 filter adaptation |
| `adaptive_fir_filter_erl.cc` | Echo return loss computation |
| `adaptive_fir_filter_erl_avx2.cc` | AVX2 ERL |
| `coarse_filter_update_gain.cc` | Coarse filter step size |
| `refined_filter_update_gain.cc` | Refined filter step size |
| `filter_analyzer.cc` | Filter quality analysis |
| `subtractor.cc` | Echo subtraction (coarse + refined filters) |
| `subtractor_output.cc` | Subtractor output processing |
| `subtractor_output_analyzer.cc` | Output analysis |

### 6.5 Echo Estimation

| Source File | Description |
|-------------|-------------|
| `erl_estimator.cc` | Echo Return Loss estimation |
| `erle_estimator.cc` | Echo Return Loss Enhancement estimation |
| `fullband_erle_estimator.cc` | Fullband ERLE |
| `subband_erle_estimator.cc` | Subband ERLE |
| `signal_dependent_erle_estimator.cc` | Signal-dependent ERLE |
| `residual_echo_estimator.cc` | Residual echo power estimation |
| `reverb_model.cc` | Reverb tail model |
| `reverb_model_estimator.cc` | Reverb estimation |
| `reverb_decay_estimator.cc` | Reverb decay estimation |
| `reverb_frequency_response.cc` | Reverb frequency response |

### 6.6 Suppression

| Source File | Description |
|-------------|-------------|
| `render_signal_analyzer.cc` | Render signal analysis |
| `comfort_noise_generator.cc` | Comfort noise generation |
| `suppression_gain.cc` | Echo suppression gain computation |
| `suppression_filter.cc` | Apply suppression in frequency domain |
| `dominant_nearend_detector.cc` | Near-end dominance detection |
| `subband_nearend_detector.cc` | Subband near-end detection |
| `vector_math.h` | Vector ops (SSE2/NEON inline) |
| `vector_math_avx2.cc` | AVX2 vector ops |

### 6.7 State, Integration, and Top-Level

| Source File | Description |
|-------------|-------------|
| `aec_state.cc` | Echo canceller state tracking |
| `echo_audibility.cc` | Echo audibility estimation |
| `echo_path_variability.cc` | Echo path change detection |
| `stationarity_estimator.cc` | Signal stationarity |
| `echo_remover.cc` | Echo removal pipeline |
| `echo_remover_metrics.cc` | Echo removal metrics |
| `block_processor.cc` | Block-level processing pipeline |
| `block_processor_metrics.cc` | Block processor metrics |
| `echo_canceller3.cc` | Top-level AEC3 |
| `config_selector.cc` | AEC3 config selection |
| `alignment_mixer.cc` | Multi-channel alignment |
| `api_call_jitter_metrics.cc` | API call jitter tracking |
| `clockdrift_detector.cc` | Clock drift detection |
| `multi_channel_content_detector.cc` | Multi-channel detection |
| `moving_average.cc` | Moving average utility |
| `transparent_mode.cc` | Transparent (passthrough) mode |

---

## Tasks

### 6.1 AEC3 Foundation

**Destination:**
```
webrtc-aec3/src/
  common.rs          # Constants, enums
  fft.rs             # AEC3 FFT wrapper
  fft_data.rs        # FftData struct + AVX2 ops
  block.rs           # Block type
  block_buffer.rs    # Block ring buffer
  spectrum_buffer.rs # Spectral buffer
  fft_buffer.rs      # FFT buffer
```

**Key types:**
```rust
/// 128-point complex FFT data
pub struct FftData {
    pub re: [f32; kFftLengthBy2Plus1],  // 65 bins
    pub im: [f32; kFftLengthBy2Plus1],
}

impl FftData {
    pub fn spectrum_average_energy(&self) -> f32;
    pub fn copy_to_packed_array(&self, packed: &mut [f32]);
    // AVX2-optimized variants selected at runtime
}
```

**Verification:**
- [ ] `aec3_fft_unittest` matched
- [ ] `fft_data_unittest` matched
- [ ] `moving_average_unittest` matched

**Commits:**
1. `feat(rust): port AEC3 constants, FFT wrapper, and FftData`
2. `feat(rust): port AEC3 block/spectrum/FFT buffers`

---

### 6.2 AEC3 Framing

**Destination:**
```
webrtc-aec3/src/
  framing/
    mod.rs
    block_delay_buffer.rs
    frame_blocker.rs
    block_framer.rs
    decimator.rs
```

**Verification:**
- [ ] `block_delay_buffer_unittest` matched
- [ ] `frame_blocker_unittest` matched
- [ ] `block_framer_unittest` matched (note: `FRIEND_TEST` used - ensure test is in `webrtc` namespace)
- [ ] `decimator_unittest` matched

**Commit:** `feat(rust): port AEC3 frame blocking and deblocking`

---

### 6.3 AEC3 Delay Estimation

**The matched filter is one of the most SIMD-critical paths.** It has SSE2, AVX2, and NEON optimizations.

**Destination:**
```
webrtc-aec3/src/
  delay/
    mod.rs
    downsampled_render_buffer.rs
    matched_filter.rs        # With inline SSE2/NEON dispatch
    matched_filter_avx2.rs   # AVX2 variant
    matched_filter_lag_aggregator.rs
    echo_path_delay_estimator.rs
    render_delay_controller.rs
    render_delay_controller_metrics.rs
    render_delay_buffer.rs
    render_buffer.rs
```

**SIMD porting for matched filter:**
The C++ code has SIMD code inline in `matched_filter.cc`:
```cpp
#if defined(WEBRTC_HAS_NEON)
  // NEON implementation
#elif defined(WEBRTC_ARCH_X86_FAMILY)
  // SSE2 implementation
#else
  // Scalar implementation
#endif
```

In Rust, use the same pattern with `cfg`:
```rust
#[cfg(target_arch = "x86_64")]
fn matched_filter_core_sse2(...) { /* SSE2 intrinsics */ }

#[cfg(target_arch = "aarch64")]
fn matched_filter_core_neon(...) { /* NEON intrinsics */ }

fn matched_filter_core_scalar(...) { /* Scalar fallback */ }
```

Plus the separate AVX2 file with `#[target_feature(enable = "avx2,fma")]`.

**Proptest:** Generate render + capture audio, run delay estimation, compare detected delay.

**Verification:**
- [ ] `matched_filter_unittest` matched
- [ ] `matched_filter_lag_aggregator_unittest` matched
- [ ] `echo_path_delay_estimator_unittest` matched
- [ ] `render_delay_controller_unittest` matched
- [ ] `render_delay_controller_metrics_unittest` matched
- [ ] `render_delay_buffer_unittest` matched
- [ ] `render_buffer_unittest` matched
- [ ] SIMD and scalar produce same delay estimates

**Commits:**
1. `feat(rust): port AEC3 matched filter with SSE2/AVX2/NEON`
2. `feat(rust): port AEC3 delay estimation pipeline`
3. `feat(rust): port AEC3 render delay buffer and controller`

---

### 6.4 AEC3 Adaptive Filter

**The most computationally intensive part of AEC3.** The adaptive FIR filter updates are heavily SIMD-optimized.

**Destination:**
```
webrtc-aec3/src/
  filter/
    mod.rs
    adaptive_fir_filter.rs       # With inline SSE2/NEON
    adaptive_fir_filter_avx2.rs  # AVX2 variant
    adaptive_fir_filter_erl.rs   # ERL computation
    adaptive_fir_filter_erl_avx2.rs
    coarse_filter_update_gain.rs
    refined_filter_update_gain.rs
    filter_analyzer.rs
    subtractor.rs
    subtractor_output.rs
    subtractor_output_analyzer.rs
```

**SIMD functions to port:**
- `AdaptPartitions()` - SSE2, AVX2, NEON variants
- `ScalePartitions()` - SSE2, AVX2, NEON variants
- `ComputeErl()` - AVX2 variant

**Proptest:**
```rust
proptest! {
    #[test]
    fn adaptive_filter_update_matches_cpp(
        render_fft in fft_data_strategy(),
        error_fft in fft_data_strategy(),
        step_size in 0.0f32..1.0f32,
    ) {
        // Run one filter update step through both
        // Compare filter coefficients after update
    }
}
```

**Verification:**
- [ ] `adaptive_fir_filter_unittest` matched (most complex AEC3 test)
- [ ] `adaptive_fir_filter_erl_unittest` matched
- [ ] `coarse_filter_update_gain_unittest` matched
- [ ] `refined_filter_update_gain_unittest` matched
- [ ] `filter_analyzer_unittest` matched
- [ ] `subtractor_unittest` matched
- [ ] SIMD variants produce same filter coefficients as scalar

**Commits:**
1. `feat(rust): port AEC3 adaptive FIR filter (scalar)`
2. `feat(rust): port AEC3 adaptive FIR filter SIMD (SSE2, AVX2, NEON)`
3. `feat(rust): port AEC3 filter update gains and analyzer`
4. `feat(rust): port AEC3 subtractor`

---

### 6.5 AEC3 Echo Estimation

**Destination:**
```
webrtc-aec3/src/
  estimation/
    mod.rs
    erl_estimator.rs
    erle_estimator.rs
    fullband_erle_estimator.rs
    subband_erle_estimator.rs
    signal_dependent_erle_estimator.rs
    residual_echo_estimator.rs
    reverb_model.rs
    reverb_model_estimator.rs
    reverb_decay_estimator.rs
    reverb_frequency_response.rs
```

**Verification:**
- [ ] `erl_estimator_unittest` matched
- [ ] `erle_estimator_unittest` matched
- [ ] `signal_dependent_erle_estimator_unittest` matched
- [ ] `residual_echo_estimator_unittest` matched
- [ ] `reverb_model_estimator_unittest` matched

**Commits:**
1. `feat(rust): port AEC3 ERL/ERLE estimators`
2. `feat(rust): port AEC3 residual echo and reverb estimators`

---

### 6.6 AEC3 Suppression

**Destination:**
```
webrtc-aec3/src/
  suppression/
    mod.rs
    render_signal_analyzer.rs
    comfort_noise_generator.rs
    suppression_gain.rs
    suppression_filter.rs
    dominant_nearend_detector.rs
    subband_nearend_detector.rs
    vector_math.rs          # SSE2/NEON inline
    vector_math_avx2.rs     # AVX2 variant
```

**Verification:**
- [ ] `render_signal_analyzer_unittest` matched
- [ ] `comfort_noise_generator_unittest` matched
- [ ] `suppression_gain_unittest` matched
- [ ] `suppression_filter_unittest` matched
- [ ] `vector_math_unittest` matched

**Commits:**
1. `feat(rust): port AEC3 vector math with SSE2/AVX2/NEON`
2. `feat(rust): port AEC3 suppression gain and filter`
3. `feat(rust): port AEC3 comfort noise and nearend detectors`

---

### 6.7 AEC3 State and Top-Level Integration

**Destination:**
```
webrtc-aec3/src/
  state/
    mod.rs
    aec_state.rs
    echo_audibility.rs
    echo_path_variability.rs
    stationarity_estimator.rs
  
  echo_remover.rs
  echo_remover_metrics.rs
  block_processor.rs
  block_processor_metrics.rs
  echo_canceller3.rs
  config_selector.rs
  alignment_mixer.rs
  api_call_jitter_metrics.rs
  clockdrift_detector.rs
  multi_channel_content_detector.rs
  transparent_mode.rs
```

**End-to-end proptest:**
```rust
proptest! {
    #[test]
    fn aec3_full_pipeline_matches_cpp(
        render_frames in proptest::collection::vec(audio_frame_f32(48000), 50..200),
        capture_frames in proptest::collection::vec(audio_frame_f32(48000), 50..200),
    ) {
        // Feed render frames, then capture frames through both implementations
        // Compare cancelled output
    }
}
```

**Verification:**
- [ ] `aec_state_unittest` matched
- [ ] `echo_path_variability_unittest` matched
- [ ] `clockdrift_detector_unittest` matched
- [ ] `multi_channel_content_detector_unittest` matched
- [ ] `alignment_mixer_unittest` matched
- [ ] `api_call_jitter_metrics_unittest` matched
- [ ] `echo_remover_unittest` matched
- [ ] `echo_remover_metrics_unittest` matched
- [ ] `block_processor_unittest` matched
- [ ] `block_processor_metrics_unittest` matched
- [ ] `echo_canceller3_unittest` matched
- [ ] `config_selector_unittest` matched
- [ ] Full AEC3 pipeline produces identical output to C++

**Commits:**
1. `feat(rust): port AEC3 state tracking (aec_state, echo audibility, variability)`
2. `feat(rust): port AEC3 echo remover pipeline`
3. `feat(rust): port AEC3 block processor`
4. `feat(rust): port AEC3 top-level echo canceller and config selector`
5. `feat(rust): port AEC3 multi-channel and utility components`

---

## Phase 6 Completion Checklist

- [ ] All 65+ AEC3 source files ported to `webrtc-aec3` crate
- [ ] All SIMD paths ported: SSE2, AVX2, NEON for matched filter, adaptive filter, vector math, FFT data
- [ ] All 42 AEC3 unit tests have Rust equivalents
- [ ] End-to-end AEC3 produces identical output to C++ for multi-frame sequences
- [ ] Delay estimation matches (within 1 sample)
- [ ] Adaptive filter coefficients match (within float tolerance)
- [ ] Suppression gains match (within float tolerance)
- [ ] `cargo test -p webrtc-aec3` passes
- [ ] C++ tests still pass

## Commit Summary

| Sub-phase | Commits | Scope |
|-----------|---------|-------|
| 6.1 Foundation | 2 | Constants, FFT, buffers |
| 6.2 Framing | 1 | Frame blocker/framer, decimator |
| 6.3 Delay | 3 | Matched filter, delay estimation, render buffer |
| 6.4 Filter | 4 | Adaptive filter, subtractor (with SIMD) |
| 6.5 Estimation | 2 | ERL/ERLE, reverb models |
| 6.6 Suppression | 3 | Vector math, suppression gain/filter, comfort noise |
| 6.7 Integration | 5 | State, echo remover, block processor, top-level |
| **Total** | **20** | |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Adaptive filter SIMD divergence | High | Critical | Compare filter taps after every update step, not just final output |
| Delay estimation off by 1 | Medium | High | Test with synthetic echo at known delays |
| Suppression gain computation precision | Medium | High | Compare gain masks in frequency domain |
| State machine divergence over time | Medium | High | Run 1000+ frame sequences; snapshot state at checkpoints |
| Memory layout differences (FftData) | Low | Medium | Use `#[repr(C)]` where comparing against C++ via FFI |
| Multi-channel path complexity | Medium | Medium | Test single-channel first, then extend |
