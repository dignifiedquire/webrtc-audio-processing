# Plan: Port WebRTC Audio Processing to Rust

## Overview

Create a modular, fully-tested Rust port of the WebRTC Audio Processing library that exposes a C-compatible API. The port maintains bit-exact compatibility with the original C++ implementation through property-based testing.

**Goals:**
- 1:1 port of all publicly exposed functionality
- Modular multi-crate architecture
- Full test coverage via property tests comparing Rust output against C++ reference
- All SIMD optimizations (SSE2, AVX2, NEON) preserved
- C-compatible API for drop-in replacement
- Incremental port with build always passing

**Non-Goals:**
- Rewriting tests in Rust (tests remain in C++)
- Changing algorithms or behavior
- Adding new features

## Current Codebase Analysis

### Public API Surface (`webrtc/api/audio/audio_processing.h`)

**Core Types:**
- `AudioProcessing` - Main interface (refcounted)
- `AudioProcessingBuilder` - Builder pattern for creating APM
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
   - Single C file FFT library
   - Supports SSE, Altivec, NEON
   - Used for 128+ point FFTs
   - **Rust strategy:** Use existing `pffft` crate or port

2. **RNNoise** (`webrtc/third_party/rnnoise/`)
   - Neural network weights (`rnn_vad_weights.cc/h`)
   - Activation functions (`rnn_activations.h`)
   - **Rust strategy:** Port as data tables, port activation functions

3. **Ooura FFT** (`webrtc/common_audio/third_party/ooura/`)
   - 128-point and 256-point FFT implementations
   - Has SSE2 and NEON variants
   - **Rust strategy:** Port directly or use `rustfft`

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
- **Current count:** 2458 tests passing

## Rust Crate Architecture

```
webrtc-audio-processing-rs/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── webrtc-apm/               # Main public crate (C API)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── ffi.rs            # C FFI exports
│   │   │   ├── config.rs         # Configuration types
│   │   │   └── processing.rs     # AudioProcessing wrapper
│   │   └── Cargo.toml
│   │
│   ├── webrtc-apm-sys/           # C++ bindings for testing
│   │   ├── src/lib.rs
│   │   ├── build.rs              # Links to C++ library
│   │   └── Cargo.toml
│   │
│   ├── webrtc-common-audio/      # DSP primitives
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── resampler/        # Push/sinc resamplers
│   │   │   ├── signal_processing/# SPL functions
│   │   │   ├── vad/              # Voice activity detection
│   │   │   ├── fir_filter/       # FIR filter implementations
│   │   │   ├── fft/              # FFT wrappers (pffft, ooura)
│   │   │   └── ring_buffer.rs
│   │   └── Cargo.toml
│   │
│   ├── webrtc-aec3/              # Echo Canceller 3
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── adaptive_fir_filter/
│   │   │   ├── matched_filter/
│   │   │   ├── echo_remover/
│   │   │   ├── suppression/
│   │   │   ├── delay/
│   │   │   └── ... (organized by subsystem)
│   │   └── Cargo.toml
│   │
│   ├── webrtc-aecm/              # Mobile Echo Control
│   │   └── ...
│   │
│   ├── webrtc-agc/               # AGC1 + AGC2
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── agc1/
│   │   │   ├── agc2/
│   │   │   └── rnn_vad/
│   │   └── Cargo.toml
│   │
│   ├── webrtc-ns/                # Noise Suppression
│   │   └── ...
│   │
│   ├── webrtc-vad/               # Voice Activity Detection
│   │   └── ...
│   │
│   ├── webrtc-simd/              # SIMD abstractions
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── sse2.rs
│   │   │   ├── avx2.rs
│   │   │   ├── neon.rs
│   │   │   └── fallback.rs
│   │   └── Cargo.toml
│   │
│   └── webrtc-apm-proptest/      # Property-based tests
│       ├── src/lib.rs
│       ├── tests/
│       │   ├── aec3_proptest.rs
│       │   ├── agc_proptest.rs
│       │   └── ...
│       └── Cargo.toml
│
├── tests/                        # C++ tests (unchanged)
│   └── ...
│
└── examples/
    └── run-offline-rs/           # Rust example app
```

### Crate Dependencies

```
webrtc-apm (main crate)
├── webrtc-common-audio
├── webrtc-aec3
│   ├── webrtc-common-audio
│   └── webrtc-simd
├── webrtc-aecm
│   └── webrtc-common-audio
├── webrtc-agc
│   ├── webrtc-common-audio
│   ├── webrtc-simd
│   └── webrtc-vad
├── webrtc-ns
│   └── webrtc-common-audio
└── webrtc-vad
    └── webrtc-common-audio
```

## Phased Implementation Plan

### Phase 1: Foundation Infrastructure (2-3 weeks)

**1.1 Project Setup**
- [ ] Create workspace structure with empty crates
- [ ] Set up CI (build + C++ test pass requirement)
- [ ] Configure Cargo features for SIMD (sse2, avx2, neon)
- [ ] Set up `webrtc-apm-sys` with bindgen to existing C++ library

**1.2 SIMD Abstraction Layer (`webrtc-simd`)**
- [ ] Define portable SIMD traits for common operations
- [ ] Implement SSE2 backend (x86/x86_64)
- [ ] Implement AVX2 backend (x86/x86_64)
- [ ] Implement NEON backend (ARM/ARM64)
- [ ] Implement scalar fallback
- [ ] Runtime CPU feature detection

**1.3 Property Test Framework (`webrtc-apm-proptest`)**
- [ ] Set up proptest/quickcheck infrastructure
- [ ] Create audio buffer generators
- [ ] Create config generators
- [ ] Implement comparison utilities (tolerance-based float comparison)
- [ ] Scaffold comparison tests calling C++ via FFI

### Phase 2: Common Audio Primitives (3-4 weeks)

**2.1 Ring Buffer**
- [ ] Port `ring_buffer.c` to Rust
- [ ] Proptest: Compare with C++ implementation

**2.2 Signal Processing Library**
- [ ] Port basic operations (`copy_set_operations`, `min_max_operations`)
- [ ] Port FFT-related (`complex_bit_reverse`, `complex_fft`, `real_fft`)
- [ ] Port filtering (`filter_ar`, `filter_ma`)
- [ ] Port resampling (`resample`, `resample_by_2`, `resample_fractional`)
- [ ] Port math operations (`spl_sqrt`, `division_operations`)
- [ ] Port cross-correlation (with NEON optimization)
- [ ] Proptest: Each function against C++ reference

**2.3 FIR Filter**
- [ ] Port scalar implementation (`fir_filter_c.cc`)
- [ ] Port SSE implementation
- [ ] Port AVX2 implementation
- [ ] Port NEON implementation
- [ ] Factory with runtime dispatch
- [ ] Proptest: Verify bitexact output

**2.4 Resampler**
- [ ] Port `Resampler` class
- [ ] Port `SincResampler` (scalar)
- [ ] Port `SincResampler` SIMD variants
- [ ] Port `PushResampler` / `PushSincResampler`
- [ ] Proptest: Input/output comparison

**2.5 FFT Wrappers**
- [ ] Port Ooura 128/256 FFT
- [ ] Integrate PFFFT (via crate or port)
- [ ] Port `pffft_wrapper.cc`
- [ ] Proptest: Forward/inverse FFT identity

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

### Phase 4: Automatic Gain Control (4-5 weeks)

**4.1 AGC1 Legacy (`webrtc-agc`)**
- [ ] Port `analog_agc.cc`
- [ ] Port `digital_agc.cc`
- [ ] Port `gain_control.h` interface
- [ ] Port loudness histogram
- [ ] Proptest: Gain computation comparison

**4.2 AGC2 Core**
- [ ] Port `limiter_db_gain_curve.cc`
- [ ] Port `interpolated_gain_curve.cc`
- [ ] Port `limiter.cc`
- [ ] Port `gain_applier.cc`
- [ ] Port `biquad_filter.cc`
- [ ] Proptest: Each component

**4.3 AGC2 Adaptive Digital**
- [ ] Port `fixed_digital_level_estimator.cc`
- [ ] Port `noise_level_estimator.cc`
- [ ] Port `speech_level_estimator.cc`
- [ ] Port `saturation_protector.cc`
- [ ] Port `adaptive_digital_gain_controller.cc`
- [ ] Proptest: Level estimation comparison

**4.4 AGC2 Input Volume Controller**
- [ ] Port `clipping_predictor.cc`
- [ ] Port `clipping_predictor_level_buffer.cc`
- [ ] Port `input_volume_controller.cc`
- [ ] Port `input_volume_stats_reporter.cc`
- [ ] Proptest: Clipping detection, volume recommendations

**4.5 RNN VAD**
- [ ] Port `rnn_vad_weights.cc` (weight tables)
- [ ] Port `rnn_activations.h` (tanh approximation, etc.)
- [ ] Port `rnn_fc.cc` / `rnn_gru.cc` (neural network layers)
- [ ] Port `rnn.cc` (RNN model)
- [ ] Port feature extraction (`spectral_features.cc`, `lp_residual.cc`, etc.)
- [ ] Port pitch search (`auto_correlation.cc`, `pitch_search.cc`)
- [ ] Port `vad_wrapper.cc`
- [ ] Proptest: Speech probability comparison

### Phase 5: Noise Suppression (2-3 weeks)

**5.1 NS Core (`webrtc-ns`)**
- [ ] Port `ns_fft.cc`
- [ ] Port `fast_math.cc`
- [ ] Port `histograms.cc`
- [ ] Port `prior_signal_model.cc` / `signal_model.cc`
- [ ] Port `noise_estimator.cc` / `quantile_noise_estimator.cc`
- [ ] Port `speech_probability_estimator.cc`
- [ ] Port `wiener_filter.cc`
- [ ] Port `noise_suppressor.cc`
- [ ] Proptest: Frame-by-frame suppression comparison

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

### Phase 7: Mobile Echo Control (1-2 weeks)

**7.1 AECM (`webrtc-aecm`)**
- [ ] Port `aecm_core.cc` (scalar)
- [ ] Port NEON optimizations
- [ ] Port `echo_control_mobile.cc`
- [ ] Proptest: AECM output comparison

### Phase 8: Audio Processing Integration (3-4 weeks)

**8.1 Audio Buffer & Utilities**
- [ ] Port `audio_buffer.cc`
- [ ] Port `AudioFrame` handling
- [ ] Port `audio_util.h` conversions
- [ ] Port `channel_buffer.cc`

**8.2 Component Wrappers**
- [ ] Port `high_pass_filter.cc`
- [ ] Port `gain_control_impl.cc`
- [ ] Port `gain_controller2.cc`
- [ ] Port `echo_control_mobile_impl.cc`
- [ ] Port `residual_echo_detector.cc`
- [ ] Port `splitting_filter.cc` / `three_band_filter_bank.cc`
- [ ] Port level adjustment components

**8.3 Main Implementation**
- [ ] Port `audio_processing_impl.cc`
- [ ] Port `audio_processing_builder_impl.cc`
- [ ] Proptest: Full ProcessStream comparison

### Phase 9: C API & Final Integration (2-3 weeks)

**9.1 C API Design**
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

**9.2 Implementation**
- [ ] Define C API in `crates/webrtc-apm/src/ffi.rs`
- [ ] Generate C headers with cbindgen
- [ ] Implement all C API functions
- [ ] Create pkg-config file

**9.3 Integration Testing**
- [ ] Modify C++ tests to optionally use Rust library via C API
- [ ] Verify all 2458 tests pass with Rust implementation
- [ ] Performance benchmarking vs C++ implementation

### Phase 10: Documentation & Release (1-2 weeks)

**10.1 Documentation**
- [ ] API documentation (rustdoc)
- [ ] C API documentation
- [ ] Migration guide from C++ library
- [ ] Architecture documentation

**10.2 Release Preparation**
- [ ] Version 0.1.0 release
- [ ] Publish to crates.io
- [ ] Create release binaries for major platforms

## Testing Strategy

### Property-Based Testing

Each Rust function will have a corresponding proptest that:
1. Generates random but valid inputs
2. Calls both Rust and C++ (via FFI) implementations
3. Compares outputs with appropriate tolerance

```rust
// Example proptest structure
#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use webrtc_apm_sys as cpp;

    proptest! {
        #[test]
        fn fir_filter_matches_cpp(
            coefficients in prop::collection::vec(any::<f32>(), 1..128),
            input in prop::collection::vec(any::<f32>(), 1..1024),
        ) {
            let rust_output = rust_fir_filter(&coefficients, &input);
            let cpp_output = unsafe { cpp::fir_filter(&coefficients, &input) };
            
            for (r, c) in rust_output.iter().zip(cpp_output.iter()) {
                prop_assert!((r - c).abs() < 1e-6);
            }
        }
    }
}
```

### Bitexactness Verification

For algorithms requiring exact output (e.g., integer processing):
- Compare byte-for-byte output
- Use existing C++ test reference data

### Continuous Integration

```yaml
# CI Pipeline
on: [push, pull_request]

jobs:
  build-cpp:
    # Ensure C++ library still builds
    
  test-cpp:
    # Run all C++ tests (must pass)
    
  build-rust:
    # Build Rust crates
    
  test-rust-proptests:
    # Run property tests
    
  test-rust-via-cpp:
    # Run C++ tests with Rust backend (after Phase 9)
```

## SIMD Implementation Strategy

### Approach

Use Rust's `std::arch` intrinsics directly (no external SIMD libraries) for maximum control and to match C++ behavior exactly.

```rust
// webrtc-simd crate structure
pub trait SimdOps {
    fn sqrt_vec(x: &mut [f32]);
    fn multiply_vec(x: &[f32], y: &[f32], z: &mut [f32]);
    fn accumulate_vec(x: &[f32], z: &mut [f32]);
    fn fir_filter(coeffs: &[f32], input: &[f32], output: &mut [f32]);
    // ...
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use std::arch::x86_64::*;
    // SSE2 implementations
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;
    // AVX2 implementations (with FMA)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    // NEON implementations
}

mod fallback {
    // Scalar implementations
}

// Runtime dispatch
pub fn get_simd_ops() -> &'static dyn SimdOps {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return &avx2::Avx2Ops;
        }
        if is_x86_feature_detected!("sse2") {
            return &sse2::Sse2Ops;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return &neon::NeonOps;
    }
    &fallback::ScalarOps
}
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

- [ ] All 2458+ C++ tests pass when using Rust implementation
- [ ] Property tests demonstrate equivalence for all components
- [ ] Performance within 10% of C++ implementation
- [ ] Builds on Linux x86_64, ARM64; macOS x86_64, ARM64; Windows x86_64
- [ ] All SIMD paths (SSE2, AVX2, NEON) functional and verified
- [ ] C API documented and usable from C/C++ projects
- [ ] Published to crates.io with complete documentation
