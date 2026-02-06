# Phase 8: Audio Processing Integration

**Status:** Not Started
**Estimated Duration:** 3-4 weeks
**Dependencies:** Phases 2-7 (all component crates)
**Outcome:** The `webrtc-apm` crate ties all components together into a complete `AudioProcessing` implementation. A Rust `AudioProcessing` can process audio identically to the C++ version through `ProcessStream` and `ProcessReverseStream`.

---

## Overview

Port the top-level `audio_processing_impl.cc` and all the glue code that connects individual components (AEC3, NS, AGC, VAD) into a unified processing pipeline. This phase does not expose a C API (that's Phase 9) - it creates the Rust-native API.

**Processing pipeline (capture path):**
```
Input (int16 or float)
  -> AudioBuffer creation
  -> CaptureLevelsAdjuster (pre-gain)
  -> HighPassFilter
  -> EchoCanceller (AEC3 or AECM)
  -> NoiseSuppression
  -> GainController2 (or GainController1)
  -> CaptureLevelsAdjuster (post-gain)
  -> Output
```

**Reverse path:**
```
Render input
  -> AudioBuffer creation
  -> EchoCanceller (AnalyzeRender)
  -> Output (passthrough)
```

---

## Source Files to Port

### Audio Buffer and Utilities

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `audio_buffer.cc` | Core multi-band audio buffer | High |
| `splitting_filter.cc` | 2/3-band filter bank | High |
| `three_band_filter_bank.cc` | 3-band analysis/synthesis | High |
| `audio_processing/include/audio_frame_proxies.cc` | Frame format helpers | Low |

### Component Wrappers

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `high_pass_filter.cc` | HPF wrapper | Low |
| `gain_control_impl.cc` | AGC1 wrapper | Medium |
| `gain_controller2.cc` | AGC2 wrapper (top-level) | Medium |
| `echo_control_mobile_impl.cc` | AECM wrapper | Medium |
| `residual_echo_detector.cc` | Echo detector wrapper | Medium |
| `rms_level.cc` | RMS level computation | Low |
| `capture_levels_adjuster/capture_levels_adjuster.cc` | Level adjustment | Medium |
| `capture_levels_adjuster/audio_samples_scaler.cc` | Sample scaling | Low |
| `post_filter.cc` | Post-filter | Low |

### Echo Detector

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `echo_detector/circular_buffer.cc` | Circular buffer | Low |
| `echo_detector/mean_variance_estimator.cc` | Stats estimator | Low |
| `echo_detector/moving_max.cc` | Moving maximum | Low |
| `echo_detector/normalized_covariance_estimator.cc` | Covariance | Medium |

### Utility

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `utility/cascaded_biquad_filter.cc` | Cascaded biquad IIR | Medium |
| `utility/delay_estimator.cc` | Legacy delay estimator | Medium |
| `utility/delay_estimator_wrapper.cc` | Delay estimator wrapper | Medium |
| `utility/pffft_wrapper.cc` | PFFFT C wrapper | Low |

### Main Implementation

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `audio_processing_impl.cc` | Main AudioProcessing implementation | Very High |

---

## Tasks

### 8.1 Audio Buffer and Splitting Filter

The `AudioBuffer` is the central data structure that all processing components operate on. It holds multi-band audio data.

**Destination:**
```
webrtc-apm/src/
  audio_buffer.rs       # AudioBuffer
  splitting_filter.rs   # 2/3-band splitting
  three_band_filter.rs  # 3-band filter bank
```

**AudioBuffer design:**
```rust
pub struct AudioBuffer {
    input_num_frames: usize,
    num_frames: usize,
    num_channels: usize,
    num_bands: usize,
    data: Vec<Vec<Vec<f32>>>,  // [band][channel][sample]
    // ... plus integer buffer for i16 path
}

impl AudioBuffer {
    pub fn new(input_num_frames: usize, num_channels: usize,
               proc_num_frames: usize, num_proc_channels: usize,
               output_num_frames: usize) -> Self;
    pub fn split_into_bands(&mut self);
    pub fn merge_bands(&mut self);
    pub fn channels_f32(&self, band: usize) -> &[&[f32]];
    pub fn channels_f32_mut(&mut self, band: usize) -> &mut [&mut [f32]];
}
```

**Verification:**
- [ ] `audio_buffer_unittest` matched
- [ ] `splitting_filter_unittest` matched
- [ ] Band splitting and merging produces identical output

**Commits:**
1. `feat(rust): port AudioBuffer and band splitting`
2. `feat(rust): port three-band filter bank`

---

### 8.2 Echo Detector

Port the lightweight echo detector (separate from AEC3).

**Destination:**
```
webrtc-apm/src/
  echo_detector/
    mod.rs
    circular_buffer.rs
    mean_variance_estimator.rs
    moving_max.rs
    normalized_covariance_estimator.rs
    echo_detector.rs
```

**Verification:**
- [ ] `circular_buffer_unittest` matched
- [ ] `mean_variance_estimator_unittest` matched
- [ ] `moving_max_unittest` matched
- [ ] `normalized_covariance_estimator_unittest` matched
- [ ] `residual_echo_detector_unittest` matched

**Commit:** `feat(rust): port echo detector components`

---

### 8.3 Utility Components

Port the shared utility components.

**Destination:**
```
webrtc-apm/src/
  utility/
    mod.rs
    cascaded_biquad_filter.rs
    delay_estimator.rs
    pffft_wrapper.rs
```

**Verification:**
- [ ] `cascaded_biquad_filter_unittest` matched
- [ ] `delay_estimator_unittest` matched
- [ ] `pffft_wrapper_unittest` matched

**Commit:** `feat(rust): port audio processing utility components`

---

### 8.4 Component Wrappers

Port the wrappers that adapt individual components for the main pipeline.

**Destination:**
```
webrtc-apm/src/
  components/
    mod.rs
    high_pass_filter.rs
    gain_control_impl.rs       # AGC1 wrapper
    gain_controller2_impl.rs   # AGC2 wrapper
    echo_control_mobile_impl.rs
    capture_levels_adjuster.rs
    audio_samples_scaler.rs
    rms_level.rs
    post_filter.rs
```

**Verification:**
- [ ] `high_pass_filter_unittest` matched
- [ ] `gain_control_unittest` matched
- [ ] `gain_controller2_unittest` matched
- [ ] `echo_control_mobile_unittest` matched
- [ ] `rms_level_unittest` matched
- [ ] `audio_samples_scaler_unittest` matched
- [ ] `capture_levels_adjuster_unittest` matched

**Commits:**
1. `feat(rust): port high pass filter and RMS level`
2. `feat(rust): port AGC1 and AGC2 wrappers`
3. `feat(rust): port AECM wrapper and capture levels adjuster`

---

### 8.5 Main AudioProcessing Implementation

Port the central `audio_processing_impl.cc` - the largest and most complex file. This orchestrates the entire pipeline.

**Destination:**
```
webrtc-apm/src/
  config.rs             # AudioProcessing::Config equivalent
  stream_config.rs      # StreamConfig, ProcessingConfig
  runtime_setting.rs    # RuntimeSetting
  processing.rs         # AudioProcessingImpl
  builder.rs            # Builder pattern
```

**Rust API design:**
```rust
pub struct AudioProcessingConfig {
    pub pipeline: PipelineConfig,
    pub pre_amplifier: PreAmplifierConfig,
    pub capture_level_adjustment: CaptureLevelAdjustmentConfig,
    pub high_pass_filter: HighPassFilterConfig,
    pub echo_canceller: EchoCancellerConfig,
    pub noise_suppression: NoiseSuppressionConfig,
    pub gain_controller1: GainController1Config,
    pub gain_controller2: GainController2Config,
}

pub struct AudioProcessing {
    config: AudioProcessingConfig,
    // All component instances
    aec3: Option<EchoCanceller3>,
    aecm: Option<EchoControlMobile>,
    ns: Option<NoiseSuppressor>,
    agc1: Option<GainControlImpl>,
    agc2: Option<GainController2>,
    hpf: Option<HighPassFilter>,
    // Buffers
    capture_buffer: AudioBuffer,
    render_buffer: AudioBuffer,
    // State
    stream_analog_level: i32,
    // ...
}

impl AudioProcessing {
    pub fn builder() -> AudioProcessingBuilder;
    pub fn apply_config(&mut self, config: AudioProcessingConfig);
    pub fn process_stream_f32(
        &mut self,
        src: &[&[f32]],
        input_config: &StreamConfig,
        output_config: &StreamConfig,
        dest: &mut [&mut [f32]],
    ) -> Result<(), ApmError>;
    pub fn process_stream_i16(
        &mut self,
        src: &[i16],
        input_config: &StreamConfig,
        output_config: &StreamConfig,
        dest: &mut [i16],
    ) -> Result<(), ApmError>;
    pub fn process_reverse_stream_f32(...) -> Result<(), ApmError>;
    pub fn process_reverse_stream_i16(...) -> Result<(), ApmError>;
    pub fn set_stream_analog_level(&mut self, level: i32);
    pub fn recommended_stream_analog_level(&self) -> i32;
    pub fn set_stream_delay_ms(&mut self, delay: i32);
    pub fn get_statistics(&self) -> AudioProcessingStats;
    pub fn get_config(&self) -> &AudioProcessingConfig;
}
```

**End-to-end proptest:**
```rust
proptest! {
    #[test]
    fn full_apm_pipeline_matches_cpp(
        config in apm_config_strategy(),
        rate in sample_rate(),
        render_frames in proptest::collection::vec(audio_frame_f32(48000), 20..100),
        capture_frames in proptest::collection::vec(audio_frame_f32(48000), 20..100),
    ) {
        // Create both Rust and C++ APM with same config
        // Process all render frames, then all capture frames
        // Compare output after each ProcessStream call
    }
}
```

**Verification:**
- [ ] `audio_processing_impl_unittest` matched
- [ ] `audio_processing_unittest` matched
- [ ] End-to-end: 100 frames through full pipeline, output matches C++
- [ ] Config changes mid-stream work correctly
- [ ] Sample rate changes work correctly
- [ ] All combinations of enabled/disabled components work
- [ ] `audio_frame_view_unittest` matched

**Commits:**
1. `feat(rust): port AudioProcessing config and stream types`
2. `feat(rust): port AudioProcessingImpl core processing pipeline`
3. `feat(rust): port AudioProcessingImpl initialization and config handling`
4. `feat(rust): add end-to-end AudioProcessing property tests`

---

## Phase 8 Completion Checklist

- [ ] AudioBuffer with band splitting ported and tested
- [ ] All component wrappers ported (HPF, AGC1, AGC2, AECM, NS)
- [ ] Echo detector ported
- [ ] Utility components ported (cascaded biquad, delay estimator, pffft wrapper)
- [ ] Main `AudioProcessingImpl` ported
- [ ] Full pipeline produces identical output to C++ through `ProcessStream`
- [ ] All configuration combinations work
- [ ] All sample rates work (8k, 16k, 32k, 48k)
- [ ] `cargo test -p webrtc-apm` passes
- [ ] All C++ unit tests have Rust equivalents
- [ ] C++ tests still pass

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1-2 | AudioBuffer and splitting | Core data structure |
| 3 | Echo detector | 5 echo detector files |
| 4 | Utilities | Biquad, delay estimator, pffft |
| 5-7 | Component wrappers | HPF, AGC, AECM, levels |
| 8-11 | Main implementation | Config, processing, init, e2e tests |
| **Total** | **11** | |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AudioBuffer layout incompatibility | Medium | Critical | Match C++ memory layout exactly; test band split/merge round-trip |
| Component interaction effects | Medium | High | Test with synthetic echo scenarios, not just random noise |
| Config change mid-stream crashes | Medium | Medium | Test rapid config toggling in proptest |
| Thread safety model differences | Low | High | Mirror C++ threading assumptions; document safety guarantees |
| Memory allocation patterns | Low | Medium | Profile memory usage; pre-allocate where possible |
