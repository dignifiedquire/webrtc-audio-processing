# Phase 3: Voice Activity Detection

**Status:** Not Started
**Estimated Duration:** 2 weeks
**Dependencies:** Phase 2 (Common Audio Primitives - signal processing functions, FFT)
**Outcome:** The `webrtc-vad` crate contains a working Rust VAD implementation that produces identical frame-by-frame decisions to the C++ reference. Both the core C VAD and the higher-level standalone VAD are ported.

---

## Overview

Port the Voice Activity Detection (VAD) module. The VAD has two layers:
1. **Core VAD** (`common_audio/vad/`) - Low-level C implementation using Gaussian Mixture Models
2. **Standalone VAD** (`modules/audio_processing/vad/`) - Higher-level C++ wrapper with pitch-based features

The core VAD is used directly by other modules (AGC1, AGC2). The standalone VAD adds pitch-based voice detection for the higher-level audio processing pipeline.

---

## Source Files to Port

### Core VAD (`webrtc/common_audio/vad/` - already in common_audio build)

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `vad_core.c` | Core VAD algorithm, state machine | High |
| `vad_filterbank.c` | Filterbank energy computation | Medium |
| `vad_gmm.c` | Gaussian Mixture Model evaluation | Medium |
| `vad_sp.c` | Signal processing helpers for VAD | Medium |
| `webrtc_vad.c` | Public C API wrapper | Low |
| `vad.cc` | C++ wrapper around C API | Low |

### Standalone VAD (`webrtc/modules/audio_processing/vad/`)

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `voice_activity_detector.cc` | High-level VAD using pitch features | High |
| `vad_audio_proc.cc` | Audio preprocessing for VAD | Medium |
| `pitch_based_vad.cc` | Pitch-based voice detection | Medium |
| `pitch_internal.cc` | Pitch estimation internals | Medium |
| `pole_zero_filter.cc` | IIR filter for preprocessing | Low |
| `standalone_vad.cc` | Standalone VAD wrapper | Low |
| `vad_circular_buffer.cc` | Circular buffer for VAD features | Low |
| `gmm.cc` | GMM utilities | Low |

---

## Tasks

### 3.1 Core VAD Port

Port the C VAD implementation to Rust. This is the most critical piece - it must be bit-exact since it's used as a building block.

**Destination:**
```
webrtc-vad/src/
  lib.rs              # Public API (WebRtcVad equivalent)
  core.rs             # VadCore - main algorithm
  filterbank.rs       # Filterbank energy computation
  gmm.rs              # Gaussian Mixture Model
  sp.rs               # Signal processing helpers
```

**Key types:**
```rust
/// VAD aggressiveness mode (matches C enum)
#[repr(i32)]
pub enum VadMode {
    Quality = 0,     // Least aggressive
    LowBitrate = 1,
    Aggressive = 2,
    VeryAggressive = 3,
}

/// Main VAD interface
pub struct Vad {
    core: VadCore,
}

impl Vad {
    pub fn new() -> Self;
    pub fn set_mode(&mut self, mode: VadMode) -> Result<(), VadError>;
    pub fn is_vad_active(&mut self, audio: &[i16], sample_rate: i32) -> Result<bool, VadError>;
    pub fn valid_rate(sample_rate: i32) -> bool;
}
```

**Porting notes:**
- The C code uses global tables (GMM parameters). In Rust, use `const` arrays.
- `vad_core.c` has significant internal state (per-frequency-band energy history). Mirror this exactly.
- Integer arithmetic must use `wrapping_add`, `wrapping_mul` where C relies on overflow behavior.
- The filterbank uses the signal processing library functions ported in Phase 2 (specifically: `WebRtcSpl_CrossCorrelation`, energy calculations).

**Add to C++ shim:**
```cpp
// Expose VAD for comparison testing
int32_t vad_process(int32_t sample_rate, rust::Slice<const int16_t> audio) -> int32_t;
```

**Proptest:**
```rust
proptest! {
    #[test]
    fn vad_decision_matches_cpp(
        rate in prop_oneof![Just(8000i32), Just(16000), Just(32000), Just(48000)],
        audio in audio_frame_i16_for_rate(rate),
        mode in 0i32..=3,
    ) {
        let rust_result = rust_vad_process(rate, mode, &audio);
        let cpp_result = cpp_vad_process(rate, mode, &audio);
        prop_assert_eq!(rust_result, cpp_result);
    }
}
```

**Verification:**
- [ ] `vad_core_unittest` behaviors matched
- [ ] `vad_filterbank_unittest` behaviors matched
- [ ] `vad_gmm_unittest` behaviors matched
- [ ] `vad_sp_unittest` behaviors matched
- [ ] `vad_unittest` behaviors matched
- [ ] Proptest: 10000+ frames with matching decisions across all modes and sample rates
- [ ] Multi-frame sequences produce identical state evolution

**Commits:**
1. `feat(rust): port core VAD filterbank and GMM`
2. `feat(rust): port core VAD algorithm and public API`

---

### 3.2 Standalone VAD Port

Port the higher-level pitch-based VAD.

**Destination:**
```
webrtc-vad/src/
  standalone/
    mod.rs                    # StandaloneVad
    voice_activity_detector.rs # VoiceActivityDetector
    vad_audio_proc.rs         # Audio preprocessing
    pitch_based_vad.rs        # Pitch-based detection
    pitch_internal.rs         # Pitch estimation
    pole_zero_filter.rs       # IIR filter
    circular_buffer.rs        # Feature circular buffer
    gmm.rs                    # GMM utilities for standalone VAD
```

**Porting order (bottom-up):**
1. `pole_zero_filter.cc` - Simple IIR filter, no dependencies
2. `vad_circular_buffer.cc` - Simple circular buffer for features
3. `gmm.cc` - GMM evaluation utilities
4. `pitch_internal.cc` - Pitch estimation helpers
5. `vad_audio_proc.cc` - Audio preprocessing pipeline
6. `pitch_based_vad.cc` - Pitch-based voice detection
7. `voice_activity_detector.cc` - Full VAD pipeline
8. `standalone_vad.cc` - Wrapper

**Proptest:** Process multi-frame audio sequences and compare VAD probability outputs.

**Verification:**
- [ ] `pole_zero_filter_unittest` behaviors matched
- [ ] `vad_circular_buffer_unittest` behaviors matched
- [ ] `gmm_unittest` behaviors matched
- [ ] `pitch_internal_unittest` behaviors matched
- [ ] `pitch_based_vad_unittest` behaviors matched
- [ ] `vad_audio_proc_unittest` behaviors matched
- [ ] `voice_activity_detector_unittest` behaviors matched
- [ ] `standalone_vad_unittest` behaviors matched

**Commits:**
1. `feat(rust): port standalone VAD filters and utilities`
2. `feat(rust): port pitch-based VAD and audio processing`
3. `feat(rust): port VoiceActivityDetector and StandaloneVad`

---

## Phase 3 Completion Checklist

- [ ] Core VAD (`webrtc/common_audio/vad/`) fully ported to `webrtc-vad`
- [ ] Standalone VAD (`webrtc/modules/audio_processing/vad/`) fully ported
- [ ] All 13 C++ VAD-related unit tests have equivalent Rust coverage
- [ ] Proptest: frame-by-frame VAD decisions match C++ for all modes and sample rates
- [ ] Multi-frame sequence testing confirms identical state evolution
- [ ] `cargo test -p webrtc-vad` passes
- [ ] `cargo test -p webrtc-apm-proptest` passes (including new VAD proptests)
- [ ] C++ tests still pass (`meson test -C builddir`)

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1 | `feat(rust): port core VAD filterbank and GMM` | filterbank.rs, gmm.rs, sp.rs |
| 2 | `feat(rust): port core VAD algorithm and public API` | core.rs, lib.rs |
| 3 | `feat(rust): port standalone VAD filters and utilities` | pole_zero_filter, circular_buffer, gmm |
| 4 | `feat(rust): port pitch-based VAD and audio processing` | pitch_internal, pitch_based_vad, vad_audio_proc |
| 5 | `feat(rust): port VoiceActivityDetector and StandaloneVad` | voice_activity_detector, standalone_vad |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GMM table precision | Low | High | Copy exact constants from C; verify with edge-case inputs |
| VAD state divergence over long sequences | Medium | High | Test with 1000+ frame sequences; compare intermediate state |
| Core VAD integer overflow behavior | Medium | Medium | Use explicit wrapping arithmetic; test boundary values |
| Pitch estimation floating-point divergence | Low | Medium | Use same algorithm structure; compare intermediate pitch values |
