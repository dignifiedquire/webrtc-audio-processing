# Phase 7: Mobile Echo Control (AECM)

**Status:** Not Started
**Estimated Duration:** 1-2 weeks
**Dependencies:** Phase 2 (Common Audio)
**Outcome:** The `webrtc-aecm` crate contains a working Rust AECM implementation with NEON optimizations that produces identical output to the C++ reference.

---

## Overview

Port the Acoustic Echo Control for Mobile (AECM) - a simplified echo canceller designed for low-power mobile devices. AECM uses fixed-point (int16/int32) arithmetic throughout, unlike the floating-point AEC3.

AECM is smaller than AEC3 (~8 files vs 65+) but has important NEON optimizations for ARM platforms.

---

## Source Files to Port

| Source File | Description | Complexity |
|-------------|-------------|------------|
| `aecm_core.cc` | Core AECM algorithm (scalar) | High |
| `aecm_core_c.cc` | C scalar implementation (non-MIPS) | Medium |
| `aecm_core_neon.cc` | NEON-optimized operations | Medium |
| `aecm_core_mips.cc` | MIPS-optimized operations | Low (skip) |
| `echo_control_mobile.cc` | Top-level AECM interface | Medium |

**Note:** MIPS support (`aecm_core_mips.cc`) is deprioritized. Focus on scalar + NEON.

---

## Tasks

### 7.1 AECM Core (Scalar)

Port the core AECM algorithm.

**Destination:**
```
webrtc-aecm/src/
  lib.rs                    # Public API
  core.rs                   # AECM core algorithm
  core_scalar.rs            # Scalar implementation
```

**Key characteristics:**
- Fixed-point arithmetic (Q14, Q15 formats)
- Uses `common_audio/signal_processing` functions heavily
- NLMS (Normalized Least Mean Squares) adaptation
- Comfort noise generation in frequency domain

**Porting notes:**
- All arithmetic is integer-based. Must use `wrapping_*` operations where C code relies on overflow.
- The core uses `WebRtcSpl_*` functions from Phase 2's signal processing port.
- Pay careful attention to Q-format conversions (shifting left/right by specific amounts).

**Add to C++ shim:**
```cpp
int32_t aecm_process(
    rust::Slice<const int16_t> render,
    rust::Slice<const int16_t> capture,
    int32_t sample_rate,
    rust::Slice<int16_t> output);
```

**Proptest:**
```rust
proptest! {
    #[test]
    fn aecm_output_matches_cpp(
        render in audio_frame_i16(8000),
        capture in audio_frame_i16(8000),
    ) {
        let rust_output = rust_aecm_process(&render, &capture);
        let cpp_output = cpp_aecm_process(&render, &capture);
        assert_i16_exact(&rust_output, &cpp_output);
    }
}
```

**Verification:**
- [ ] `echo_control_mobile_bit_exact_unittest` matched (critical - bit-exact test)
- [ ] All integer arithmetic matches exactly (no rounding differences)

**Commit:** `feat(rust): port AECM core scalar implementation`

---

### 7.2 AECM NEON Optimizations

Port the NEON-optimized functions.

**Destination:** `webrtc-aecm/src/core_neon.rs`

**Functions to port:**
- NLMS adaptation (NEON)
- Frequency-domain operations (NEON)
- Various inner loops accelerated with NEON intrinsics

**Verification:**
- [ ] NEON output matches scalar output (bit-exact for integer operations)
- [ ] Performance improvement visible on ARM platforms

**Commit:** `feat(rust): port AECM NEON optimizations`

---

### 7.3 AECM Top-Level

Port the top-level echo control mobile interface.

**Destination:** `webrtc-aecm/src/echo_control_mobile.rs`

**Public API:**
```rust
pub struct EchoControlMobile {
    // Internal state
}

impl EchoControlMobile {
    pub fn new(sample_rate: i32) -> Self;
    pub fn process_render_audio(&mut self, render: &[i16]) -> Result<(), AecmError>;
    pub fn process_capture_audio(
        &mut self,
        capture: &[i16],
        output: &mut [i16],
    ) -> Result<(), AecmError>;
    pub fn set_echo_path(&mut self, path: &[i16]) -> Result<(), AecmError>;
    pub fn get_echo_path(&self) -> Vec<i16>;
}
```

**Verification:**
- [ ] `echo_control_mobile_unittest` matched
- [ ] `echo_control_mobile_bit_exact_unittest` matched
- [ ] Multi-frame processing produces identical output

**Commit:** `feat(rust): port EchoControlMobile top-level interface`

---

## Phase 7 Completion Checklist

- [ ] AECM core algorithm ported (scalar)
- [ ] NEON optimizations ported
- [ ] Top-level interface ported
- [ ] Bit-exact with C++ for integer (i16) processing
- [ ] `echo_control_mobile_unittest` matched
- [ ] `echo_control_mobile_bit_exact_unittest` matched
- [ ] `cargo test -p webrtc-aecm` passes
- [ ] Proptest: 1000+ frame pairs with exact output match
- [ ] C++ tests still pass

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1 | `feat(rust): port AECM core scalar implementation` | core, core_scalar |
| 2 | `feat(rust): port AECM NEON optimizations` | core_neon |
| 3 | `feat(rust): port EchoControlMobile top-level interface` | echo_control_mobile |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fixed-point overflow behavior | High | Critical | Use wrapping arithmetic everywhere; test edge values |
| Q-format conversion errors | Medium | High | Add assertion macros for Q-format range checks in debug builds |
| NEON int16 saturating arithmetic | Medium | Medium | Use Rust saturating_* methods where C uses NEON saturation intrinsics |
| Missing SPL function from Phase 2 | Low | Medium | Ensure all WebRtcSpl functions used by AECM are ported |
