# Phase 2: Common Audio Primitives

**Status:** In Progress (ring buffer complete)
**Estimated Duration:** 2-3 weeks
**Dependencies:** Phase 1 (Foundation Infrastructure)
**Outcome:** The `webrtc-common-audio` crate contains working Rust implementations of audio utilities, FIR filters, resamplers, and FFT wrappers. Every function is proptest-verified against the C++ reference.

---

## Overview

Port the subset of `webrtc/common_audio/` needed by the modern audio processing pipeline (AEC3, AGC2, NS). The Signal Processing Library (30+ fixed-point C files) is **not ported** — it is only used by AECM (removed upstream), AGC1 (deprecated), and VAD filterbank. See master plan for rationale.

## Source Files to Port

### From `webrtc/common_audio/` (modern pipeline dependencies only):

**Ring Buffer (1 file) — COMPLETE:**
- `ring_buffer.c` -> standalone `webrtc-ring-buffer` crate (16 tests)

**Audio Utilities (3 files):**
- `audio_util.cc` + `include/audio_util.h` -> `audio_util.rs` (int16/float conversions)
- `channel_buffer.cc` -> `channel_buffer.rs` (multi-channel buffer)
- `smoothing_filter.cc` -> `smoothing_filter.rs` (exponential smoothing)

**FIR Filter (4 implementations + factory):**
- `fir_filter_c.cc` (scalar)
- `fir_filter_sse.cc` (SSE2)
- `fir_filter_avx2.cc` (AVX2+FMA)
- `fir_filter_neon.cc` (NEON)
- `fir_filter_factory.cc`

**Resampler (4 files):**
- `resampler/sinc_resampler.cc` + `sinc_resampler_sse.cc` + `sinc_resampler_avx2.cc` + `sinc_resampler_neon.cc`
- `resampler/push_resampler.cc`
- `resampler/push_sinc_resampler.cc`

**FFT (Ooura + PFFFT wrapper):**
- `third_party/ooura/fft_size_128/ooura_fft.cc` + SSE2/NEON variants
- `third_party/ooura/fft_size_256/fft4g.cc`

**Not ported (legacy):**
- `signal_processing/` (30+ files) — only used by AECM, AGC1, VAD filterbank
- `resampler/resampler.cc` — legacy resampler, modern pipeline uses SincResampler
- `audio_converter.cc` — thin wrapper, inline in integration phase
- `vad/` (6 files) — deferred to Phase 3

---

## Tasks

### 2.1 Ring Buffer — COMPLETE

Ported as standalone `webrtc-ring-buffer` crate (not inside `webrtc-common-audio`).

**Crate:** `crates/webrtc-ring-buffer/`
**API:** Generic `RingBuffer<T>` with `NonZero<usize>` capacity, `isize` cursor movement, `ZeroCopyResult` enum.
**Tests:** 16 (unit + proptest with `test-strategy`)

**Verification:**
- [x] `ring_buffer_unittest` patterns replicated as Rust unit tests
- [x] Proptest stress test (random read/write/move sequences)
- [x] No unsafe code
- [x] Clippy clean

**Commit:** `feat(rust): port ring buffer as webrtc-ring-buffer crate`

---

### 2.2 Audio Utilities

Port the audio utility functions used by the modern pipeline.

**Source files:**
- `audio_util.cc` + `include/audio_util.h` — int16/float conversion, scaling, clamping
- `channel_buffer.cc` + `channel_buffer.h` — multi-channel interleaved/deinterleaved buffer
- `smoothing_filter.cc` — exponential smoothing filter

**Destination:**
```
webrtc-common-audio/src/
  audio_util.rs         # S16ToFloat, FloatToS16, FloatS16ToS16, etc.
  channel_buffer.rs     # ChannelBuffer<T>
  smoothing_filter.rs   # SmoothingFilter
```

**Add to C++ shim:** Export `S16ToFloat`, `FloatToS16`, `FloatS16ToS16` etc. for comparison.

**Proptest:**
- int16 -> float -> int16 roundtrip preserves values
- Float scaling matches C++ exactly
- ChannelBuffer interleave/deinterleave roundtrip

**Verification:**
- [ ] `audio_util_unittest` behaviors matched
- [ ] `channel_buffer_unittest` behaviors matched
- [ ] `smoothing_filter_unittest` behaviors matched
- [ ] Proptest passes for each function (1000+ cases minimum)

**Commits:**
1. `feat(rust): port audio_util (int16/float conversions)`
2. `feat(rust): port ChannelBuffer and SmoothingFilter`

Note: Signal Processing Library (30+ fixed-point C files) is **not ported**.
See master plan "Excluded Modules" for rationale.

---

### 2.3 FIR Filter

Port the FIR filter with all SIMD variants.

**Source files:**
- `fir_filter_c.cc` - Scalar reference
- `fir_filter_sse.cc` - SSE2
- `fir_filter_avx2.cc` - AVX2 + FMA
- `fir_filter_neon.cc` - NEON
- `fir_filter_factory.cc` - Runtime dispatch

**Destination:**
```
webrtc-common-audio/src/
  fir_filter/
    mod.rs          # FirFilter trait + factory
    scalar.rs       # Scalar implementation
    sse2.rs         # SSE2 (uses webrtc-simd or direct std::arch)
    avx2.rs         # AVX2+FMA
    neon.rs         # NEON
```

**Design:**
```rust
pub trait FirFilter: Send {
    fn filter(&mut self, input: &[f32], output: &mut [f32]);
    fn filter_length(&self) -> usize;
}

pub fn create_fir_filter(coefficients: &[f32], max_input_length: usize) -> Box<dyn FirFilter> {
    // Runtime dispatch based on CPU features
}
```

**SIMD porting approach:**
1. First port the scalar implementation and verify it's bit-exact
2. Port SSE2, comparing against scalar output
3. Port AVX2+FMA, comparing against scalar output
4. Port NEON, comparing against scalar output

**Critical:** The AVX2 version uses FMA instructions (`_mm256_fmadd_ps`). Ensure the Rust version uses `_mm256_fmadd_ps` from `std::arch::x86_64` with `#[target_feature(enable = "avx2,fma")]`.

**Proptest:**
```rust
proptest! {
    #[test]
    fn fir_scalar_matches_cpp(
        coeffs in fir_coefficients(128),
        input in proptest::collection::vec(-1.0f32..=1.0f32, 1..1024),
    ) {
        let rust_output = rust_fir_filter_scalar(&coeffs, &input);
        let cpp_output = cpp_fir_filter(&coeffs, &input);
        assert_f32_near(&rust_output, &cpp_output, 1e-6);
    }
}
```

**Verification:**
- [ ] `fir_filter_unittest` behaviors matched
- [ ] Scalar implementation is bit-exact with C++ scalar
- [ ] SSE2 output matches scalar output (within float tolerance)
- [ ] AVX2 output matches scalar output (within float tolerance)
- [ ] NEON output matches scalar output (within float tolerance)
- [ ] Proptest: 10000+ cases pass for each variant

**Commits:**
1. `feat(rust): port scalar FIR filter`
2. `feat(rust): port SSE2 FIR filter`
3. `feat(rust): port AVX2+FMA FIR filter`
4. `feat(rust): port NEON FIR filter`
5. `feat(rust): add FIR filter factory with runtime SIMD dispatch`

---

### 2.4 Resampler

Port the audio resampling chain used by the modern pipeline. The legacy `Resampler` class (which depends on the SPL library) is not ported.

Dependency chain: `PushResampler` -> `PushSincResampler` -> `SincResampler` (+ SIMD variants)

**Port order (bottom-up):**

**Step 1: SincResampler (core)**
- `sinc_resampler.cc` - The main sinc interpolation engine
- `sinc_resampler_sse.cc` - SSE2 inner loop
- `sinc_resampler_avx2.cc` - AVX2+FMA inner loop
- `sinc_resampler_neon.cc` - NEON inner loop

**Step 2: Push Resamplers**
- `push_sinc_resampler.cc` - Wraps SincResampler for push-style API
- `push_resampler.cc` - Top-level push resampler

**Destination:**
```
webrtc-common-audio/src/
  resampler/
    mod.rs              # Re-exports
    sinc_resampler.rs   # Core SincResampler
    sinc_sse2.rs        # SSE2 inner loop
    sinc_avx2.rs        # AVX2+FMA inner loop
    sinc_neon.rs        # NEON inner loop
    push_sinc.rs        # PushSincResampler
    push_resampler.rs   # PushResampler
```

**Proptest strategy:**
- Generate audio at one sample rate
- Resample to another sample rate via both Rust and C++
- Compare output buffers
- Test all rate combinations: 8k, 16k, 32k, 48k in both directions

**Verification:**
- [ ] `sinc_resampler_unittest` behaviors matched
- [ ] `push_resampler_unittest` behaviors matched
- [ ] `push_sinc_resampler_unittest` behaviors matched
- [ ] Proptest: rate conversions produce matching output

**Commits:**
1. `feat(rust): port SincResampler (scalar)`
2. `feat(rust): port SincResampler SIMD variants (SSE2, AVX2, NEON)`
3. `feat(rust): port PushSincResampler and PushResampler`

---

### 2.5 FFT Wrappers

Port or wrap the FFT implementations.

**Strategy:**
- **PFFFT:** Compile `webrtc/third_party/pffft/pffft.c` via `cc` crate and write thin Rust FFI bindings. This ensures bit-exact behavior. Do NOT use a pure Rust FFT here.
- **Ooura 128-point:** Port to Rust (relatively small, ~300 lines). Include SSE2/NEON variants.
- **Ooura 256-point:** Port `fft4g.cc` to Rust.

**Destination:**
```
webrtc-common-audio/src/
  fft/
    mod.rs          # FFT traits and re-exports
    pffft.rs        # Safe Rust wrapper around C pffft
    ooura128.rs     # Ooura 128-point FFT (ported to Rust)
    ooura128_sse2.rs
    ooura128_neon.rs
    ooura256.rs     # Ooura 256-point FFT (ported to Rust)
```

**build.rs addition for pffft:**
```rust
// In webrtc-common-audio/build.rs
cc::Build::new()
    .file("../../webrtc/third_party/pffft/pffft.c")
    .flag_if_supported("-msse2")  // or neon equivalent
    .compile("pffft");
```

**Proptest:**
- Forward FFT then inverse FFT should reconstruct the input (within tolerance)
- Compare Rust FFT output against C++ FFT output for random inputs

**Verification:**
- [ ] `pffft_wrapper_unittest` behaviors matched
- [ ] Forward/inverse FFT identity holds (tolerance < 1e-5)
- [ ] Ooura 128 matches C++ output bit-for-bit (integer paths) or within tolerance (float paths)

**Commits:**
1. `feat(rust): add pffft C library compilation and safe Rust wrapper`
2. `feat(rust): port Ooura 128-point FFT with SSE2/NEON variants`
3. `feat(rust): port Ooura 256-point FFT`

---

## Phase 2 Completion Checklist

- [x] Ring buffer ported as `webrtc-ring-buffer` crate (16 tests)
- [ ] Audio utilities ported (audio_util, channel_buffer, smoothing_filter)
- [ ] Every ported function has a proptest comparing against C++ reference
- [ ] All relevant C++ unit tests have equivalent Rust test coverage
- [ ] SIMD variants (SSE2, AVX2, NEON) ported for FIR filter and SincResampler
- [ ] FFT wrappers produce bit-exact results
- [ ] `cargo nextest run -p webrtc-common-audio` passes all tests
- [ ] `cargo nextest run -p webrtc-apm-proptest` passes all property tests
- [ ] C++ tests still pass (`meson test -C builddir`)
- [ ] No regressions in any previously passing test

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1 | `feat(rust): port ring buffer as webrtc-ring-buffer crate` | **DONE** |
| 2-3 | `feat(rust): port audio utilities` | audio_util, channel_buffer, smoothing |
| 4-8 | `feat(rust): port FIR filter [variant]` | scalar, SSE2, AVX2, NEON, factory |
| 9-11 | `feat(rust): port resampler [component]` | SincResampler, SIMD, push |
| 12-14 | `feat(rust): port FFT [type]` | pffft, ooura128, ooura256 |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| FIR filter SIMD not bit-exact | Medium | Medium | Compare intermediate accumulator values, not just final sum |
| SincResampler state divergence | Low | High | Seed both with same initial state; compare after each frame |
| Ooura FFT twiddle factor precision | Low | Medium | Copy exact constants from C source |
| PFFFT link issues | Low | Low | cc crate is well-tested; use same flags as Meson |
