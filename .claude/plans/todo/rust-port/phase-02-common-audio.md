# Phase 2: Common Audio Primitives

**Status:** Not Started
**Estimated Duration:** 3-4 weeks
**Dependencies:** Phase 1 (Foundation Infrastructure)
**Outcome:** The `webrtc-common-audio` crate contains working Rust implementations of ring buffer, signal processing functions, FIR filters, resamplers, and FFT wrappers. Every function is proptest-verified against the C++ reference.

---

## Overview

Port the `webrtc/common_audio/` module - the DSP primitive layer that all higher-level audio processing modules depend on. This is the largest leaf dependency and must be bit-exact with the C++ implementation.

## Source Files to Port

### From `webrtc/common_audio/` (Meson build list):

**Ring Buffer (1 file):**
- `ring_buffer.c` -> `ring_buffer.rs`

**Signal Processing Library (30+ files):**
- `signal_processing/auto_correlation.c`
- `signal_processing/auto_corr_to_refl_coef.c`
- `signal_processing/complex_bit_reverse.c`
- `signal_processing/complex_fft.c`
- `signal_processing/copy_set_operations.c`
- `signal_processing/cross_correlation.c`
- `signal_processing/division_operations.c`
- `signal_processing/dot_product_with_scale.cc`
- `signal_processing/downsample_fast.c`
- `signal_processing/energy.c`
- `signal_processing/filter_ar.c`
- `signal_processing/filter_ar_fast_q12.c`
- `signal_processing/filter_ma_fast_q12.c`
- `signal_processing/get_hanning_window.c`
- `signal_processing/get_scaling_square.c`
- `signal_processing/ilbc_specific_functions.c`
- `signal_processing/levinson_durbin.c`
- `signal_processing/lpc_to_refl_coef.c`
- `signal_processing/min_max_operations.c`
- `signal_processing/randomization_functions.c`
- `signal_processing/real_fft.c`
- `signal_processing/refl_coef_to_lpc.c`
- `signal_processing/resample.c`
- `signal_processing/resample_48khz.c`
- `signal_processing/resample_by_2.c`
- `signal_processing/resample_by_2_internal.c`
- `signal_processing/resample_fractional.c`
- `signal_processing/spl_init.c`
- `signal_processing/spl_inl.c`
- `signal_processing/spl_sqrt.c`
- `signal_processing/splitting_filter.c`
- `signal_processing/sqrt_of_one_minus_x_squared.c`
- `signal_processing/vector_scaling_operations.c`

**FIR Filter (4 implementations + factory):**
- `fir_filter_c.cc` (scalar)
- `fir_filter_sse.cc` (SSE2)
- `fir_filter_avx2.cc` (AVX2+FMA)
- `fir_filter_neon.cc` (NEON)
- `fir_filter_factory.cc`

**Resampler (5 files):**
- `resampler/resampler.cc`
- `resampler/sinc_resampler.cc` + `sinc_resampler_sse.cc` + `sinc_resampler_avx2.cc` + `sinc_resampler_neon.cc`
- `resampler/push_resampler.cc`
- `resampler/push_sinc_resampler.cc`

**FFT (Ooura + PFFFT wrapper):**
- `third_party/ooura/fft_size_128/ooura_fft.cc` + SSE2/NEON variants
- `third_party/ooura/fft_size_256/fft4g.cc`
- `third_party/spl_sqrt_floor/spl_sqrt_floor.c`

**Other:**
- `audio_converter.cc`
- `audio_util.cc`
- `channel_buffer.cc`
- `smoothing_filter.cc`
- `vad/` (6 files - deferred to Phase 3)

---

## Tasks

### 2.1 Ring Buffer

Port the C ring buffer to safe Rust.

**Source:** `webrtc/common_audio/ring_buffer.c` (+ `ring_buffer.h`)
**Destination:** `crates/webrtc-common-audio/src/ring_buffer.rs`

The C implementation uses raw pointer arithmetic. The Rust port should use a `Vec<u8>` backing store with read/write cursors.

**API to implement:**
```rust
pub struct RingBuffer {
    data: Vec<u8>,
    read_pos: usize,
    write_pos: usize,
    element_count: usize,
    element_size: usize,
}

impl RingBuffer {
    pub fn new(element_count: usize, element_size: usize) -> Self;
    pub fn write(&mut self, data: &[u8]) -> usize;
    pub fn read(&mut self, data: &mut [u8]) -> usize;
    pub fn move_read_ptr(&mut self, elements: i32) -> usize;
    pub fn available_read(&self) -> usize;
    pub fn available_write(&self) -> usize;
}
```

**Add to C++ shim:** Export `WebRtc_CreateBuffer`, `WebRtc_WriteBuffer`, `WebRtc_ReadBuffer` etc. for comparison.

**Proptest:**
- Random sequence of write/read operations -> compare state with C++ implementation
- Edge cases: full buffer, empty buffer, wrap-around

**Verification:**
- [ ] `ring_buffer_unittest` test cases replicated as Rust unit tests
- [ ] Proptest passes with 10000+ cases
- [ ] No unsafe code in Rust implementation

**Commit:** `feat(rust): port ring buffer to webrtc-common-audio`

---

### 2.2 Signal Processing Library - Basic Operations

Port the fixed-point and utility functions. These are mostly pure C functions operating on `int16_t`/`int32_t` arrays.

**Group 1: Copy/Set/MinMax (simple, low risk)**
- `copy_set_operations.c` -> `signal_processing/copy_set.rs`
- `min_max_operations.c` -> `signal_processing/min_max.rs`
- `energy.c` -> `signal_processing/energy.rs`
- `vector_scaling_operations.c` -> `signal_processing/vector_scaling.rs`
- `get_scaling_square.c` -> `signal_processing/scaling_square.rs`

**Group 2: Math operations**
- `division_operations.c` -> `signal_processing/division.rs`
- `spl_sqrt.c` -> `signal_processing/spl_sqrt.rs`
- `sqrt_of_one_minus_x_squared.c` -> `signal_processing/sqrt_one_minus.rs`
- `dot_product_with_scale.cc` -> `signal_processing/dot_product.rs`
- `third_party/spl_sqrt_floor/spl_sqrt_floor.c` -> `signal_processing/sqrt_floor.rs`

**Group 3: Filter operations**
- `filter_ar.c` -> `signal_processing/filter_ar.rs`
- `filter_ar_fast_q12.c` -> `signal_processing/filter_ar_fast.rs`
- `filter_ma_fast_q12.c` -> `signal_processing/filter_ma_fast.rs`
- `levinson_durbin.c` -> `signal_processing/levinson_durbin.rs`

**Group 4: Correlation and spectral**
- `auto_correlation.c` -> `signal_processing/auto_correlation.rs`
- `auto_corr_to_refl_coef.c` -> `signal_processing/auto_corr_refl.rs`
- `cross_correlation.c` -> `signal_processing/cross_correlation.rs`
- `complex_bit_reverse.c` -> `signal_processing/complex_bit_reverse.rs`
- `complex_fft.c` -> `signal_processing/complex_fft.rs`
- `real_fft.c` -> `signal_processing/real_fft.rs`
- `lpc_to_refl_coef.c` -> `signal_processing/lpc_refl.rs`
- `refl_coef_to_lpc.c` -> `signal_processing/refl_lpc.rs`

**Group 5: Resampling helpers**
- `downsample_fast.c` -> `signal_processing/downsample_fast.rs`
- `resample.c` -> `signal_processing/resample.rs`
- `resample_48khz.c` -> `signal_processing/resample_48khz.rs`
- `resample_by_2.c` + `resample_by_2_internal.c` -> `signal_processing/resample_by_2.rs`
- `resample_fractional.c` -> `signal_processing/resample_fractional.rs`
- `splitting_filter.c` -> `signal_processing/splitting_filter.rs`

**Group 6: Misc**
- `get_hanning_window.c` -> `signal_processing/hanning_window.rs`
- `ilbc_specific_functions.c` -> `signal_processing/ilbc.rs`
- `randomization_functions.c` -> `signal_processing/random.rs`
- `spl_init.c` + `spl_inl.c` -> `signal_processing/spl_init.rs`

**Module structure:**
```
webrtc-common-audio/src/
  signal_processing/
    mod.rs              # Re-exports, SPL global state
    copy_set.rs
    min_max.rs
    energy.rs
    vector_scaling.rs
    ...
```

**Critical note on fixed-point math:** Many of these functions use Q-format fixed-point arithmetic (Q12, Q14, Q15, etc.). The Rust port must use the exact same bit widths and shift amounts. Use `i16`, `i32`, `i64` and explicit `wrapping_*` / `overflowing_*` operations where the C code relies on integer overflow behavior.

**Add to C++ shim:** Export each SPL function for proptest comparison.

**Proptest strategy:** For each function:
1. Generate random valid inputs (correct types, array lengths)
2. Call both Rust and C++ via FFI
3. Assert bit-exact equality for integer functions
4. Assert near-equality (tolerance 1e-6) for float functions

**Verification per group:**
- [ ] All existing C++ unit tests for signal_processing replicated
- [ ] Proptest passes for each function (1000+ cases minimum)
- [ ] `signal_processing_unittest` and `real_fft_unittest` test behaviors matched

**Commits (one per group, each must compile and test green):**
1. `feat(rust): port signal processing copy/set/minmax operations`
2. `feat(rust): port signal processing math operations`
3. `feat(rust): port signal processing filter operations`
4. `feat(rust): port signal processing correlation and spectral operations`
5. `feat(rust): port signal processing resampling helpers`
6. `feat(rust): port signal processing misc operations`

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

Port the audio resampling chain. This has a dependency chain:
`PushResampler` -> `PushSincResampler` -> `SincResampler` (+ SIMD variants) and `Resampler`

**Port order (bottom-up):**

**Step 1: SincResampler (core)**
- `sinc_resampler.cc` - The main sinc interpolation engine
- `sinc_resampler_sse.cc` - SSE2 inner loop
- `sinc_resampler_avx2.cc` - AVX2+FMA inner loop
- `sinc_resampler_neon.cc` - NEON inner loop

**Step 2: Legacy Resampler**
- `resampler.cc` - Uses the signal_processing resample functions

**Step 3: Push Resamplers**
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
    resampler.rs        # Legacy Resampler
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
- [ ] `resampler_unittest` behaviors matched
- [ ] Proptest: rate conversions produce matching output

**Commits:**
1. `feat(rust): port SincResampler (scalar)`
2. `feat(rust): port SincResampler SIMD variants (SSE2, AVX2, NEON)`
3. `feat(rust): port legacy Resampler`
4. `feat(rust): port PushSincResampler and PushResampler`

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

### 2.6 Utility Classes

Port the remaining common_audio classes.

**Files:**
- `audio_converter.cc` -> `audio_converter.rs` - Sample format conversion
- `audio_util.cc` + `include/audio_util.h` -> `audio_util.rs` - int16/float conversion
- `channel_buffer.cc` -> `channel_buffer.rs` - Multi-channel buffer
- `smoothing_filter.cc` -> `smoothing_filter.rs` - Exponential smoothing

**Verification:**
- [ ] `audio_converter_unittest` behaviors matched
- [ ] `audio_util_unittest` behaviors matched
- [ ] `channel_buffer_unittest` behaviors matched
- [ ] `smoothing_filter_unittest` behaviors matched

**Commit:** `feat(rust): port common audio utility classes (converter, channel buffer, smoothing filter)`

---

## Phase 2 Completion Checklist

- [ ] All `webrtc/common_audio/` source files ported (except `vad/` - Phase 3)
- [ ] Every ported function has a proptest comparing against C++ reference
- [ ] All relevant C++ unit tests have equivalent Rust test coverage
- [ ] SIMD variants (SSE2, AVX2, NEON) ported for FIR filter and SincResampler
- [ ] FFT wrappers produce bit-exact results
- [ ] `cargo test -p webrtc-common-audio` passes all tests
- [ ] `cargo test -p webrtc-apm-proptest` passes all property tests
- [ ] C++ tests still pass (`meson test -C builddir`)
- [ ] No regressions in any previously passing test

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1 | `feat(rust): port ring buffer` | ring_buffer.rs |
| 2-7 | `feat(rust): port signal processing [group]` | 6 groups of SPL functions |
| 8-12 | `feat(rust): port FIR filter [variant]` | scalar, SSE2, AVX2, NEON, factory |
| 13-16 | `feat(rust): port resampler [component]` | SincResampler, SIMD, legacy, push |
| 17-19 | `feat(rust): port FFT [type]` | pffft, ooura128, ooura256 |
| 20 | `feat(rust): port common audio utilities` | converter, channel_buffer, smoothing |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fixed-point arithmetic differences | Medium | High | Use wrapping arithmetic explicitly; test edge cases (overflow, underflow) |
| FIR filter SIMD not bit-exact | Medium | Medium | Compare intermediate accumulator values, not just final sum |
| SincResampler state divergence | Low | High | Seed both with same initial state; compare after each frame |
| Ooura FFT twiddle factor precision | Low | Medium | Copy exact constants from C source |
| PFFFT link issues | Low | Low | cc crate is well-tested; use same flags as Meson |
