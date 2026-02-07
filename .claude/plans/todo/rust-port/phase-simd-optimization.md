# SIMD Optimization Phase

**Status:** Complete
**Estimated Duration:** 3-4 weeks
**Dependencies:** Phases 1-7 (all component crates complete)
**Outcome:** All SIMD paths matching the C++ codebase are ported, `cpufeatures` replaces manual detection, shared SIMD code lives in `webrtc-simd`, and comprehensive tests verify SIMD vs scalar equivalence.

---

## Overview

The C++ codebase uses `Aec3Optimization` enum to dispatch between Scalar/SSE2/AVX2/NEON for hot-path functions. The Rust port currently has:
- `webrtc-simd` crate with 8 generic SIMD operations (all 4 backends)
- Ooura 128-point FFT with SSE2 and NEON (fully wired up)
- SincResampler using `webrtc-simd::dual_dot_product` (all backends)
- AGC2 RNN VAD using `webrtc-simd::dot_product` (all backends)
- AEC3 `VectorMath` using `webrtc-simd` sqrt/multiply/accumulate (all backends)
- AEC3 `FftData::spectrum` using `webrtc-simd::power_spectrum` (all backends)

**Missing SIMD paths (scalar-only in Rust, SIMD in C++):**
- `MatchedFilterCore` — SSE2, AVX2, NEON (convolution + filter update)
- `AdaptPartitions` — SSE2, AVX2, NEON (frequency-domain filter adaptation)
- `ApplyFilter` — SSE2, AVX2, NEON (frequency-domain filter application)
- `ComputeFrequencyResponse` — SSE2, AVX2, NEON (H2 computation)
- `ComputeErl` — SSE2, AVX2, NEON (echo return loss from H2)
- PFFFT — SSE/NEON (4-wide butterfly operations)

**Infrastructure improvements:**
- Replace `is_x86_feature_detected!` / `is_aarch64_feature_detected!` with `cpufeatures`
- Remove `force-*` feature flags (cpufeatures handles this better)

**Not needed (confirmed):**
- Inline assembly (`core::arch::asm!`) — all C++ intrinsics have direct `std::arch` equivalents
- MIPS SIMD — architecture not in scope
- ARM32 assembly — only used by excluded SPL/AECM modules
- FIR filter SIMD — not consumed by any ported module

---

## Inventory: C++ SIMD → Rust Status

### AEC3 Module-Level SIMD (uses `Aec3Optimization` dispatch)

| C++ Function | C++ File(s) | ISAs | Rust File | Status |
|---|---|---|---|---|
| `MatchedFilterCore` | `matched_filter.cc` | SSE2, NEON inline | `matched_filter.rs` | **Scalar only** |
| `MatchedFilterCore_AVX2` | `matched_filter_avx2.cc` | AVX2 | `matched_filter.rs` | **Missing** |
| `AdaptPartitions` | `adaptive_fir_filter.cc` | SSE2, NEON inline | `adaptive_fir_filter.rs` | **Scalar only** |
| `AdaptPartitions_Avx2` | `adaptive_fir_filter_avx2.cc` | AVX2 | `adaptive_fir_filter.rs` | **Missing** |
| `ApplyFilter` | `adaptive_fir_filter.cc` | SSE2, NEON inline | `adaptive_fir_filter.rs` | **Scalar only** |
| `ApplyFilter_Avx2` | `adaptive_fir_filter_avx2.cc` | AVX2 | `adaptive_fir_filter.rs` | **Missing** |
| `ComputeFrequencyResponse` | `adaptive_fir_filter.cc` | SSE2, NEON inline | `adaptive_fir_filter.rs` | **Scalar only** |
| `ComputeFrequencyResponse_Avx2` | `adaptive_fir_filter_avx2.cc` | AVX2 | `adaptive_fir_filter.rs` | **Missing** |
| `ComputeErl` | `adaptive_fir_filter_erl.cc` | SSE2, NEON inline | `adaptive_fir_filter_erl.rs` | **Scalar only** |
| `ComputeErl` (AVX2) | `adaptive_fir_filter_erl_avx2.cc` | AVX2 | `adaptive_fir_filter_erl.rs` | **Missing** |
| `VectorMath::{Sqrt,Multiply,Accumulate}` | `vector_math.h`, `vector_math_avx2.cc` | SSE2, AVX2, NEON | `vector_math.rs` | **Done** (via webrtc-simd) |
| `FftData::Spectrum` | `fft_data.h`, `fft_data_avx2.cc` | SSE2, AVX2, NEON | `fft_data.rs` | **Done** (via webrtc-simd) |

### Common Audio SIMD

| C++ Function | C++ File | ISAs | Rust Crate | Status |
|---|---|---|---|---|
| `SincResampler::Convolve` | `sinc_resampler_{sse,avx2,neon}.cc` | SSE2, AVX2, NEON | webrtc-common-audio | **Done** (via webrtc-simd) |
| `OouraFft` inner functions | `ooura_fft_{sse2,neon}.cc` | SSE2, NEON | webrtc-fft | **Done** (dedicated SIMD) |
| `FIRFilter::{Filter}` | `fir_filter_{sse,avx2,neon}.cc` | SSE2, AVX2, NEON | — | **Not needed** (no consumer) |

### FFT SIMD

| C++ Function | C++ File | ISAs | Rust Crate | Status |
|---|---|---|---|---|
| Ooura 128-point | `ooura_fft_{sse2,neon}.cc` | SSE2, NEON | webrtc-fft | **Done** |
| PFFFT butterflies | `pffft.c` | SSE, NEON | webrtc-fft | **Scalar only** |

### AGC2 SIMD

| C++ Function | C++ File | ISAs | Rust Crate | Status |
|---|---|---|---|---|
| `VectorMath::DotProduct` | `rnn_vad/vector_math_avx2.cc` | AVX2 (SSE2/NEON via inline) | webrtc-simd | **Done** |

---

## Step-by-Step Plan

### Step 1: Migrate to `cpufeatures` (webrtc-simd)

**Scope:** Replace manual `is_x86_feature_detected!` with `cpufeatures` crate.

**Why:** `cpufeatures` provides:
- Atomic cached detection (one CPUID call, then atomic load)
- `new()` constructor pattern returning `Option<TokenType>` — proof of support
- Used by RustCrypto ecosystem (battle-tested)
- Simpler API, no raw `unsafe` dispatch needed for detection

**Changes:**

1. Add `cpufeatures` dependency to `webrtc-simd/Cargo.toml`
2. In `lib.rs`, replace `detect_backend()`:

```rust
// Before:
pub fn detect_backend() -> SimdBackend {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return SimdBackend::Avx2;
    }
    if is_x86_feature_detected!("sse2") {
        return SimdBackend::Sse2;
    }
    SimdBackend::Scalar
}

// After:
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(avx2_fma_token, "avx2", "fma");
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(sse2_token, "sse2");

pub fn detect_backend() -> SimdBackend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if avx2_fma_token::new().is_some() {
            return SimdBackend::Avx2;
        }
        if sse2_token::new().is_some() {
            return SimdBackend::Sse2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return SimdBackend::Neon; // always available
    }
    #[cfg(not(any(
        target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"
    )))]
    SimdBackend::Scalar
}
```

3. Remove `force-scalar`, `force-sse2`, `force-avx2`, `force-neon` feature flags
   - If testing needs scalar fallback, use `RUSTFLAGS` to disable target features
4. Update all consumers — no API change needed, `detect_backend()` signature unchanged

**Tests:** Existing tests pass unchanged. Add test that `detect_backend()` returns a non-Scalar backend on CI.

**Commit:** `refactor(simd): migrate CPU detection to cpufeatures crate`

---

### Step 2: Add AEC3 SIMD Operations to webrtc-simd

**Scope:** Add new operations needed by AEC3 matched filter, adaptive filter, and ERL computation. These operations are more complex than the existing generic ones — they involve cross-correlation with circular buffer indexing and frequency-domain complex arithmetic.

**Analysis:** The AEC3 SIMD functions are **not generic operations** like dot_product. They are algorithm-specific:

- `MatchedFilterCore` = NLMS cross-correlation with circular buffer indexing, filter update, and optional accumulated error tracking
- `AdaptPartitions` = frequency-domain complex multiply-accumulate across filter partitions
- `ApplyFilter` = frequency-domain complex multiply-accumulate for filter output
- `ComputeFrequencyResponse` = compute |H(f)|² from complex filter taps
- `ComputeErl` = element-wise min across frequency response partitions

**Decision:** These are NOT suitable for `webrtc-simd` as generic operations. They should be **inline SIMD within the AEC3 crate** using `std::arch` intrinsics directly, with `SimdBackend` for dispatch. Only truly reusable operations belong in `webrtc-simd`.

However, some building blocks may be useful to add to `webrtc-simd`:
- `elementwise_min(a: &[f32], b: &[f32], out: &mut [f32])` — used by `ComputeErl`
- `complex_multiply_accumulate(x_re, x_im, h_re, h_im, acc_re, acc_im)` — used by `AdaptPartitions` and `ApplyFilter`

**New webrtc-simd operations:**

| Operation | Signature | Used By |
|---|---|---|
| `elementwise_min` | `(a: &[f32], b: &[f32], out: &mut [f32])` | `ComputeErl` |
| `complex_multiply_accumulate` | `(x_re: &[f32], x_im: &[f32], h_re: &[f32], h_im: &[f32], acc_re: &mut [f32], acc_im: &mut [f32])` | `AdaptPartitions`, `ApplyFilter` |

Implement in all 4 backends (Scalar, SSE2, AVX2, NEON). Add scalar-vs-SIMD comparison tests.

**Commit:** `feat(simd): add elementwise_min and complex_multiply_accumulate operations`

---

### Step 3: SIMD MatchedFilterCore (AEC3)

**Scope:** Add SSE2, AVX2, and NEON paths to `matched_filter_core()` in `crates/webrtc-aec3/src/matched_filter.rs`.

**C++ reference:** `webrtc/modules/audio_processing/aec3/matched_filter.cc` (SSE2 + NEON inline, ~200 lines each) and `matched_filter_avx2.cc` (~180 lines).

**Algorithm:** For each sample in the sub-block:
1. Compute `x2_sum = Σ x[k]²` and `s = Σ h[k] * x[k]` (circular buffer indexing)
2. Compute error `e = y[i] - s`
3. If `x2_sum > threshold` and no saturation: update `h[k] += smoothing * e / x2_sum * x[k]`
4. Optionally accumulate error for pre-echo detection

**SIMD strategy:**
- Process 4 (SSE2/NEON) or 8 (AVX2) filter taps per iteration
- Accumulate `x2_sum` and `s` in vector registers
- Horizontal reduction at the end of inner loop
- Filter update: vectorized multiply-add
- Circular buffer indexing needs careful handling at buffer boundaries

**Implementation approach:**
1. Add `SimdBackend` field to `MatchedFilter` struct (plumb from `detect_backend()`)
2. Create `matched_filter_simd.rs` module with SSE2/AVX2/NEON inner functions
3. Dispatch from `matched_filter_core()` based on backend
4. Keep existing scalar as fallback

**C++ intrinsics used (SSE2 path):**
- `_mm_loadu_ps`, `_mm_storeu_ps` — unaligned load/store
- `_mm_mul_ps`, `_mm_add_ps` — multiply, add
- `_mm_set1_ps` — broadcast scalar
- Horizontal sum via `_mm_movehl_ps` + `_mm_shuffle_ps` + `_mm_add_ps`

**C++ intrinsics used (AVX2 path):**
- `_mm256_loadu_ps`, `_mm256_storeu_ps` — 256-bit load/store
- `_mm256_fmadd_ps` — fused multiply-add (critical for performance)
- `_mm256_hadd_ps`, `_mm256_permutevar8x32_ps` — horizontal reduction
- `_mm256_extractf128_ps` — extract 128-bit lanes

**C++ intrinsics used (NEON path):**
- `vld1q_f32`, `vst1q_f32` — load/store
- `vmlaq_f32` — multiply-accumulate
- `vpadd_f32`, `vget_lane_f32` — horizontal reduction

**All have direct `std::arch` Rust equivalents.** No inline assembly needed.

**Tests:**
- Existing `matched_filter_core_converges` test validates correctness
- Add `matched_filter_core_simd_matches_scalar` — run both paths, compare output within tolerance
- Add test with various filter sizes (aligned and unaligned to SIMD width)

**Commit:** `feat(aec3): add SIMD paths for MatchedFilterCore (SSE2/AVX2/NEON)`

---

### Step 4: SIMD AdaptiveFilter Operations (AEC3)

**Scope:** Add SSE2, AVX2, and NEON paths to `compute_frequency_response()`, `adapt_partitions()`, and `apply_filter()` in `crates/webrtc-aec3/src/adaptive_fir_filter.rs`.

**C++ reference:**
- `adaptive_fir_filter.cc` — SSE2 + NEON inline (~300 lines each, 3 functions × 2 ISAs)
- `adaptive_fir_filter_avx2.cc` — AVX2 (~250 lines, 3 functions)

**Functions:**

#### 4a. `compute_frequency_response` (H² computation)
- For each partition: `H2[k] = max(H2[k], re[k]² + im[k]²)`
- SIMD: vectorized multiply + max
- SSE2: `_mm_mul_ps`, `_mm_add_ps`, `_mm_max_ps`
- AVX2: `_mm256_mul_ps`, `_mm256_fmadd_ps`, `_mm256_max_ps`
- NEON: `vmulq_f32`, `vmlaq_f32`, `vmaxq_f32`

#### 4b. `adapt_partitions` (frequency-domain filter update)
- For each partition: complex multiply `X * G`, accumulate into filter H
- `H_re += X_re * G_re + X_im * G_im`
- `H_im += X_re * G_im - X_im * G_re`
- SIMD: vectorized complex multiply-accumulate (uses `complex_multiply_accumulate` from Step 2, or inline)
- AVX2 uses FMA: `_mm256_fmadd_ps`, `_mm256_fnmadd_ps`

#### 4c. `apply_filter` (frequency-domain filter application)
- For each partition: complex multiply `X * H`, accumulate into output S
- Same structure as adapt_partitions but different source/dest
- SIMD: identical intrinsic pattern

**Implementation approach:**
1. Create `adaptive_fir_filter_simd.rs` module with platform-specific functions
2. Add `SimdBackend` field to `AdaptiveFirFilter` (plumb through constructor)
3. Dispatch in the three public methods

**Tests:**
- Add `compute_frequency_response_simd_matches_scalar`
- Add `adapt_partitions_simd_matches_scalar`
- Add `apply_filter_simd_matches_scalar`
- Use randomized FftData inputs via proptest

**Commit:** `feat(aec3): add SIMD paths for AdaptiveFirFilter (SSE2/AVX2/NEON)`

---

### Step 5: SIMD ComputeErl (AEC3)

**Scope:** Add SIMD to `compute_erl()` in `crates/webrtc-aec3/src/adaptive_fir_filter_erl.rs`.

**C++ reference:**
- `adaptive_fir_filter_erl.cc` — SSE2 + NEON inline (~30 lines each)
- `adaptive_fir_filter_erl_avx2.cc` — AVX2 (~30 lines)

**Algorithm:** For each frequency bin, compute `erl[k] = min over partitions of H2[partition][k]`.

**SIMD approach:** Use `elementwise_min` from webrtc-simd (Step 2), or inline:
- SSE2: `_mm_min_ps`
- AVX2: `_mm256_min_ps`
- NEON: `vminq_f32`

**Implementation:** Add `SimdBackend` parameter to `compute_erl()`, dispatch to platform-specific inner functions.

**Tests:**
- Add `compute_erl_simd_matches_scalar` with randomized H2 data

**Commit:** `feat(aec3): add SIMD paths for ComputeErl (SSE2/AVX2/NEON)`

---

### Step 6: SIMD PFFFT (webrtc-fft)

**Scope:** Add SSE and NEON 4-wide butterfly operations to PFFFT in `crates/webrtc-fft/src/pffft.rs`.

**C++ reference:** `webrtc/third_party/pffft/src/pffft.c` — uses macro-based SIMD abstraction:
```c
// The C code defines a set of macros (VMUL, VADD, VSUB, INTERLEAVE2, etc.)
// that expand to SSE, NEON, or scalar operations depending on platform.
// SIMD_SZ = 4 for SSE/NEON, 1 for scalar.
```

**PFFFT SIMD macros (C → Rust mapping):**

| C Macro | SSE Intrinsic | NEON Intrinsic | Rust `std::arch` |
|---|---|---|---|
| `VMUL(a,b)` | `_mm_mul_ps` | `vmulq_f32` | Direct |
| `VADD(a,b)` | `_mm_add_ps` | `vaddq_f32` | Direct |
| `VSUB(a,b)` | `_mm_sub_ps` | `vsubq_f32` | Direct |
| `LD_PS1(v)` | `_mm_set1_ps` | `vdupq_n_f32` | Direct |
| `VMADD(a,b,c)` | `a*b+c` | `vmlaq_f32` | Direct |
| `INTERLEAVE2(a,b,c,d)` | `_mm_unpacklo_ps`/`hi` | `vtrn1q_f32`/`2` | Direct |
| `UNINTERLEAVE2(a,b,c,d)` | `_mm_shuffle_ps` | `vuzp1q_f32`/`2` | Direct |
| `VSWAPHL(a,b)` | `_mm_shuffle_ps(b,a,...)` | `vcombine_f32(vget_high/low)` | Direct |
| `VALIGNED(p)` | `((uintptr)p & 0xF) == 0` | N/A | Pointer alignment check |
| `VCPLXMUL(ar,ai,br,bi)` | Complex multiply via shuffles | Direct ops | Direct |

**Implementation approach:**

The Rust PFFFT scalar code processes 1 float at a time (`SIMD_SZ=1`). The SIMD version processes 4 at a time (`SIMD_SZ=4`). This changes array indexing and loop bounds throughout the code.

Strategy:
1. Create a `SimdVec4` trait (internal to webrtc-fft) abstracting the 4-wide operations
2. Implement for `f32` (scalar, processes 1 element), `__m128` (SSE), `float32x4_t` (NEON)
3. Make the FFT core generic over `SimdVec4`
4. Runtime dispatch in `Pffft::new()` to select implementation

Alternative (simpler): duplicate the core functions with SIMD intrinsics, like Ooura FFT does.

**Recommendation:** Use the duplication approach (match Ooura pattern). The PFFFT code is ~1800 lines scalar; SIMD versions will be similar length but with vectorized operations. A trait abstraction would save code but add complexity. Since Ooura already uses the duplication pattern successfully, stay consistent.

**Files:**
- `pffft_sse.rs` — SSE path (~1800 lines)
- `pffft_neon.rs` — NEON path (~1800 lines)
- Runtime dispatch in `pffft.rs` constructor and `forward`/`backward`/`convolve_accumulate`

**Tests:**
- Existing scalar tests validate correctness
- Add `pffft_simd_matches_scalar` — compare SIMD output against scalar for various FFT sizes

**Commit:** `feat(fft): add SIMD paths for PFFFT (SSE/NEON)`

**Note:** This is the largest single step (~3600 lines of SIMD code). Consider splitting into two commits: SSE first, NEON second.

---

### Step 7: Plumb SimdBackend Through AEC3 Pipeline

**Scope:** Thread `SimdBackend` from `detect_backend()` at construction time through the AEC3 object graph, replacing any `detect_backend()` calls in hot paths.

**Current state:** Some AEC3 components call `detect_backend()` per-frame (e.g., `FftData::spectrum` calls it inline). This should be cached at construction time.

**Changes:**
1. Add `backend: SimdBackend` field to key AEC3 structs:
   - `MatchedFilter` (currently scalar)
   - `AdaptiveFirFilter` (currently scalar)
   - `EchoRemover` (passes to children)
   - `BlockProcessor` (passes to children)
   - `EchoCanceller3` wrapper in webrtc-apm
2. Pass `detect_backend()` result at construction time
3. Update `FftData::spectrum` to take `SimdBackend` parameter (or store in struct)
4. Verify no `detect_backend()` calls remain in hot paths

**Tests:** Existing tests pass. No behavior change.

**Commit:** `refactor(aec3): plumb SimdBackend through AEC3 pipeline`

---

### Step 8: SIMD Verification Tests

**Scope:** Comprehensive cross-backend verification for all SIMD paths.

**Test categories:**

1. **Scalar vs SIMD comparison** (per operation):
   - Run identical inputs through scalar and detected SIMD backend
   - Compare outputs within tolerance (1e-3 for accumulated ops, 1e-6 for elementwise)
   - Test edge cases: input length 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64

2. **Cross-backend consistency** (when multiple backends available):
   - On x86_64: compare SSE2 vs AVX2 results
   - Verify they match within floating-point tolerance

3. **Proptest randomized verification**:
   - `MatchedFilterCore`: random filter coefficients, random input signal
   - `AdaptPartitions`: random FftData partitions
   - `ComputeFrequencyResponse`: random filter data
   - `ComputeErl`: random H2 partitions
   - PFFFT: random input, verify forward→inverse roundtrip

4. **End-to-end integration** (in webrtc-apm):
   - Process identical audio with `force-scalar` and default backend
   - Compare output samples within tolerance
   - Verify AEC3 echo cancellation quality is preserved

**Commit:** `test(simd): comprehensive SIMD vs scalar verification tests`

---

## Step Summary

| Step | Component | Scope | New Lines (est.) |
|---|---|---|---|
| 1 | webrtc-simd | cpufeatures migration | ~50 |
| 2 | webrtc-simd | elementwise_min, complex_multiply_accumulate | ~400 |
| 3 | webrtc-aec3 | MatchedFilterCore SSE2/AVX2/NEON | ~600 |
| 4 | webrtc-aec3 | AdaptiveFirFilter SSE2/AVX2/NEON (3 functions) | ~900 |
| 5 | webrtc-aec3 | ComputeErl SSE2/AVX2/NEON | ~150 |
| 6 | webrtc-fft | PFFFT SSE/NEON | ~3600 |
| 7 | webrtc-aec3 | SimdBackend plumbing | ~100 |
| 8 | all | Verification tests | ~500 |
| **Total** | | | **~6300** |

---

## Inline Assembly Assessment

**Conclusion: NOT needed.**

All C++ SIMD code uses standard intrinsics with direct `std::arch` Rust equivalents:

| Intrinsic Category | C++ | Rust `std::arch` | Gap |
|---|---|---|---|
| Load/Store | `_mm_loadu_ps`, `_mm256_loadu_ps`, `vld1q_f32` | Direct mapping | None |
| Arithmetic | `_mm_mul_ps`, `_mm_add_ps`, `_mm_sub_ps` | Direct mapping | None |
| FMA | `_mm256_fmadd_ps`, `vmlaq_f32` | Direct mapping | None |
| Horizontal | `_mm_hadd_ps`, `vpadd_f32` | Direct mapping | None |
| Comparison | `_mm_max_ps`, `_mm_min_ps`, `vmaxq_f32` | Direct mapping | None |
| Shuffle | `_mm_shuffle_ps`, `_mm_unpacklo_ps` | Direct mapping | None |
| Sqrt | `_mm_sqrt_ps`, `vsqrtq_f32` | Direct mapping | None |
| NEON FMA | `vmlaq_f32` (always available on aarch64) | Direct mapping | None |
| NEON reciprocal sqrt | `vrsqrteq_f32`, `vrsqrtsq_f32` | Direct mapping | None |
| NEON interleave | `vtrn1q_f32`, `vuzp1q_f32`, `vcombine_f32` | Direct mapping | None |

The C++ codebase has 3 ARM assembly files (`*.S`) but they are **only used by excluded modules** (AECM, SPL). The modern pipeline (AEC3, AGC2, NS) uses zero inline assembly.

Rust `std::arch` intrinsics compile to identical machine code as C++ intrinsics (verified by comparing assembly output for existing operations in webrtc-simd).

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SIMD floating-point precision differences | Low | Medium | Tolerance-based comparison in tests |
| PFFFT SIMD port complexity (1800 lines × 2) | Medium | Medium | Port SSE first, validate, then NEON |
| AVX2 vzeroupper penalty on transitions | Low | Low | Rust compiler handles automatically |
| NEON FMA precision vs SSE2 non-FMA | Low | Low | Match C++ behavior per-platform |
| cpufeatures doesn't support all platforms | Very Low | Low | Fallback to Scalar is always available |
