# Phase 1: Foundation Infrastructure

**Status:** Not Started
**Estimated Duration:** 2-3 weeks
**Dependencies:** None (first phase)
**Outcome:** A buildable Rust workspace with SIMD abstractions, cxx-based C++ FFI for comparison testing, and a proptest harness. All C++ tests still pass. The Rust crates compile but contain no audio logic yet.

---

## Overview

Set up the complete Rust workspace structure, build system integration, SIMD abstraction layer, and property-based testing infrastructure. This phase produces no audio processing logic but establishes all the scaffolding needed for incremental porting.

## Prerequisites

- Rust toolchain (stable, latest)
- Existing C++ project builds and all 2432 tests pass
- `meson`, `ninja`, `pkg-config` installed

## Key Technology Decisions

| Concern | Decision | Rationale |
|---------|----------|-----------|
| C++ interop for testing | `cxx` crate with C++ shim layer | AudioProcessing is abstract; cxx needs concrete free functions. Shims call virtual methods. |
| SIMD | `std::arch` intrinsics + runtime dispatch | Stable Rust, bit-exact with C++ intrinsics, no nightly needed |
| Logging | `tracing` crate, feature-gated | Industry standard, optional dependency for library crates |
| Property testing | `proptest` | Composable strategies, good f32 range support |
| C header generation | `cbindgen` | Battle-tested (Firefox), generates from `#[repr(C)]` + `extern "C"` |
| FFT | Compile `pffft.c` via `cc` crate | Bit-exact with upstream; source already in `webrtc/third_party/pffft/` |

---

## Tasks

### 1.1 Workspace Setup

Create the Cargo workspace with all crate stubs.

**Files to create:**

```
crates/
  Cargo.toml                    # Workspace root (at repo root level)
  webrtc-apm/Cargo.toml         # Main public crate (C API)
  webrtc-apm/src/lib.rs
  webrtc-apm-sys/Cargo.toml     # cxx-based C++ bindings for testing
  webrtc-apm-sys/src/lib.rs
  webrtc-apm-sys/build.rs
  webrtc-common-audio/Cargo.toml
  webrtc-common-audio/src/lib.rs
  webrtc-aec3/Cargo.toml
  webrtc-aec3/src/lib.rs
  webrtc-aecm/Cargo.toml
  webrtc-aecm/src/lib.rs
  webrtc-agc/Cargo.toml
  webrtc-agc/src/lib.rs
  webrtc-ns/Cargo.toml
  webrtc-ns/src/lib.rs
  webrtc-vad/Cargo.toml
  webrtc-vad/src/lib.rs
  webrtc-simd/Cargo.toml
  webrtc-simd/src/lib.rs
  webrtc-apm-proptest/Cargo.toml
  webrtc-apm-proptest/src/lib.rs
```

**Workspace Cargo.toml structure:**
```toml
[workspace]
resolver = "2"
members = [
  "crates/webrtc-apm",
  "crates/webrtc-apm-sys",
  "crates/webrtc-common-audio",
  "crates/webrtc-aec3",
  "crates/webrtc-aecm",
  "crates/webrtc-agc",
  "crates/webrtc-ns",
  "crates/webrtc-vad",
  "crates/webrtc-simd",
  "crates/webrtc-apm-proptest",
]

[workspace.dependencies]
# Shared dependency versions
tracing = { version = "0.1", optional = true }
proptest = "1.9"
cxx = "1.0"
cxx-build = "1.0"
cbindgen = "0.29"
cc = "1.0"
```

**Verification:**
- [ ] `cargo build` succeeds for all crates
- [ ] `cargo test` runs (no tests yet, but compiles)
- [ ] C++ `meson test -C builddir` still passes all 2432 tests

**Commit:** `feat(rust): initialize workspace with empty crate stubs`

---

### 1.2 CI Configuration

Set up CI that enforces both C++ and Rust builds.

**CI jobs:**
1. `build-cpp` - Meson build + test (existing)
2. `build-rust` - `cargo build --workspace`
3. `test-rust` - `cargo test --workspace`
4. `clippy` - `cargo clippy --workspace -- -D warnings`
5. `fmt` - `cargo fmt --all -- --check`

**Key rule:** CI must never be broken. Every commit must pass both C++ and Rust builds.

**Files to create:**
- `.github/workflows/rust.yml` (or equivalent for GitLab CI)
- `rustfmt.toml` (consistent formatting)
- `clippy.toml` (lint configuration)

**Verification:**
- [ ] CI pipeline runs and passes
- [ ] Both C++ and Rust builds are gated

**Commit:** `ci: add Rust build and test pipeline`

---

### 1.3 SIMD Abstraction Layer (`webrtc-simd`)

Port the SIMD dispatch pattern used in the C++ codebase. The C++ code compiles AVX2 code into separate translation units with `-mavx2 -mfma` flags and uses runtime detection.

**Architecture:**

```rust
// webrtc-simd/src/lib.rs

/// Trait for vectorized math operations used across audio processing.
/// Each method has a scalar fallback and optional SIMD acceleration.
pub trait SimdOps: Send + Sync {
    fn accumulate(&self, x: &[f32], y: &mut [f32]);
    fn multiply_accumulate(&self, x: &[f32], y: &[f32], z: &mut [f32]);
    fn sqrt_vec(&self, x: &mut [f32]);
    fn dot_product(&self, x: &[f32], y: &[f32]) -> f32;
    // Additional ops added as needed during porting
}
```

**Files to create:**
```
webrtc-simd/src/
  lib.rs         # SimdOps trait, get_simd_ops() dispatch function
  fallback.rs    # Scalar implementations
  sse2.rs        # SSE2 implementations (cfg(target_arch = "x86_64"))
  avx2.rs        # AVX2+FMA implementations
  neon.rs        # NEON implementations (cfg(target_arch = "aarch64"))
  detect.rs      # Runtime CPU feature detection wrapper
```

**Runtime dispatch pattern:**
```rust
pub fn get_simd_ops() -> &'static dyn SimdOps {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

**Cargo.toml features:**
```toml
[features]
default = []
# Force specific backend (for testing/benchmarking)
force-scalar = []
force-sse2 = []
force-avx2 = []
force-neon = []
```

**Important notes:**
- Start with `SimdOps` trait containing only the operations needed for Phase 2 (FIR filter, sinc resampler)
- Grow the trait incrementally as phases demand new operations
- Each SIMD function must have a unit test comparing its output against the scalar fallback
- Use `#[target_feature(enable = "avx2,fma")]` on AVX2 functions

**Verification:**
- [ ] `cargo test -p webrtc-simd` passes
- [ ] Tests run on both x86_64 (SSE2/AVX2) and aarch64 (NEON) if available
- [ ] Scalar fallback produces identical results to SIMD paths (within f32 tolerance)
- [ ] `is_x86_feature_detected!` dispatches correctly at runtime

**Commit:** `feat(rust): add SIMD abstraction layer with fallback, SSE2, AVX2, NEON`

---

### 1.4 C++ FFI Bridge (`webrtc-apm-sys`)

Create cxx-based bindings to the existing C++ library. Since `AudioProcessing` is an abstract class with virtual methods, we write a thin C++ shim layer that exposes concrete free functions.

**C++ shim files (new):**
```
crates/webrtc-apm-sys/
  cpp/
    shim.h        # C++ shim declarations
    shim.cc       # C++ shim implementations
```

**Shim pattern:**
```cpp
// shim.h - Thin wrappers around virtual AudioProcessing methods
#pragma once
#include <memory>
#include <cstdint>
#include "rust/cxx.h"

namespace webrtc_shim {

struct ApmHandle;  // Opaque handle to AudioProcessing

// Creation/destruction
std::unique_ptr<ApmHandle> create_apm();
void destroy_apm(std::unique_ptr<ApmHandle> handle);

// Processing (for comparison testing)
int32_t process_stream_i16(
    ApmHandle& handle,
    rust::Slice<const int16_t> src,
    int32_t input_sample_rate, size_t input_channels,
    int32_t output_sample_rate, size_t output_channels,
    rust::Slice<int16_t> dest);

int32_t process_stream_f32(
    ApmHandle& handle,
    rust::Slice<const float> src,  // interleaved for simplicity
    int32_t input_sample_rate, size_t input_channels,
    int32_t output_sample_rate, size_t output_channels,
    rust::Slice<float> dest);

// Individual component functions (for per-module comparison testing)
// These will be added incrementally as each phase needs them
void fir_filter_f32(rust::Slice<const float> coeffs,
                    rust::Slice<const float> input,
                    rust::Slice<float> output);

} // namespace webrtc_shim
```

**build.rs:**
```rust
fn main() {
    // Build C++ shim
    cxx_build::bridge("src/lib.rs")
        .file("cpp/shim.cc")
        .include("../../webrtc")  // Access to C++ headers
        .flag_if_supported("-std=c++20")
        .compile("webrtc_apm_shim");

    // Link against the existing C++ library
    println!("cargo:rustc-link-lib=webrtc-audio-processing-3");
    println!("cargo:rustc-link-search=native=../../builddir");
}
```

**Key design decisions:**
- Shim functions are added incrementally (start minimal, grow per phase)
- Each audio processing component gets its own shim function for isolated comparison
- The shim wraps the actual `AudioProcessing` implementation, not mocks
- build.rs links against the Meson-built C++ library

**Verification:**
- [ ] `cargo build -p webrtc-apm-sys` succeeds
- [ ] Can call `create_apm()` from Rust and get a valid handle
- [ ] Can round-trip a simple audio buffer through C++ `ProcessStream`
- [ ] Memory is properly managed (no leaks in valgrind/ASAN)

**Commit:** `feat(rust): add cxx-based C++ FFI bridge for comparison testing`

---

### 1.5 Property Test Framework (`webrtc-apm-proptest`)

Set up proptest infrastructure with audio-specific generators.

**Files to create:**
```
webrtc-apm-proptest/src/
  lib.rs          # Re-exports, common utilities
  generators.rs   # Audio buffer generators
  comparison.rs   # Float comparison utilities
```

**Audio generators:**
```rust
// generators.rs
use proptest::prelude::*;

/// Generate a mono audio frame at a given sample rate
pub fn audio_frame_f32(sample_rate: u32) -> impl Strategy<Value = Vec<f32>> {
    let frame_size = (sample_rate / 100) as usize; // ~10ms
    proptest::collection::vec(-1.0f32..=1.0f32, frame_size..=frame_size)
}

/// Generate interleaved stereo audio
pub fn stereo_frame_f32(sample_rate: u32) -> impl Strategy<Value = Vec<f32>> {
    let frame_size = (sample_rate / 100) as usize * 2;
    proptest::collection::vec(-1.0f32..=1.0f32, frame_size..=frame_size)
}

/// Generate i16 audio frame
pub fn audio_frame_i16(sample_rate: u32) -> impl Strategy<Value = Vec<i16>> {
    let frame_size = (sample_rate / 100) as usize;
    proptest::collection::vec(i16::MIN..=i16::MAX, frame_size..=frame_size)
}

/// Generate FIR filter coefficients
pub fn fir_coefficients(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0f32..=1.0f32, 1..=max_len)
}

/// Generate a valid sample rate
pub fn sample_rate() -> impl Strategy<Value = u32> {
    prop_oneof![
        Just(8000u32),
        Just(16000u32),
        Just(32000u32),
        Just(48000u32),
    ]
}
```

**Float comparison utilities:**
```rust
// comparison.rs

/// Compare two f32 slices with absolute tolerance
pub fn assert_f32_near(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "Mismatch at index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

/// Compare two i16 slices (bit-exact)
pub fn assert_i16_exact(actual: &[i16], expected: &[i16]) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "Mismatch at index {i}: actual={a}, expected={e}");
    }
}

/// Compare with relative tolerance for larger values
pub fn assert_f32_relative(actual: &[f32], expected: &[f32], rel_tol: f32, abs_tol: f32) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = abs_tol.max(e.abs() * rel_tol);
        assert!(
            (a - e).abs() <= tol,
            "Mismatch at index {i}: actual={a}, expected={e}, diff={}, tol={tol}",
            (a - e).abs()
        );
    }
}
```

**Verification:**
- [ ] `cargo test -p webrtc-apm-proptest` passes
- [ ] Generators produce valid audio data (correct lengths, value ranges)
- [ ] Comparison utilities correctly detect mismatches
- [ ] At least one end-to-end proptest runs: generate audio -> process through C++ -> verify output is reasonable

**Commit:** `feat(rust): add proptest framework with audio generators and comparison utilities`

---

### 1.6 Tracing Integration

Set up feature-gated tracing across the workspace.

**Pattern for each crate's Cargo.toml:**
```toml
[features]
default = []
tracing = ["dep:tracing"]

[dependencies]
tracing = { workspace = true, optional = true }
```

**Pattern for each crate's code:**
```rust
// At crate root
#[cfg(feature = "tracing")]
use tracing::{debug, info, warn, error, instrument};

// Internal macro fallbacks when tracing is disabled
#[cfg(not(feature = "tracing"))]
macro_rules! debug { ($($tt:tt)*) => {} }
#[cfg(not(feature = "tracing"))]
macro_rules! info { ($($tt:tt)*) => {} }
#[cfg(not(feature = "tracing"))]
macro_rules! warn { ($($tt:tt)*) => {} }
#[cfg(not(feature = "tracing"))]
macro_rules! error { ($($tt:tt)*) => {} }
```

**Verification:**
- [ ] `cargo build --workspace` succeeds without tracing feature
- [ ] `cargo build --workspace --features tracing` succeeds
- [ ] No tracing code compiles into the binary when feature is off

**Commit:** `feat(rust): add feature-gated tracing support across workspace`

---

## Phase 1 Completion Checklist

- [ ] All Rust crates compile (`cargo build --workspace`)
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo test --workspace` passes (framework tests)
- [ ] C++ tests still pass (`meson test -C builddir` - 2432 tests)
- [ ] CI pipeline enforces all of the above
- [ ] SIMD dispatch works on current platform
- [ ] cxx bridge can call into C++ library
- [ ] proptest generators produce valid audio data
- [ ] All commits are atomic and reviewable

## Commit Summary

| Order | Commit Message | What's Included |
|-------|---------------|-----------------|
| 1 | `feat(rust): initialize workspace with empty crate stubs` | Cargo.toml files, empty lib.rs files |
| 2 | `ci: add Rust build and test pipeline` | CI config, rustfmt.toml, clippy.toml |
| 3 | `feat(rust): add SIMD abstraction layer with fallback, SSE2, AVX2, NEON` | webrtc-simd crate with all backends |
| 4 | `feat(rust): add cxx-based C++ FFI bridge for comparison testing` | webrtc-apm-sys crate with shim layer |
| 5 | `feat(rust): add proptest framework with audio generators and comparison utilities` | webrtc-apm-proptest crate |
| 6 | `feat(rust): add feature-gated tracing support across workspace` | tracing macros in all crates |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| cxx can't wrap some C++ types | Medium | Medium | Fall back to raw `extern "C"` FFI for those specific functions |
| SIMD intrinsics differ from C++ | Low | High | Compare intermediate values, not just final output |
| build.rs link issues with Meson output | Medium | Low | Use pkg-config or explicit library path |
| CI flakiness | Low | Medium | Pin toolchain versions, use deterministic builds |
