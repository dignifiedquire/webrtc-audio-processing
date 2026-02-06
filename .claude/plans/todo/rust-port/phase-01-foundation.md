# Phase 1: Foundation Infrastructure

**Status:** Complete
**Actual Duration:** ~1 week
**Dependencies:** None (first phase)
**Outcome:** A buildable Rust workspace with SIMD abstractions, cxx-based C++ FFI for comparison testing, proptest harness with test-strategy derive macros, and tracing. 34 Rust tests passing.

---

## Overview

Set up the complete Rust workspace structure, build system integration, SIMD abstraction layer, and property-based testing infrastructure. This phase produces no audio processing logic but establishes all the scaffolding needed for incremental porting.

## Technology Decisions (Finalized)

| Concern | Decision | Notes |
|---------|----------|-------|
| Rust edition | 2024 | Resolver 3, unsafe-op-in-unsafe-fn by default |
| MSRV | 1.91 | |
| Formatting | Default rustfmt (no custom rustfmt.toml) | |
| C++ interop | `cxx` crate with C++ shim layer | Feature-gated (`cxx-bridge`) |
| SIMD | `SimdBackend` enum + `std::arch` intrinsics | Not trait objects — enum is Copy + Eq |
| Logging | `tracing` crate, direct dependency | NOT feature-gated |
| Property testing | `proptest` + `test-strategy` derive macros | `#[derive(Arbitrary)]`, `#[proptest]` |
| Test runner | `cargo nextest` | NOT `cargo test` |
| C header generation | `cbindgen` (Phase 9) | |
| FFT | `pffft.c` via `cc` crate (Phase 2) | |

## Workspace Lints

```toml
[workspace.lints.rust]
unexpected_cfgs = { level = "deny", check-cfg = ["cfg(docsrs)"] }
elided_lifetimes_in_paths = "warn"
unnameable_types = "warn"
unreachable_pub = "warn"
missing_debug_implementations = "warn"

[workspace.lints.clippy]
or_fun_call = "warn"
use_self = "warn"
unused_async = "warn"
absolute_paths = "warn"
manual_let_else = "warn"
allow_attributes_without_reason = "warn"
mod_module_files = "deny"
```

All crates inherit via `[lints] workspace = true`.

## Commits (Actual)

| # | Commit Message | What's Included |
|---|---------------|-----------------|
| 1 | `feat(rust): initialize workspace with empty crate stubs` | 10 crates, Cargo.tomls, lib.rs stubs, .gitignore |
| 2 | `ci: add rustfmt and clippy configuration` | rustfmt.toml, clippy.toml |
| 3 | `feat(rust): add SIMD abstraction layer with fallback, SSE2, AVX2, NEON` | SimdOps trait, 4 backends, 15 tests |
| 4 | `feat(rust): add cxx-based C++ FFI bridge for comparison testing` | webrtc-apm-sys with shim, feature-gated |
| 5 | `chore: use Rust edition 2024, MSRV 1.85, remove custom rustfmt.toml` | Edition 2024 unsafe-fn fixes |
| 6 | `feat(rust): add proptest framework with audio generators and comparison utilities` | generators.rs, comparison.rs, 17 tests |
| 7 | `refactor(rust): use SimdBackend enum instead of trait objects, make tracing a direct dependency` | Enum dispatch, tracing non-optional |
| 8 | `refactor(rust): use test-strategy derive macros for proptest` | #[derive(Arbitrary)] types, #[proptest] |
| 9 | `chore: add workspace lints, resolver 3, MSRV 1.91` | All lints, pub(crate) fixes |
| 10 | `chore: deny unexpected_cfgs, verify cargo nextest` | unexpected_cfgs = deny |
| 11 | `docs: configure docs.rs for feature-gated items` | docsrs cfg, metadata.docs.rs |

## Key Design: SimdBackend Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

// Methods: dot_product, dual_dot_product, multiply_accumulate, sum
// Runtime detection: detect_backend() -> SimdBackend
```

Backend modules export `pub(crate)` free functions. The enum dispatches via `match`.

## Key Design: Proptest with test-strategy

```rust
#[derive(Debug, Clone, Arbitrary)]
pub struct MonoFrameF32 {
    pub sample_rate: SampleRate,
    #[strategy(audio_frame_f32(#sample_rate.hz()))]
    pub samples: Vec<f32>,
}

#[proptest]
fn process_preserves_length(frame: MonoFrameF32) {
    // test body
}
```

Types: `SampleRate` (enum), `ChannelCount` (enum), `MonoFrameF32`, `MonoFrameI16`, `MultiChannelFrameF32`.

## Key Design: Feature-Gated FFI

- `webrtc-apm-sys` has `cxx-bridge` feature (requires pre-built C++ library)
- `webrtc-apm-proptest` has `cpp-comparison` feature (enables FFI comparison tests)
- Both have `package.metadata.docs.rs` with `all-features = true` and `--cfg docsrs`
- Feature-gated items annotated with `#[cfg_attr(docsrs, doc(cfg(...)))]`

## Verification

- [x] `cargo build --workspace` succeeds
- [x] `cargo nextest run` passes (34 tests: 16 SIMD + 18 proptest)
- [x] `cargo clippy --all-targets` clean (zero warnings)
- [x] C++ `meson test -C builddir` still passes
- [x] SIMD dispatch works (NEON on aarch64, SSE2/AVX2 on x86_64)
- [x] Scalar fallback matches SIMD output (within f32 tolerance)
- [x] `unexpected_cfgs = deny` catches typos
