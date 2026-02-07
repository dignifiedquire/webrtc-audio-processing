# Rust Port Plan

**Goal:** Port WebRTC Audio Processing from C++20 to Rust, producing a modular multi-crate library with a C-compatible drop-in replacement API.

**Source:** WebRTC M145 (branch-heads/7632), version 3.0
**Scope:** 623 source files, 2432 tests, SSE2/AVX2/NEON SIMD optimizations

---

## Phase Overview

| Phase | Name | Duration | Commits | Dependencies | Status |
|-------|------|----------|---------|--------------|--------|
| 1 | [Foundation Infrastructure](phase-01-foundation.md) | ~1 week | 11 | None | **Complete** |
| 2 | [Common Audio Primitives](phase-02-common-audio.md) | ~3 weeks | 10 | Phase 1 | **Complete** (SIMD pending) |
| 4 | [Automatic Gain Control (AGC2)](phase-04-agc.md) | 3-4 weeks | 14 | Phase 2 | **Complete** |
| 5 | [Noise Suppression](phase-05-noise-suppression.md) | 2-3 weeks | 7 | Phase 2 | **Complete** |
| 6 | [Echo Cancellation (AEC3)](phase-06-echo-cancellation.md) | 6-8 weeks | 21 | Phase 2 | **Complete** (SIMD pending) |
| 7 | [Audio Processing Integration](phase-07-integration.md) | 3-4 weeks | 5 | Phases 2,4-6 | **Complete** |
| SIMD | [SIMD Optimization](phase-simd-optimization.md) | 3-4 weeks | 8 | Phases 1-7 | **Complete** |
| 8 | [C API & Final Integration](phase-08-c-api.md) | 2-3 weeks | 7 | Phase 7 | **Next** |
| 9 | [Documentation & Release](phase-09-docs-release.md) | 1-2 weeks | 6 | Phase 8 | Not Started |
| **Total** | | **~22-32 weeks** | **~90** | | |

**Excluded (not ported):** AECM (removed upstream M146), AGC1 (deprecated), SPL library (legacy fixed-point), Core VAD (SPL-dependent, modern pipeline uses AGC2's RNN VAD).
See [master plan](../rust-port.md) for rationale.

## Dependency Graph

```
Phase 1 (Foundation) ---- COMPLETE
  |
  v
Phase 2 (Common Audio) -- COMPLETE
  |
  +---> Phase 4 (AGC2) -- COMPLETE
  |
  +---> Phase 5 (NS) ---- COMPLETE
  |
  +---> Phase 6 (AEC3) -- COMPLETE
  |
  v
Phase 7 (Integration) --- COMPLETE
  |
  v
SIMD Optimization ------- COMPLETE
  |
  v
Phase 8 (C API) --------- NEXT
  |
  v
Phase 9 (Docs & Release)
```

Phase 3 (VAD) removed — core VAD depends on SPL (excluded), and the modern pipeline uses AGC2's RNN VAD instead.

## Technology Stack (Finalized in Phase 1)

| Concern | Choice | Notes |
|---------|--------|-------|
| Rust edition | 2024 | Resolver 3, MSRV 1.91 |
| C++ interop (testing) | `cxx` + C++ shim layer | Feature-gated `cxx-bridge` |
| SIMD | `SimdBackend` enum + `std::arch` intrinsics | Copy + Eq, not trait objects |
| Logging | `tracing` (direct dependency) | NOT feature-gated |
| Property testing | `proptest` + `test-strategy` | `#[derive(Arbitrary)]`, `#[proptest]` |
| Test runner | `cargo nextest` | NOT `cargo test` |
| C header generation | `cbindgen` | Phase 9 |
| FFT | Pure Rust `webrtc-fft` crate | Ooura 128, fft4g, PFFFT (scalar) |
| Benchmarking | `criterion` | |
| Docs | `docs.rs` with `--cfg docsrs` | `all-features = true` |

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

## Crate Architecture

```
webrtc-apm (main crate, C API)
  +-- webrtc-common-audio + tracing
  +-- webrtc-aec3 + webrtc-simd + webrtc-fft + tracing
  +-- webrtc-agc2 + webrtc-simd + webrtc-fft + tracing
  +-- webrtc-ns + webrtc-fft + tracing
  +-- webrtc-simd
  +-- webrtc-fft
  +-- webrtc-ring-buffer

Testing crates (publish = false):
  webrtc-apm-sys      -- cxx FFI to C++ (feature: cxx-bridge)
  webrtc-apm-proptest -- proptest + test-strategy generators
```

## Commit Discipline

Every commit must:
1. Compile (`cargo build --workspace`)
2. Pass all Rust tests (`cargo nextest run`)
3. Pass all C++ tests (`meson test -C builddir`)
4. Pass clippy (`cargo clippy --all-targets` — zero warnings)

Commits should be atomic and focused:
- One logical change per commit
- Working state at every commit
- Clear commit message with conventional commit prefix (`feat`, `fix`, `test`, `docs`, `bench`, `ci`, `chore`, `refactor`)

## Verification Strategy

Each ported component is verified at three levels:

1. **Unit tests** - Mirror the C++ unit test behaviors in Rust
2. **Property tests** - Random input comparison via `proptest` + `test-strategy` + cxx FFI to C++ reference
3. **Integration tests** - End-to-end C++ test suite (Phase 9) validates complete pipeline

For SIMD code, additionally:
4. **Cross-variant** - SIMD output compared against scalar fallback

## Original Master Plan

The original comprehensive plan is preserved at:
[`../rust-port.md`](../rust-port.md)

These per-phase plans expand on it with:
- Exact file-to-file porting maps
- Per-commit scope and messages
- Verification checklists
- Risk assessments
- C++ shim additions needed
- Proptest strategies
