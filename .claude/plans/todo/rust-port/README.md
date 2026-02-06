# Rust Port Plan

**Goal:** Port WebRTC Audio Processing from C++20 to Rust, producing a modular multi-crate library with a C-compatible drop-in replacement API.

**Source:** WebRTC M145 (branch-heads/7632), version 3.0
**Scope:** 623 source files, 2432 tests, SSE2/AVX2/NEON SIMD optimizations

---

## Phase Overview

| Phase | Name | Duration | Commits | Dependencies | Status |
|-------|------|----------|---------|--------------|--------|
| 1 | [Foundation Infrastructure](phase-01-foundation.md) | ~1 week | 11 | None | **Complete** |
| 2 | [Common Audio Primitives](phase-02-common-audio.md) | 3-4 weeks | 20 | Phase 1 | Not Started |
| 3 | [Voice Activity Detection](phase-03-vad.md) | 2 weeks | 5 | Phase 2 | Not Started |
| 4 | [Automatic Gain Control](phase-04-agc.md) | 4-5 weeks | 17 | Phase 2, 3 | Not Started |
| 5 | [Noise Suppression](phase-05-noise-suppression.md) | 2-3 weeks | 6 | Phase 2 | Not Started |
| 6 | [Echo Cancellation (AEC3)](phase-06-echo-cancellation.md) | 6-8 weeks | 20 | Phase 2 | Not Started |
| 7 | [Mobile Echo Control (AECM)](phase-07-aecm.md) | 1-2 weeks | 3 | Phase 2 | Not Started |
| 8 | [Audio Processing Integration](phase-08-integration.md) | 3-4 weeks | 11 | Phases 2-7 | Not Started |
| 9 | [C API & Final Integration](phase-09-c-api.md) | 2-3 weeks | 7 | Phase 8 | Not Started |
| 10 | [Documentation & Release](phase-10-docs-release.md) | 1-2 weeks | 6 | Phase 9 | Not Started |
| **Total** | | **~28-37 weeks** | **~106** | | |

## Dependency Graph

```
Phase 1 (Foundation) ---- COMPLETE
  |
  v
Phase 2 (Common Audio)
  |
  +---> Phase 3 (VAD) --+
  |                      |
  +---> Phase 4 (AGC) <-+
  |
  +---> Phase 5 (NS)
  |
  +---> Phase 6 (AEC3)
  |
  +---> Phase 7 (AECM)
  |
  v
Phase 8 (Integration) <--- Phases 3-7
  |
  v
Phase 9 (C API)
  |
  v
Phase 10 (Docs & Release)
```

Phases 3-7 can be worked on in parallel after Phase 2 completes (except Phase 4 depends on Phase 3 for core VAD).

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
| FFT (bit-exact) | `pffft.c` via `cc` crate | Phase 2 |
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
  +-- webrtc-aec3 + webrtc-simd + tracing
  +-- webrtc-aecm + tracing
  +-- webrtc-agc + webrtc-simd + webrtc-vad + tracing
  +-- webrtc-ns + tracing
  +-- webrtc-vad + tracing
  +-- webrtc-simd

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
