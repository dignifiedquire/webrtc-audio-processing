# Rust Port Plan

**Goal:** Port WebRTC Audio Processing from C++20 to Rust, producing a modular multi-crate library with a C-compatible drop-in replacement API.

**Source:** WebRTC M145 (branch-heads/7632), version 3.0
**Scope:** 623 source files, 2432 tests, SSE2/AVX2/NEON SIMD optimizations

---

## Phase Overview

| Phase | Name | Duration | Commits | Dependencies | Status |
|-------|------|----------|---------|--------------|--------|
| 1 | [Foundation Infrastructure](phase-01-foundation.md) | 2-3 weeks | 6 | None | Not Started |
| 2 | [Common Audio Primitives](phase-02-common-audio.md) | 3-4 weeks | 20 | Phase 1 | Not Started |
| 3 | [Voice Activity Detection](phase-03-vad.md) | 2 weeks | 5 | Phase 2 | Not Started |
| 4 | [Automatic Gain Control](phase-04-agc.md) | 4-5 weeks | 17 | Phase 2, 3 | Not Started |
| 5 | [Noise Suppression](phase-05-noise-suppression.md) | 2-3 weeks | 6 | Phase 2 | Not Started |
| 6 | [Echo Cancellation (AEC3)](phase-06-echo-cancellation.md) | 6-8 weeks | 20 | Phase 2 | Not Started |
| 7 | [Mobile Echo Control (AECM)](phase-07-aecm.md) | 1-2 weeks | 3 | Phase 2 | Not Started |
| 8 | [Audio Processing Integration](phase-08-integration.md) | 3-4 weeks | 11 | Phases 2-7 | Not Started |
| 9 | [C API & Final Integration](phase-09-c-api.md) | 2-3 weeks | 7 | Phase 8 | Not Started |
| 10 | [Documentation & Release](phase-10-docs-release.md) | 1-2 weeks | 6 | Phase 9 | Not Started |
| **Total** | | **~28-37 weeks** | **~101** | | |

## Dependency Graph

```
Phase 1 (Foundation)
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

## Technology Stack

| Concern | Choice | Version |
|---------|--------|---------|
| C++ interop (testing) | `cxx` + C++ shim layer | 1.0.x |
| SIMD | `std::arch` intrinsics + runtime dispatch | stable |
| Logging | `tracing` (feature-gated) | 0.1.x |
| Property testing | `proptest` | 1.9.x |
| C header generation | `cbindgen` | 0.29.x |
| FFT (bit-exact) | `pffft.c` via `cc` crate | bundled |
| Benchmarking | `criterion` | latest |

## Crate Architecture

```
webrtc-apm (main crate, C API)
  +-- webrtc-common-audio (DSP primitives)
  +-- webrtc-aec3 (echo cancellation)
  |     +-- webrtc-common-audio
  |     +-- webrtc-simd
  +-- webrtc-aecm (mobile echo control)
  |     +-- webrtc-common-audio
  +-- webrtc-agc (gain control)
  |     +-- webrtc-common-audio
  |     +-- webrtc-simd
  |     +-- webrtc-vad
  +-- webrtc-ns (noise suppression)
  |     +-- webrtc-common-audio
  +-- webrtc-vad (voice activity detection)
  |     +-- webrtc-common-audio
  +-- webrtc-simd (SIMD abstractions)
```

Testing crates (not published):
- `webrtc-apm-sys` - cxx bindings to C++ for comparison testing
- `webrtc-apm-proptest` - Property test harness and audio generators

## Commit Discipline

Every commit must:
1. Compile (`cargo build --workspace`)
2. Pass all Rust tests (`cargo test --workspace`)
3. Pass all C++ tests (`meson test -C builddir`)
4. Pass clippy (`cargo clippy --workspace -- -D warnings`)
5. Be formatted (`cargo fmt --all -- --check`)

Commits should be atomic and focused:
- One logical change per commit
- Working state at every commit
- Clear commit message with conventional commit prefix (`feat`, `fix`, `test`, `docs`, `bench`, `ci`)

## Verification Strategy

Each ported component is verified at three levels:

1. **Unit tests** - Mirror the C++ unit test behaviors in Rust
2. **Property tests** - Random input comparison via `proptest` + cxx FFI to C++ reference
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
