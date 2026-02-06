# Phase 10: Documentation & Release

**Status:** Not Started
**Estimated Duration:** 1-2 weeks
**Dependencies:** Phase 9 (C API & Final Integration)
**Outcome:** The Rust port is fully documented, published to crates.io, and ready for downstream adoption. Migration guides, API docs, and architecture documentation are complete.

---

## Overview

Finalize documentation, create release artifacts, and publish to crates.io. This phase ensures the port is usable by both Rust and C consumers.

---

## Tasks

### 10.1 Rustdoc API Documentation

Add comprehensive rustdoc comments to all public APIs.

**Coverage requirements:**
- Every public struct, enum, trait, and function has a doc comment
- Examples for key APIs (AudioProcessing creation, configuration, processing)
- Module-level documentation explaining architecture
- Safety documentation for all `unsafe` functions
- `#![deny(missing_docs)]` at crate level

**Key documentation to write:**

```rust
/// Audio processing pipeline for real-time communication.
///
/// # Example
///
/// ```rust
/// use webrtc_apm::{AudioProcessing, AudioProcessingConfig, StreamConfig};
///
/// let config = AudioProcessingConfig {
///     echo_canceller: EchoCancellerConfig { enabled: true, ..Default::default() },
///     noise_suppression: NoiseSuppressionConfig { enabled: true, ..Default::default() },
///     ..Default::default()
/// };
///
/// let mut apm = AudioProcessing::builder()
///     .config(config)
///     .build()
///     .expect("Failed to create AudioProcessing");
///
/// // Process a 10ms frame at 48kHz
/// let input_config = StreamConfig::new(48000, 1);
/// let output_config = StreamConfig::new(48000, 1);
/// let mut output = vec![0.0f32; 480];
/// apm.process_stream_f32(&[&input], &input_config, &output_config, &mut [&mut output])
///     .expect("Processing failed");
/// ```
pub struct AudioProcessing { ... }
```

**Verification:**
- [ ] `cargo doc --workspace --no-deps` generates without warnings
- [ ] All public items documented
- [ ] Code examples compile and run

**Commit:** `docs(rust): add comprehensive rustdoc API documentation`

---

### 10.2 C API Documentation

Document the C API for non-Rust consumers.

**Files:**
- `crates/webrtc-apm/include/wap_audio_processing.h` - Doxygen comments
- `docs/c-api-guide.md` - C API usage guide (only if requested)

**Doxygen comments in header:**
```c
/**
 * @brief Create a new audio processing instance.
 *
 * @param config Pointer to configuration struct. Must not be NULL.
 * @return Pointer to new audio processing instance, or NULL on failure.
 *         Must be freed with wap_audio_processing_destroy().
 *
 * @note The returned instance is NOT thread-safe. Callers must ensure
 *       that ProcessStream and ProcessReverseStream are not called
 *       concurrently.
 */
WapAudioProcessing* wap_audio_processing_create(const WapConfig* config);
```

**Verification:**
- [ ] All C API functions have Doxygen comments
- [ ] Thread safety documented
- [ ] Memory ownership documented
- [ ] Error handling documented

**Commit:** `docs(rust): add C API documentation`

---

### 10.3 Architecture Documentation

Document the crate architecture for maintainers.

**Content:**
- Crate dependency graph
- Module-to-module mapping (C++ -> Rust)
- SIMD dispatch architecture
- Testing strategy (proptest + C++ test suite)
- How to add new components
- How to update when upstream WebRTC changes

This should be in the existing `CLAUDE.md` or a linked architecture doc.

**Verification:**
- [ ] Architecture is understandable to new contributors
- [ ] Dependency graph is accurate

**Commit:** `docs(rust): add architecture documentation for Rust port`

---

### 10.4 Migration Guide

Document how to migrate from the C++ library to the Rust implementation.

**Sections:**
1. Build system changes (Meson -> Cargo, or Meson with Rust backend)
2. API mapping (C++ methods -> C API functions -> Rust API)
3. Configuration mapping
4. Known differences (if any)
5. Performance characteristics

**Commit:** `docs(rust): add migration guide from C++ to Rust`

---

### 10.5 crates.io Publication

Prepare and publish crates to crates.io.

**Publication order (respecting dependencies):**
1. `webrtc-simd`
2. `webrtc-common-audio`
3. `webrtc-vad`
4. `webrtc-agc` (depends on webrtc-vad)
5. `webrtc-ns`
6. `webrtc-aec3`
7. `webrtc-aecm`
8. `webrtc-apm` (depends on all above)

**Cargo.toml preparation for each crate:**
```toml
[package]
name = "webrtc-apm"
version = "0.1.0"
edition = "2021"
rust-version = "1.82"
license = "BSD-3-Clause"
description = "Rust port of WebRTC Audio Processing (echo cancellation, noise suppression, AGC)"
repository = "https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing"
keywords = ["webrtc", "audio", "echo-cancellation", "noise-suppression", "agc"]
categories = ["multimedia::audio", "api-bindings"]
```

**Pre-publish checklist:**
- [ ] All crate metadata complete (description, license, repository, keywords)
- [ ] `cargo publish --dry-run` succeeds for each crate
- [ ] No unnecessary files in package (check `.gitignore`, `Cargo.toml` `exclude`)
- [ ] MSRV (minimum supported Rust version) documented and tested
- [ ] License files included in each crate
- [ ] CHANGELOG.md exists

**Commit:** `release(rust): prepare crates for crates.io publication`

---

### 10.6 Release Artifacts

Create release binaries for major platforms.

**Platforms:**
- Linux x86_64 (GNU + musl)
- Linux aarch64
- macOS x86_64
- macOS aarch64 (Apple Silicon)
- Windows x86_64 (MSVC)

**Artifacts per platform:**
- Shared library (`.so` / `.dylib` / `.dll`)
- Static library (`.a` / `.lib`)
- C header (`wap_audio_processing.h`)
- pkg-config file

**CI automation:**
```yaml
# Build release artifacts on each platform
release:
  strategy:
    matrix:
      os: [ubuntu-latest, macos-latest, windows-latest]
      arch: [x86_64, aarch64]
  steps:
    - cargo build --release -p webrtc-apm
    - # Package artifacts
```

**Verification:**
- [ ] Binaries build on all target platforms
- [ ] Libraries load and function correctly on each platform
- [ ] SIMD dispatch works on each platform

**Commit:** `ci(rust): add release artifact build pipeline`

---

## Phase 10 Completion Checklist

- [ ] All public APIs documented with rustdoc
- [ ] C API documented with Doxygen comments
- [ ] Architecture documentation complete
- [ ] Migration guide written
- [ ] All crates published to crates.io
- [ ] Release binaries available for all major platforms
- [ ] CHANGELOG exists
- [ ] MSRV documented and CI-enforced

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1 | `docs(rust): add comprehensive rustdoc API documentation` | All crates |
| 2 | `docs(rust): add C API documentation` | C header |
| 3 | `docs(rust): add architecture documentation for Rust port` | CLAUDE.md / arch doc |
| 4 | `docs(rust): add migration guide from C++ to Rust` | Migration guide |
| 5 | `release(rust): prepare crates for crates.io publication` | Cargo.toml metadata |
| 6 | `ci(rust): add release artifact build pipeline` | CI config |

---

## Success Criteria (Full Project)

At the end of Phase 10, the entire Rust port is complete. Final validation:

- [ ] **All 2432 C++ tests pass** against Rust backend
- [ ] **Property tests** demonstrate equivalence for all components
- [ ] **Performance within 10%** of C++ implementation
- [ ] **Builds on**: Linux x86_64, Linux aarch64, macOS x86_64, macOS aarch64, Windows x86_64
- [ ] **All SIMD paths** (SSE2, AVX2, NEON) functional and verified
- [ ] **C API** documented and usable from C/C++ projects
- [ ] **Published** to crates.io with complete documentation
- [ ] **Drop-in replacement** for existing C++ library consumers
