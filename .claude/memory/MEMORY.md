# WebRTC Audio Processing - Project Memory

## Project State
- Version 3.0, WebRTC M145 (branch-heads/7632), C++20
- 623 source files, 2432 tests passing, 185 disabled
- Build: Meson + Ninja, deps: abseil-cpp >= 20240722
- Main public API: `webrtc/api/audio/audio_processing.h`

## Tool Preferences
- ALWAYS use `rg` (ripgrep) via Bash for searching, NEVER the Grep tool
- Parameterless `new()` should be skipped; use `Default` instead

## Rust Port - Phases 1-2, 4-6 Complete
- 9 phases (1,2,4-9), plans in `.claude/plans/todo/rust-port/`
- Phase 3 (VAD) and AECM phase removed; empty scaffold crates deleted
- Master plan overview in `.claude/plans/todo/rust-port.md`
- Edition 2024, resolver 3, MSRV 1.91, no custom rustfmt.toml
- Tech: cxx (with C++ shim), tracing (direct dep, NOT feature-gated), proptest + test-strategy, std::arch SIMD, cbindgen
- Test runner: cargo nextest (NOT cargo test)
- SimdBackend is an enum (not trait objects) for type safety
- Workspace lints: unexpected_cfgs=deny, unreachable_pub, absolute_paths, mod_module_files=deny, etc.
- Commit discipline: every commit must be a fully working state (build + test green)
- 544 Rust tests passing across workspace
- Phase 2 done: ring buffer, audio_util, channel_buffer, sinc resampler (w/ SIMD), push resamplers, FFT (3 implementations)
- Phase 2 deferred: smoothing_filter (no consumer yet), FIR filter (AEC3 only), SincResampler AVX2
- FFT: standalone webrtc-fft crate (pure Rust, no cc crate), Ooura 128 + fft4g + PFFFT scalar
- Phase 5 done: webrtc-ns crate with 11 modules, 70 tests, single-channel 16kHz pipeline
- Phase 4 (AGC2) complete: 165 tests, 19 modules
  - webrtc-agc2 crate: RNN VAD (17 modules) + AGC2 core (biquad, gain curves, gain applier,
    limiter, level estimators, saturation protector, adaptive gain controller, clipping predictor)
  - Deferred to Phase 7: input_volume_controller, gain_controller2 (need AudioBuffer)
- Phase 6 (AEC3) complete: 172 tests, 61 source files, ~15,800 lines of Rust
  - webrtc-aec3 crate: full echo cancellation pipeline (21 steps)
  - Includes SIMD: MatchedFilter (SSE2/AVX2/NEON), AdaptiveFirFilter (SSE2/AVX2/NEON),
    ERL computation (SSE2/AVX2/NEON), plus webrtc-simd additions (sqrt, multiply, accumulate, power_spectrum)
  - Deferred to Phase 7: echo_canceller3.cc, config_selector.cc (need AudioBuffer, SwapQueue, threading)

## Excluded Modules (not ported)
- AECM: removed upstream M146 (Jan 2026), disabled by default
- AGC1: deprecated, entangled with SPL library, tracked by bugs.webrtc.org/7494
- Core VAD: depends on SPL, modern pipeline uses AGC2's RNN VAD
- SPL (signal_processing/): 30+ fixed-point C files, only used by AECM/AGC1/VAD filterbank
- Modern pipeline (AEC3, AGC2, NS) uses none of SPL

## Phase Numbering (after cleanup)
- Phase 7: Audio Processing Integration — NEXT
- Phase 8: C API & Final Integration
- Phase 9: Documentation & Release

## Key Architecture Notes
- AudioProcessing is abstract class with virtual methods - cxx needs shim layer
- SIMD: SSE2 inline in some files, AVX2 in separate .cc files with -mavx2, NEON inline + ARM assembly
- Meson builds AVX2 into separate static libs (`webrtc_audio_processing_privatearch`)
- NEON enabled via `-ffp-contract=fast` for FMA matching
- pffft.c bundled in webrtc/third_party/pffft/ — now ported to pure Rust in webrtc-fft crate
- RNN VAD uses custom tanh approximation - critical for bit-exactness

## Codebase Layout
- `webrtc/modules/audio_processing/` - 343 files, main processing
- `webrtc/common_audio/` - 115 files, DSP primitives
- `webrtc/api/` - 61 files, public API
- `webrtc/rtc_base/` - 83 files, base utilities
- `tests/` - 125 test .cc files, resources in tests/resources/
- `patches/` - 10 patches for platform compat (BSD, MinGW, MSVC, etc.)

## Phase 4 Patterns & Lessons
- FC/GRU layers take `VectorMath` (not `SimdBackend` directly) — wrap with `VectorMath::new(backend)`
- `SequenceBuffer<S, N>` has 2 const generics; `push` takes `&[f32; N]` (not slice)
- Const generic arithmetic (`[f32; S * N]`) NOT allowed in stable Rust — use `Vec<f32>` instead
- PFFFT `forward`/`backward` can't use same buffer for input and output (Rust borrow rules)
- `absolute_paths` lint: must import `std::f32::consts::PI` etc., not use inline
- `allow` attributes need `reason = "..."` due to workspace `allow_attributes_without_reason` lint
- Test data files in `tests/resources/audio_processing/agc2/rnn_vad/` — read as binary f32 LE
- ARM64 has separate test reference data (`pitch_lp_res_arm64.dat`) due to FP differences

## Phase 6 Patterns & Lessons
- C++ virtual interfaces (EchoRemover, BlockProcessor, RenderDelayBuffer) → Rust concrete structs
- ApmDataDumper, NeuralResidualEchoEstimator, RTC_HISTOGRAM: all skipped in port
- SubtractorOutput field naming: C++ `e2_refined`/`e2_coarse` (scalars) → Rust `e2_refined_sum`/`e2_coarse_sum`
- `vec![T::default(); n]` requires Clone — use `(0..n).map(|_| T::default()).collect()` instead
- Borrow checker: avoid early immutable borrows of arrays that are later mutated; restructure code flow
- C++ `std::optional<size_t> x = size_t_val` implicit conversion — `size_t` always converts to Some
- AEC3 has 5 different circular buffer types — do NOT reuse webrtc-ring-buffer; port each exactly
- SignalTransition: 30-sample crossfade between refined/coarse filter outputs
- MatchedFilter SIMD: most complex port — NEON uses FMA, SSE2 uses separate mul+add

## Existing Rust Ecosystem
- tonarino/webrtc-audio-processing crate exists (FFI wrapper, NOT a port)
- pffft_rust crate exists but stale (2020)
- std::simd NOT stable - use std::arch intrinsics

# Environment
