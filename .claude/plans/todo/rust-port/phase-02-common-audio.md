# Phase 2: Common Audio Primitives

**Status:** Complete (128 Rust tests passing)
**Duration:** ~3 weeks
**Dependencies:** Phase 1 (Foundation Infrastructure)
**Outcome:** The `webrtc-common-audio` crate contains audio utilities, channel buffers, and resamplers. The `webrtc-fft` crate contains pure Rust FFT implementations (Ooura 128, fft4g, PFFFT scalar). The `webrtc-ring-buffer` crate provides a generic ring buffer.

---

## Overview

Port the subset of `webrtc/common_audio/` needed by the modern audio processing pipeline (AEC3, AGC2, NS). The Signal Processing Library (30+ fixed-point C files) is **not ported** — it is only used by AECM (removed upstream), AGC1 (deprecated), and VAD filterbank. See master plan for rationale.

## Source Files to Port

### From `webrtc/common_audio/` (modern pipeline dependencies only):

**Ring Buffer (1 file) — COMPLETE:**
- `ring_buffer.c` -> standalone `webrtc-ring-buffer` crate (16 tests)

**Audio Utilities (2 files) — COMPLETE:**
- `audio_util.cc` + `include/audio_util.h` -> `audio_util.rs` (int16/float conversions)
- `channel_buffer.cc` -> `channel_buffer.rs` (multi-channel, multi-band buffer)

**Resampler (4 files) — COMPLETE:**
- `resampler/sinc_resampler.cc` + SIMD convolution via `webrtc-simd` -> `sinc_resampler.rs`
- `resampler/push_resampler.cc` -> `push_resampler.rs`
- `resampler/push_sinc_resampler.cc` -> `push_sinc_resampler.rs`

**FFT (Ooura + PFFFT) — COMPLETE (scalar), SIMD in progress:**
- `third_party/ooura/fft_size_128/ooura_fft.cc` -> `webrtc-fft` crate `ooura_fft.rs` (scalar)
- `third_party/ooura/fft_size_128/ooura_fft_sse2.cc` -> SSE2 SIMD (pending)
- `third_party/ooura/fft_size_128/ooura_fft_neon.cc` -> NEON SIMD (pending)
- `third_party/ooura/fft_size_256/fft4g.cc` -> `webrtc-fft` crate `fft4g.rs` (scalar only in C++)
- `third_party/pffft/src/pffft.c` -> `webrtc-fft` crate `pffft.rs` (pure Rust scalar)

**Deferred (not needed by current downstream phases):**
- `smoothing_filter.cc` — no downstream consumer in modern pipeline
- `fir_filter_*.cc` — only consumed by AEC3 (Phase 6), port alongside AEC3
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

### 2.2 Audio Utilities — COMPLETE

**Commits:**
1. `b7c105c feat(rust): port audio_util (sample format conversions)`
2. `0ab5248 feat(rust): port ChannelBuffer (multi-channel, multi-band audio buffer)`

**What was ported:**
- `audio_util.rs` — S16ToFloat, FloatToS16, FloatS16ToS16, etc.
- `channel_buffer.rs` — Generic `ChannelBuffer<T>` with multi-band support, `ChannelView`/`ChannelViewMut` slice types

**Deferred:**
- `smoothing_filter.cc` — No downstream consumer in modern pipeline (NS, AGC2, AEC3 don't use it)

---

### 2.3 FIR Filter — DEFERRED to Phase 6 (AEC3)

Only consumed by AEC3. Will be ported alongside AEC3 when needed.

---

### 2.4 Resampler — COMPLETE

**Commits:**
1. `9d49769 feat(rust): port SincResampler with SIMD-accelerated convolution`
2. `fdb098c feat(rust): port PushSincResampler and PushResampler`

**What was ported:**
- `sinc_resampler.rs` — Core sinc interpolation engine, uses `webrtc-simd` for SIMD-accelerated convolution (not separate SIMD files — the inner loop delegates to `webrtc-simd::sum` and `webrtc-simd::multiply_accumulate`)
- `push_sinc_resampler.rs` — Push-style wrapper around SincResampler
- `push_resampler.rs` — Top-level push resampler with channel support

---

### 2.5 FFT Wrappers — COMPLETE (scalar), SIMD pending

All FFT implementations live in the standalone `webrtc-fft` crate (pure Rust, no `cc` crate, no C code).

**Commits:**
1. `b90fcb7 feat(rust): add webrtc-fft crate with Ooura 128-point FFT`
2. `d8f2d40 feat(rust): port Ooura fft4g variable-size FFT`
3. `25a5f19 feat(rust): port PFFFT scalar FFT`
4. `2fb0947 refactor(rust): review improvements and proptests for webrtc-fft`

**What was ported:**
- `ooura_fft.rs` — 128-point fixed-size real FFT (scalar), 6 unit + 3 proptests
- `fft4g.rs` — Variable-size real FFT for power-of-2 sizes (scalar only in C++), 3 unit + 3 proptests
- `pffft.rs` — Variable-size real/complex FFT for composite sizes (pure Rust scalar), 10 unit + 4 proptests

**SIMD pending (will be added before NS port):**
- Ooura 128 SSE2 — 4 inner functions (`cft1st_128`, `cftmdl_128`, `rftfsub_128`, `rftbsub_128`)
- Ooura 128 NEON — same 4 inner functions

---

## Phase 2 Completion Checklist

- [x] Ring buffer ported as `webrtc-ring-buffer` crate (16 tests)
- [x] Audio utilities ported (audio_util, channel_buffer)
- [x] Resampler ported (sinc_resampler with SIMD, push_sinc_resampler, push_resampler)
- [x] FFT ported as `webrtc-fft` crate — pure Rust (ooura_fft, fft4g, pffft scalar)
- [x] Proptests for FFT (10 property tests across 3 modules)
- [x] `cargo nextest run --workspace` — 128 tests passing
- [x] `cargo clippy --workspace --all-targets` — zero warnings
- [ ] Ooura 128 FFT SSE2 SIMD (pending)
- [ ] Ooura 128 FFT NEON SIMD (pending)

**Deferred items (with rationale):**
- smoothing_filter — no downstream consumer
- FIR filter — only AEC3, port in Phase 6
- PFFFT SIMD — large scope, only AGC2, port in Phase 4

## Commit Summary

| Order | Commit | Status |
|-------|--------|--------|
| 1 | `ed42108 feat(rust): port ring buffer as webrtc-ring-buffer crate` | Done |
| 2 | `b7c105c feat(rust): port audio_util (sample format conversions)` | Done |
| 3 | `0ab5248 feat(rust): port ChannelBuffer` | Done |
| 4 | `9d49769 feat(rust): port SincResampler with SIMD-accelerated convolution` | Done |
| 5 | `fdb098c feat(rust): port PushSincResampler and PushResampler` | Done |
| 6 | `88cee75 refactor(rust): review improvements across all crates` | Done |
| 7 | `b90fcb7 feat(rust): add webrtc-fft crate with Ooura 128-point FFT` | Done |
| 8 | `d8f2d40 feat(rust): port Ooura fft4g variable-size FFT` | Done |
| 9 | `25a5f19 feat(rust): port PFFFT scalar FFT` | Done |
| 10 | `2fb0947 refactor(rust): review improvements and proptests for webrtc-fft` | Done |
