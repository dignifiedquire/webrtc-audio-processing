# Phase 9: C API & Final Integration

**Status:** Not Started
**Estimated Duration:** 2-3 weeks
**Dependencies:** Phase 8 (Audio Processing Integration)
**Outcome:** The Rust `webrtc-apm` crate exposes a stable C API via `cbindgen`, producing a drop-in replacement shared library. All 2432 C++ tests pass when linked against the Rust implementation.

---

## Overview

Create the C-compatible API that allows existing C/C++ consumers (PulseAudio, PipeWire, etc.) to use the Rust implementation as a drop-in replacement. This phase also validates the entire port by running the existing C++ test suite against the Rust backend.

---

## Tasks

### 9.1 C API Design

Design a C API that mirrors the essential functionality of the C++ `AudioProcessing` class.

**Destination:**
```
webrtc-apm/src/
  ffi.rs             # C FFI function exports
  ffi_types.rs       # #[repr(C)] types for the C API
```

**API:**
```rust
// ffi_types.rs
#[repr(C)]
pub struct WapConfig {
    pub echo_cancellation_enabled: bool,
    pub echo_cancellation_mobile_mode: bool,
    pub noise_suppression_enabled: bool,
    pub noise_suppression_level: WapNsLevel,
    pub gain_controller1_enabled: bool,
    pub gain_controller1_mode: WapAgc1Mode,
    pub gain_controller2_enabled: bool,
    pub gain_controller2_fixed_gain_db: f32,
    pub high_pass_filter_enabled: bool,
    // ... other fields matching AudioProcessing::Config
}

#[repr(C)]
pub enum WapNsLevel {
    Low = 0,
    Moderate = 1,
    High = 2,
    VeryHigh = 3,
}

#[repr(C)]
pub enum WapAgc1Mode {
    AdaptiveAnalog = 0,
    AdaptiveDigital = 1,
    FixedDigital = 2,
}

#[repr(C)]
pub enum WapError {
    None = 0,
    Unspecified = -1,
    BadParameter = -6,
    BadSampleRate = -7,
    BadDataLength = -8,
    BadNumberChannels = -9,
}

#[repr(C)]
pub struct WapStreamConfig {
    pub sample_rate_hz: i32,
    pub num_channels: usize,
}

#[repr(C)]
pub struct WapStats {
    pub has_voice: bool,
    pub echo_return_loss: f32,
    pub echo_return_loss_enhancement: f32,
    pub divergent_filter_fraction: f32,
    pub delay_ms: i32,
    pub residual_echo_likelihood: f32,
}
```

```rust
// ffi.rs
use std::ptr;

/// Opaque handle to AudioProcessing
pub struct WapAudioProcessing {
    inner: AudioProcessing,
}

#[no_mangle]
pub extern "C" fn wap_audio_processing_create(
    config: *const WapConfig,
) -> *mut WapAudioProcessing {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_audio_processing_destroy(
    apm: *mut WapAudioProcessing,
) {
    if !apm.is_null() {
        unsafe { drop(Box::from_raw(apm)); }
    }
}

#[no_mangle]
pub extern "C" fn wap_apply_config(
    apm: *mut WapAudioProcessing,
    config: *const WapConfig,
) -> WapError {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_process_stream_i16(
    apm: *mut WapAudioProcessing,
    src: *const i16,
    input_config: *const WapStreamConfig,
    output_config: *const WapStreamConfig,
    dest: *mut i16,
) -> WapError {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_process_stream_f32(
    apm: *mut WapAudioProcessing,
    src: *const *const f32,  // deinterleaved channels
    input_config: *const WapStreamConfig,
    output_config: *const WapStreamConfig,
    dest: *mut *mut f32,
) -> WapError {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_process_reverse_stream_i16(
    apm: *mut WapAudioProcessing,
    src: *const i16,
    input_config: *const WapStreamConfig,
    output_config: *const WapStreamConfig,
    dest: *mut i16,
) -> WapError {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_process_reverse_stream_f32(
    apm: *mut WapAudioProcessing,
    src: *const *const f32,
    input_config: *const WapStreamConfig,
    output_config: *const WapStreamConfig,
    dest: *mut *mut f32,
) -> WapError {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_set_stream_analog_level(
    apm: *mut WapAudioProcessing,
    level: i32,
) {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_recommended_stream_analog_level(
    apm: *const WapAudioProcessing,
) -> i32 {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_set_stream_delay_ms(
    apm: *mut WapAudioProcessing,
    delay: i32,
) -> WapError {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_get_statistics(
    apm: *const WapAudioProcessing,
    stats: *mut WapStats,
) -> WapError {
    // ...
}

#[no_mangle]
pub extern "C" fn wap_get_config(
    apm: *const WapAudioProcessing,
    config: *mut WapConfig,
) -> WapError {
    // ...
}
```

**Safety:** All C API functions must:
- Check for null pointers
- Catch panics at the FFI boundary (`std::panic::catch_unwind`)
- Return error codes, never panic across the FFI boundary
- Document thread safety guarantees (same as C++ API)

**Verification:**
- [ ] All FFI functions handle null pointers gracefully
- [ ] No panics escape across FFI boundary
- [ ] Memory is properly managed (create/destroy lifecycle)

**Commit:** `feat(rust): define C API types and function signatures`

---

### 9.2 C Header Generation

Use cbindgen to auto-generate the C header.

**Files:**
- `crates/webrtc-apm/cbindgen.toml` - cbindgen configuration
- `crates/webrtc-apm/build.rs` - Generate header during build

**cbindgen.toml:**
```toml
language = "C"
include_guard = "WAP_AUDIO_PROCESSING_H"
tab_width = 4
style = "both"
cpp_compat = true

[export]
prefix = "Wap"

[export.rename]
"WapAudioProcessing" = "WapAudioProcessing"

[parse]
parse_deps = false
```

**build.rs:**
```rust
fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::generate(crate_dir)
        .expect("Unable to generate C bindings")
        .write_to_file("include/wap_audio_processing.h");
}
```

**Verification:**
- [ ] Generated header compiles with both C11 and C++20 compilers
- [ ] Header includes all public API functions
- [ ] Types are correctly represented

**Commit:** `feat(rust): add cbindgen C header generation`

---

### 9.3 Shared Library Build

Configure the crate to produce a C-compatible shared library.

**Cargo.toml:**
```toml
[lib]
name = "wap_audio_processing"
crate-type = ["cdylib", "staticlib", "rlib"]
```

**Meson integration:**
Create a Meson build option to use the Rust backend:
```meson
# In meson.build
rust_backend = get_option('rust-backend')
if rust_backend
  # Build Rust library
  # Link against it instead of C++
endif
```

**pkg-config file:**
Generate `wap-audio-processing.pc` for downstream projects.

**Verification:**
- [ ] `cargo build --release` produces `.so`/`.dylib`/`.dll`
- [ ] `cargo build --release` produces `.a`/`.lib`
- [ ] Library exports all C API symbols
- [ ] pkg-config file is correct

**Commit:** `feat(rust): configure shared/static library output and pkg-config`

---

### 9.4 C++ Test Suite Validation

The ultimate validation: run the existing C++ test suite against the Rust implementation.

**Approach:**
Create a C++ compatibility shim that wraps the Rust C API to look like the C++ `AudioProcessing` interface. Then recompile the test suite linking against this shim.

**Files:**
```
tests/
  rust_compat/
    rust_audio_processing.h    # C++ class wrapping Rust C API
    rust_audio_processing.cc   # Implementation
```

**Pattern:**
```cpp
// rust_audio_processing.h
class RustAudioProcessing : public AudioProcessing {
public:
    RustAudioProcessing();
    ~RustAudioProcessing() override;
    
    int Initialize() override;
    void ApplyConfig(const Config& config) override;
    int ProcessStream(const int16_t* const src, ...) override;
    // ... all virtual methods delegate to wap_* C functions
    
private:
    WapAudioProcessing* handle_;
};
```

**Test execution:**
```bash
# Build Rust library
cargo build --release -p webrtc-apm

# Build C++ tests linking against Rust backend
meson setup builddir-rust -Drust-backend=true
ninja -C builddir-rust

# Run tests
meson test -C builddir-rust -v
```

**Expected outcome:** All 2432 tests pass.

**Handling expected differences:**
- Some tests may need tolerance adjustments due to floating-point operation ordering
- Document any tests that require modifications
- Goal: zero test modifications needed

**Verification:**
- [ ] All 2432 C++ tests pass against Rust backend
- [ ] No test modifications required (ideal)
- [ ] Performance within 10% of C++ implementation
- [ ] Memory usage comparable to C++ implementation

**Commits:**
1. `feat(rust): add C++ compatibility shim for test validation`
2. `feat(rust): integrate Rust backend with Meson test suite`
3. `test(rust): validate all 2432 C++ tests pass against Rust backend`

---

### 9.5 Performance Benchmarking

Compare Rust and C++ performance.

**Benchmark dimensions:**
- ProcessStream latency (per-frame)
- ProcessReverseStream latency (per-frame)
- Memory usage (peak RSS)
- Library size (stripped binary)

**Configurations to benchmark:**
1. Echo cancellation only
2. Noise suppression only
3. AGC2 only
4. All components enabled
5. Different sample rates (16k, 48k)

**Tools:**
- `criterion` crate for Rust micro-benchmarks
- Custom benchmark harness comparing Rust and C++ wall-clock time

**Target:** Rust within 10% of C++ performance. If slower, profile and optimize hot paths.

**Verification:**
- [ ] Benchmark results documented
- [ ] Performance within 10% of C++ for all configurations
- [ ] No unexpected memory growth

**Commit:** `bench(rust): add performance benchmarks comparing Rust vs C++`

---

## Phase 9 Completion Checklist

- [ ] C API defined with all essential functions
- [ ] cbindgen generates valid C header
- [ ] Shared and static libraries built
- [ ] pkg-config file generated
- [ ] All 2432 C++ tests pass against Rust backend
- [ ] Performance within 10% of C++
- [ ] Memory usage comparable
- [ ] Documentation for C API consumers

## Commit Summary

| Order | Commit | Scope |
|-------|--------|-------|
| 1 | `feat(rust): define C API types and function signatures` | ffi.rs, ffi_types.rs |
| 2 | `feat(rust): add cbindgen C header generation` | cbindgen.toml, build.rs |
| 3 | `feat(rust): configure shared/static library output and pkg-config` | Cargo.toml, meson |
| 4 | `feat(rust): add C++ compatibility shim for test validation` | rust_compat/ |
| 5 | `feat(rust): integrate Rust backend with Meson test suite` | meson.build changes |
| 6 | `test(rust): validate all 2432 C++ tests pass against Rust backend` | Test results |
| 7 | `bench(rust): add performance benchmarks comparing Rust vs C++` | Benchmarks |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| C++ tests fail against Rust backend | Medium | High | Debug failing tests individually; may need tolerance adjustments |
| Performance regression | Medium | High | Profile hot paths; optimize SIMD dispatch |
| C API design misses use cases | Low | Medium | Study PulseAudio's usage of the C++ API |
| cbindgen type mapping issues | Low | Low | Manual header review and compilation test |
| Panic across FFI boundary | Medium | Critical | catch_unwind in every extern "C" fn; aggressive testing |
