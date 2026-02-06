# Claude Code Context for webrtc-audio-processing

## Tool Preferences

- **ALWAYS use `rg` (ripgrep) for searching**, never the Grep tool. Use `rg` via Bash.

## Project Overview

This is a Linux packaging-friendly copy of the AudioProcessing module from the WebRTC project. It provides echo cancellation, noise suppression, automatic gain control, and other audio processing capabilities used by projects like PulseAudio.

**Repository:** https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing/

## Current State

- **Version:** 2.1
- **WebRTC Base:** M131 (branch-heads/6778)
- **Build System:** Meson
- **C++ Standard:** C++17
- **Test Count:** 699 unit tests passing (19 disabled, matching upstream)

## Key Directories

```
webrtc-audio-processing/
├── webrtc/                    # Main source code (synced from upstream)
│   ├── api/audio/             # Public API (audio_processing.h)
│   ├── modules/audio_processing/  # Core audio processing
│   │   ├── aec3/              # Echo Canceller 3
│   │   ├── agc/               # AGC1 (legacy)
│   │   ├── agc2/              # AGC2 (modern)
│   │   ├── ns/                # Noise suppression
│   │   ├── vad/               # Voice activity detection
│   │   └── ...
│   ├── common_audio/          # Audio utilities (resampler, filters)
│   └── rtc_base/              # Base utilities
├── tests/                     # Unit tests
│   ├── unit/                  # Test source files
│   │   ├── aec3/              # AEC3 tests (30 files)
│   │   └── agc2/              # AGC2 tests (16 files)
│   ├── test_utils/            # Test utilities and mocks
│   │   └── mock/              # Mock classes
│   └── resources/             # Test audio files (PCM)
├── patches/                   # Patches applied on top of upstream
├── examples/                  # Example applications
└── .claude/plans/             # Planning documents
```

## Build Instructions

```bash
# Configure with tests enabled
meson setup builddir -Dtests=enabled

# Build
ninja -C builddir

# Run tests
meson test -C builddir -v

# Install locally for testing with PulseAudio
meson setup builddir -Dprefix=$(pwd)/install
ninja -C builddir install
```

## Upstream Reference

The upstream WebRTC source for M131 is checked out at:
```
../webrtc-upstream  (branch-heads/6778)
```

To fetch the correct branch:
```bash
cd ../webrtc-upstream
git fetch origin branch-heads/6778
git checkout FETCH_HEAD
```

## Plans and Documentation

### Upgrade Plan
**File:** `.claude/plans/merry-swinging-bonbon.md`

Comprehensive plan for upgrading from M131 to M136, including:
- Code synchronization workflow
- Patch reapplication strategy
- ABI compatibility checking
- Test reference value updates

### Test Coverage Plan
**File:** `.claude/plans/improve-test-coverage.md`

Detailed inventory of test porting progress:
- 53 test files ported (756 tests)
- 58+ test files remaining to port
- Prioritized porting order by component

## Test Porting Guidelines

### Exact Upstream Match Policy
Tests should match upstream exactly, with only include path conversions. Do not:
- Add or remove tests
- Change whitespace or comments
- Modify TODO comments
- Add additional assertions

### Include Conversions
When porting upstream tests, convert includes using sed:
```bash
sed -e 's|"modules/|"webrtc/modules/|g' \
    -e 's|"common_audio/|"webrtc/common_audio/|g' \
    -e 's|"api/|"webrtc/api/|g' \
    -e 's|"rtc_base/|"webrtc/rtc_base/|g' \
    -e 's|"system_wrappers/|"webrtc/system_wrappers/|g' \
    -e 's|"test/gtest.h"|<gtest/gtest.h>|g' \
    -e 's|"test/gmock.h"|<gmock/gmock.h>|g' \
    -e 's|"test/testsupport/|"tests/test_utils/|g' \
    -e 's|"test/field_trial.h"|"tests/test_utils/field_trial.h"|g' \
    upstream_test.cc > tests/unit/test.cc
```

### Test Utilities
Test utilities in `tests/test_utils/` are synced from upstream with include path conversions:
- `field_trial.h/cc` - ScopedFieldTrials for field trial overrides (has NULL fix in destructor)
- `rtc_expect_death.h` - Death test macro wrapper
- `audio_buffer_tools.h/cc` - AudioBuffer test helpers
- `bitexactness_tools.h/cc` - Bit-exact comparison helpers
- `file_utils.h/cc` - File path utilities
- `input_audio_file.h/cc` - Audio file reading
- `mock/*.h/cc` - Mock classes from upstream `aec3/mock/`

### FRIEND_TEST Requirements
For tests that access private members via `FRIEND_TEST_ALL_PREFIXES`:
- Tests must be in `namespace webrtc` (not anonymous namespace)
- Test class name must match the FRIEND_TEST declaration in the header

### Test Resources
Audio test files are in `tests/resources/`:
- `near*.pcm`, `far*.pcm` - Stereo PCM at various sample rates
- `audio_processing/agc/agc_audio.pcm` - 16kHz mono for AGC tests

### Adding New Tests
1. Copy upstream test file to appropriate `tests/unit/` subdirectory
2. Convert includes using sed or manual editing
3. Add source file to `tests/meson.build` in `unit_test_sources`
4. Build and run: `ninja -C builddir && meson test -C builddir -v`

## Patches (10 total)

| Patch | Purpose | Risk |
|-------|---------|------|
| `0001-arch.h-Add-s390x-support.patch` | s390x architecture | Low |
| `0001-AECM-MIPS-Use-uintptr_t-for-pointer-arithmetic.patch` | MIPS fixes | Low |
| `0001-common_audio-Add-MIPS_DSP_R1_LE-guard-for-vector-sca.patch` | MIPS DSP | Low |
| `0001-Fix-compilation-with-gcc-15.patch` | GCC 15 compat | Low |
| `0001-Some-fixes-for-MinGW.patch` | MinGW support | Low |
| `0001-Add-support-for-BSD-systems.patch` | BSD support | Medium |
| `0001-Fix-up-XMM-intrinsics-usage-on-MSVC.patch` | MSVC SIMD | Medium |
| `0001-meson-Fixes-for-MSVC-build.patch` | MSVC build | Medium |
| `0001-Allow-disabling-inline-SSE.patch` | SSE flexibility | High |
| `0001-Fix-build-with-abseil-cpp-202508.patch` | abseil compat | High |

## Current Work: Test Coverage Expansion

### Completed (53 files, 756 tests)
- Root level: 7 tests (rms_level, audio_buffer, splitting_filter, high_pass_filter, audio_frame_view, gain_control, residual_echo_detector)
- AEC3: 30 tests
- AGC2: 16 tests

### Remaining Priority Order
1. **Echo Detector** (4 files) - circular_buffer, mean_variance_estimator, moving_max, normalized_covariance_estimator
2. **Utility** (3 files) - cascaded_biquad_filter, delay_estimator, pffft_wrapper
3. **Capture Levels Adjuster** (2 files)
4. **NS** (1 file) - noise_suppressor
5. **VAD** (8 files)
6. **Remaining AEC3** (12 files)
7. **AGC1** (2 files)
8. **AGC2 RNN_VAD** (11 files)
9. **Common Audio** (~15 files)

## Dependencies

- **abseil-cpp:** >= 20240722
- **gtest/gmock:** For tests (can use subproject fallback)

## Key APIs

Main public header: `webrtc/api/audio/audio_processing.h`

```cpp
#include <webrtc/api/audio/audio_processing.h>

// Create AudioProcessing instance
auto apm = webrtc::AudioProcessingBuilder().Create();

// Configure
webrtc::AudioProcessing::Config config;
config.echo_canceller.enabled = true;
config.noise_suppression.enabled = true;
config.gain_controller2.enabled = true;
apm->ApplyConfig(config);

// Process audio
apm->ProcessStream(...);
apm->ProcessReverseStream(...);
```

## Session Notes (February 5, 2025)

### Test Synchronization Complete
- All test files replaced with exact upstream versions (include path conversions only)
- All test utilities synced with upstream
- All mock files synced with upstream `aec3/mock/` directory
- **Final count: 699 tests passing, 19 disabled (matching upstream)**

### Key Fixes Applied
1. **field_trial.cc NULL fix**: Added NULL check before `FieldTrialsStringIsValid()` to prevent crash when `previous_field_trials_` is NULL
2. **Metrics enabled**: Created custom `test_main.cc` that calls `webrtc::metrics::Enable()` for metrics-dependent tests
3. **NEON FMA precision**: Added `-ffp-contract=fast` to enable FMA in C code to match NEON precision

### Upstream Reference
- Source at `../webrtc-upstream` (branch-heads/6778 / M131)
- Tests match upstream exactly with only include path conversions
