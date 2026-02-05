# Plan: Improve Test Coverage at Function Level

## Overview

Expand unit test coverage from the current ~40 tests to comprehensive function-level testing by porting upstream WebRTC tests.

**Current State:**
- 4 test files, ~87 tests
- Tests cover: RmsLevel, AudioBuffer, SplittingFilter, HighPassFilter
- Major gaps: AEC3, AGC, NS, VAD, main AudioProcessing API

**Target:**
- Port all applicable upstream WebRTC M131 (branch-heads/6778) tests
- Achieve function-level coverage for all public APIs and major components

**Critical:** All tests must be sourced from **WebRTC M131 (branch-heads/6778)** to match the implementation version. Using tests from a different version will cause bit-exact test failures due to mismatched reference values.

## Upstream Test Inventory

**Source:** `https://webrtc.googlesource.com/src/+/refs/branch-heads/6778/modules/audio_processing/`

### Root Level (`modules/audio_processing/`)

| Test File | Status | Priority | Dependencies |
|-----------|--------|----------|--------------|
| `rms_level_unittest.cc` | ✅ Ported | - | None |
| `audio_buffer_unittest.cc` | ✅ Ported | - | None |
| `splitting_filter_unittest.cc` | ✅ Ported | - | None |
| `high_pass_filter_unittest.cc` | ✅ Ported | - | None |
| `audio_frame_view_unittest.cc` | ❌ Not ported | HIGH | None |
| `gain_control_unittest.cc` | ❌ Not ported | HIGH | None |
| `gain_controller2_unittest.cc` | ❌ Not ported | HIGH | Test data files |
| `residual_echo_detector_unittest.cc` | ❌ Not ported | HIGH | None |
| `echo_control_mobile_unittest.cc` | ❌ Not ported | MEDIUM | None |
| `audio_processing_unittest.cc` | ❌ Not ported | HIGH | Test data files, protobuf |
| `audio_processing_impl_unittest.cc` | ❌ Not ported | MEDIUM | Complex setup |
| `audio_processing_impl_locking_unittest.cc` | ❌ Not ported | LOW | Threading |
| `audio_processing_performance_unittest.cc` | ❌ Not ported | LOW | Benchmarking |
| `echo_control_mobile_bit_exact_unittest.cc` | ❌ Not ported | LOW | Protobuf |

### AEC3 (`modules/audio_processing/aec3/`)

| Test File | Priority | Notes |
|-----------|----------|-------|
| `echo_canceller3_unittest.cc` | HIGH | Main EC3 entry point |
| `aec_state_unittest.cc` | HIGH | State management |
| `block_processor_unittest.cc` | HIGH | Core processing |
| `echo_remover_unittest.cc` | HIGH | Echo removal |
| `adaptive_fir_filter_unittest.cc` | MEDIUM | Filter adaptation |
| `matched_filter_unittest.cc` | MEDIUM | Delay estimation |
| `comfort_noise_generator_unittest.cc` | MEDIUM | CNG |
| `decimator_unittest.cc` | LOW | Downsampling |
| `fft_data_unittest.cc` | LOW | FFT utilities |
| `aec3_fft_unittest.cc` | LOW | FFT |
| `moving_average_unittest.cc` | LOW | Utility |
| `block_framer_unittest.cc` | LOW | Framing |
| `frame_blocker_unittest.cc` | LOW | Blocking |
| `erl_estimator_unittest.cc` | LOW | Metrics |
| `erle_estimator_unittest.cc` | LOW | Metrics |
| ... (15+ more) | LOW | Various utilities |

### AGC2 (`modules/audio_processing/agc2/`)

| Test File | Priority | Notes |
|-----------|----------|-------|
| `adaptive_digital_gain_controller_unittest.cc` | HIGH | Main AGC2 |
| `input_volume_controller_unittest.cc` | HIGH | Volume control |
| `limiter_unittest.cc` | HIGH | Limiting |
| `clipping_predictor_unittest.cc` | MEDIUM | Clipping prevention |
| `saturation_protector_unittest.cc` | MEDIUM | Saturation |
| `speech_level_estimator_unittest.cc` | MEDIUM | Level estimation |
| `noise_level_estimator_unittest.cc` | MEDIUM | Noise estimation |
| `gain_applier_unittest.cc` | LOW | Gain application |
| `biquad_filter_unittest.cc` | LOW | Filtering |
| `interpolated_gain_curve_unittest.cc` | LOW | Curves |
| `limiter_db_gain_curve_unittest.cc` | LOW | Curves |
| `vad_wrapper_unittest.cc` | LOW | VAD integration |
| ... (5+ more) | LOW | Various utilities |

### Noise Suppression (`modules/audio_processing/ns/`)

| Test File | Priority | Notes |
|-----------|----------|-------|
| `noise_suppressor_unittest.cc` | HIGH | Main NS entry point |

### VAD (`modules/audio_processing/vad/`)

| Test File | Priority | Notes |
|-----------|----------|-------|
| `voice_activity_detector_unittest.cc` | HIGH | Main VAD |
| `standalone_vad_unittest.cc` | MEDIUM | Standalone VAD |
| `gmm_unittest.cc` | LOW | GMM |
| `pitch_based_vad_unittest.cc` | LOW | Pitch detection |
| `pitch_internal_unittest.cc` | LOW | Pitch internals |
| `pole_zero_filter_unittest.cc` | LOW | Filtering |
| `vad_audio_proc_unittest.cc` | LOW | Audio processing |
| `vad_circular_buffer_unittest.cc` | LOW | Buffer |

## Phased Implementation

### Phase 1: Self-Contained Tests (No External Dependencies)

Port tests that don't require test data files or protobuf.

**Files to port:**
1. `audio_frame_view_unittest.cc` - Simple view abstraction
2. `gain_control_unittest.cc` - AGC1 (if self-contained)
3. `residual_echo_detector_unittest.cc` - Echo detection

**Estimated tests:** 20-30

### Phase 2: AGC2 Component Tests

Port AGC2 subsystem tests.

**Files to port:**
1. `agc2/adaptive_digital_gain_controller_unittest.cc`
2. `agc2/limiter_unittest.cc`
3. `agc2/input_volume_controller_unittest.cc`
4. `agc2/clipping_predictor_unittest.cc`
5. `agc2/saturation_protector_unittest.cc`
6. `agc2/speech_level_estimator_unittest.cc`
7. `agc2/noise_level_estimator_unittest.cc`
8. `agc2/gain_applier_unittest.cc`
9. `agc2/biquad_filter_unittest.cc`

**Estimated tests:** 50-80

### Phase 3: AEC3 Component Tests

Port AEC3 subsystem tests.

**Files to port:**
1. `aec3/echo_canceller3_unittest.cc`
2. `aec3/aec_state_unittest.cc`
3. `aec3/block_processor_unittest.cc`
4. `aec3/echo_remover_unittest.cc`
5. `aec3/adaptive_fir_filter_unittest.cc`
6. `aec3/matched_filter_unittest.cc`
7. `aec3/comfort_noise_generator_unittest.cc`
8. Additional utility tests as needed

**Estimated tests:** 80-120

### Phase 4: Noise Suppression and VAD

**Files to port:**
1. `ns/noise_suppressor_unittest.cc`
2. `vad/voice_activity_detector_unittest.cc`
3. `vad/standalone_vad_unittest.cc`
4. Additional VAD tests as needed

**Estimated tests:** 30-50

### Phase 5: Integration Tests

Port higher-level integration tests (may require test data).

**Files to port:**
1. `audio_processing_impl_unittest.cc`
2. `gain_controller2_unittest.cc` (full integration)
3. `echo_control_mobile_unittest.cc`

**Estimated tests:** 40-60

## Test Infrastructure Improvements

### Required Test Utilities

Add to `tests/test_utils/`:

1. **Signal generators** (already have some in agc2_testing_common.h):
   - `WhiteNoiseGenerator` - ✅ exists
   - `SineGenerator` - ✅ exists
   - `PulseGenerator` - needs porting
   - `ChirpGenerator` - for sweep tests

2. **Audio buffer helpers**:
   - `CreateMonoBuffer()`
   - `CreateStereoBuffer()`
   - `FillWithTestPattern()`

3. **Comparison utilities**:
   - `ExpectNear()` for float arrays
   - `VerifyBitExact()` for reference comparison
   - `ComputeSNR()` for quality metrics

4. **Test fixtures**:
   - `AudioProcessingTestBase` - common setup
   - `AEC3TestFixture` - echo canceller setup
   - `AGC2TestFixture` - gain controller setup

### Directory Structure

```
tests/
├── meson.build
├── test_utils/
│   ├── audio_buffer_tools.h/cc      # ✅ exists
│   ├── bitexactness_tools.h         # ✅ exists
│   ├── signal_generators.h/cc       # NEW
│   ├── test_fixtures.h/cc           # NEW
│   └── comparison_utils.h           # NEW
└── unit/
    ├── rms_level_unittest.cc        # ✅ exists
    ├── audio_buffer_unittest.cc     # ✅ exists
    ├── splitting_filter_unittest.cc # ✅ exists
    ├── high_pass_filter_unittest.cc # ✅ exists
    ├── audio_frame_view_unittest.cc # Phase 1
    ├── gain_control_unittest.cc     # Phase 1
    ├── residual_echo_detector_unittest.cc # Phase 1
    ├── agc2/                         # Phase 2
    │   ├── limiter_unittest.cc
    │   ├── gain_applier_unittest.cc
    │   └── ...
    ├── aec3/                         # Phase 3
    │   ├── echo_canceller3_unittest.cc
    │   ├── aec_state_unittest.cc
    │   └── ...
    ├── ns/                           # Phase 4
    │   └── noise_suppressor_unittest.cc
    └── vad/                          # Phase 4
        ├── voice_activity_detector_unittest.cc
        └── ...
```

## Porting Guidelines

### For Each Test File

1. **Fetch upstream source from M131 (branch-heads/6778):**
   ```bash
   # IMPORTANT: Always use branch-heads/6778 to match M131
   curl -s "https://webrtc.googlesource.com/src/+/refs/branch-heads/6778/modules/audio_processing/PATH_unittest.cc?format=TEXT" \
     | base64 -d > /tmp/upstream_test.cc
   ```

   **Never use `refs/heads/main`** - the main branch has different implementations and reference values.

2. **Adapt includes:**
   - Change `test/gtest.h` → `<gtest/gtest.h>`
   - Change `modules/audio_processing/...` → `webrtc/modules/audio_processing/...`
   - Add `tests/test_utils/...` for local utilities

3. **Handle dependencies:**
   - If test needs data files → skip or create minimal inline data
   - If test needs protobuf → skip or use hardcoded values
   - If test needs threading → consider deferring

4. **Update meson.build:**
   ```meson
   test_sources += files(
     'unit/new_test_unittest.cc',
   )
   ```

5. **Verify reference values match M131:**
   - All `kReference` arrays must come from the M131 test file
   - Do NOT update reference values from a different WebRTC version
   - Run tests to verify they pass with the M131 implementation

## Files to Modify

| File | Changes |
|------|---------|
| `tests/meson.build` | Add new test sources, organize by subdirectory |
| `tests/test_utils/signal_generators.h/cc` | NEW - signal generation |
| `tests/test_utils/test_fixtures.h/cc` | NEW - common fixtures |
| `tests/test_utils/comparison_utils.h` | NEW - comparison helpers |
| `tests/unit/*.cc` | NEW - ported test files |

## Verification

After each phase:

```bash
# Build with tests
meson setup builddir -Dtests=enabled
ninja -C builddir

# Run all tests
meson test -C builddir -v

# Check test count
./builddir/tests/apm_unit_tests --gtest_list_tests | wc -l
```

## Success Criteria

| Phase | Target Test Count | Components Covered |
|-------|-------------------|-------------------|
| Current | 87 | RmsLevel, AudioBuffer, SplittingFilter, HighPassFilter |
| Phase 1 | 120 | + AudioFrameView, GainControl, ResidualEchoDetector |
| Phase 2 | 200 | + AGC2 subsystem |
| Phase 3 | 320 | + AEC3 subsystem |
| Phase 4 | 370 | + NS, VAD |
| Phase 5 | 430 | + Integration tests |

## Version Tracking

**Implementation Version:** WebRTC M131 (branch-heads/6778)
**Test Source:** `https://webrtc.googlesource.com/src/+/refs/branch-heads/6778/`

When upgrading the implementation to a newer WebRTC version (e.g., M136), all test reference values must be updated simultaneously. See `.claude/plans/upgrade-to-m136.md` for the upgrade plan.

## Excluded Tests

Tests that will NOT be ported (require too much infrastructure):

1. **Protobuf-dependent tests:**
   - `echo_control_mobile_bit_exact_unittest.cc` - needs protobuf for reference data
   - Parts of `audio_processing_unittest.cc` that use AEC dumps

2. **Performance tests:**
   - `audio_processing_performance_unittest.cc` - benchmarking infrastructure

3. **Threading tests:**
   - `audio_processing_impl_locking_unittest.cc` - complex threading setup

4. **Tests requiring external data files:**
   - Any tests that load WAV files or reference recordings
