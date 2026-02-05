# Plan: Improve Test Coverage at Function Level

**Status: ✅ COMPLETE**

## Overview

Expand unit test coverage by porting upstream WebRTC M131 tests.

**Final State:**
- **2458 tests passing, 37 skipped, 185 disabled** (2495 total ran)
- All tests match upstream exactly (with only include path conversions)
- Test infrastructure complete (mocks, utilities, field trials, fake_clock, protobuf)
- NEON optimizations enabled on ARM64 Mac
- Platform-specific reference files for ApmTest.Process

**Implementation Version:** WebRTC M131 (branch-heads/6778)

---

## Final Test Count by Category

| Category | Test Files | Status |
|----------|------------|--------|
| Root level | 14 | ✅ Complete |
| AEC3 | 42 | ✅ Complete |
| AGC1 | 2 | ✅ Complete |
| AGC2 | 18 | ✅ Complete |
| AGC2/RNN_VAD | 15 | ✅ Complete |
| Capture Levels | 2 | ✅ Complete |
| Echo Detector | 4 | ✅ Complete |
| NS | 1 | ✅ Complete |
| VAD | 8 | ✅ Complete |
| Utility | 3 | ✅ Complete |
| Common Audio | 17 | ✅ Complete |
| Test Utils | 1 | ✅ Complete |
| **TOTAL** | **127 files** | **Complete** |

---

## Tests Not Ported (Infrastructure Limitations)

| Test | Reason |
|------|--------|
| `aec_dump/aec_dump_integration_test.cc` | Requires AEC dump infrastructure + task queue |
| `aec_dump/aec_dump_unittest.cc` | Requires AEC dump infrastructure + task queue |
| `audio_processing_impl_locking_unittest.cc` | Requires threading primitives |
| `audio_processing_performance_unittest.cc` | Benchmarking infrastructure |
| `test/conversational_speech/generator_unittest.cc` | Complex test tooling |
| `test/debug_dump_test.cc` | Requires debug dump replayer |
| `test/echo_canceller3_config_json_unittest.cc` | Requires JSON config parser |
| `test/fake_recording_device_unittest.cc` | Source not in project |

---

## Test Infrastructure Created

### Test Utilities (`tests/test_utils/`)
- `audio_buffer_tools.h/cc` - Audio buffer helpers
- `audio_processing_builder_for_testing.h/cc` - APM builder for tests
- `bitexactness_tools.h/cc` - Bit-exact comparison
- `echo_canceller_test_tools.h/cc` - RandomizeSampleVector, DelayBuffer
- `echo_control_mock.h` - MockEchoControl
- `fake_clock.h/cc` - rtc::ScopedFakeClock
- `field_trial.h/cc` - ScopedFieldTrials
- `file_utils.h/cc` - File I/O, ResourcePath, TempFilename
- `input_audio_file.h/cc` - Audio file reading
- `performance_timer.h/cc` - Test timing
- `protobuf_utils.h/cc` - Protobuf helpers
- `rnn_vad_test_utils.h/cc` - RNN_VAD test helpers
- `rtc_expect_death.h` - Death test macro
- `task_queue_for_test.h` - TaskQueueForTest stub
- `test_utils.h/cc` - Int16FrameData, ChannelBufferVectorWriter
- `thread_stub.h` - rtc::Thread stub

### Mock Classes (`tests/test_utils/mock/`)
- `mock_block_processor.h/cc`
- `mock_echo_remover.h/cc`
- `mock_render_delay_buffer.h/cc`
- `mock_render_delay_controller.h/cc`

### Test Resources (`tests/resources/`)
- AGC audio data files
- RNN_VAD binary test data
- Stereo PCM files (8kHz to 192kHz)
- Protobuf reference files (platform-specific)
- AEC dump files

---

## Build System Changes

1. **NEON auto-enable on ARM64**: `meson.build` uses `neon_opt.allowed()` instead of `neon_opt.enabled()`
2. **NEON FMA precision**: Added `-ffp-contract=fast` for NEON builds
3. **Protobuf support**: Added protobuf code generation for test data
4. **Abseil flags**: Added `absl::ParseCommandLine()` for `--write_apm_ref_data`

---

## Platform-Specific Reference Files

| Platform | Reference File |
|----------|----------------|
| x86_64 (no AVX2) | `output_data_float.pb` |
| x86_64 (AVX2) | `output_data_float_avx2.pb` |
| Intel Mac | `output_data_mac.pb` |
| ARM64 Mac | `output_data_mac_arm64.pb` |
| Fixed-point | `output_data_fixed.pb` |

---

## Completion Date

February 5, 2025
