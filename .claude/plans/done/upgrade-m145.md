# Plan: Upgrade WebRTC Audio Processing to M145

**Status: ✅ COMPLETE**

## Overview

Upgraded webrtc-audio-processing from WebRTC M131 (version 2.1) to M145 (version 3.0).

**Previous State:**
- WebRTC M131 (branch-heads/6778)
- Project version 2.1
- 10 patches in `patches/` directory

**Current State:**
- WebRTC M145 (branch-heads/7632)
- Project version 3.0
- 4 patches still needed (6 integrated upstream)
- **2432 unit tests passing** (100%)

---

## Completed Work

### Phase 1: Preparation ✅

- Created `upgrade-m145-v2` branch
- Tagged `pre-upgrade-m145-v2` for rollback safety

### Phase 2: Code Synchronization ✅

Synced directories in order:
1. `webrtc/api/` - New Environment API, BuiltinAudioProcessingBuilder
2. `webrtc/rtc_base/` - Namespace consolidation (rtc:: → webrtc::)
3. `webrtc/common_audio/` - Audio utilities
4. `webrtc/modules/audio_processing/` - Core processing
5. `webrtc/system_wrappers/` - Platform abstractions

### Phase 3: Test Synchronization ✅

- Synced 125 test files from upstream
- Fixed include paths for test utilities
- Created missing test utilities:
  - `create_test_field_trials.h/cc`
  - `rtc_expect_death.h`

### Phase 4: Patch Application ✅

**Patches still needed (4):**
- `0001-arch.h-Add-s390x-support.patch`
- `0001-Add-support-for-BSD-systems.patch`
- `0001-Fix-up-XMM-intrinsics-usage-on-MSVC.patch`
- `0001-Some-fixes-for-MinGW.patch`

**Patches now upstream (6):**
- `0001-AECM-MIPS-Use-uintptr_t-for-pointer-arithmetic.patch`
- `0001-common_audio-Add-MIPS_DSP_R1_LE-guard-for-vector-sca.patch`
- `0001-Fix-compilation-with-gcc-15.patch`
- `0001-meson-Fixes-for-MSVC-build.patch`
- `0001-Allow-disabling-inline-SSE.patch`
- `0001-Fix-build-with-abseil-cpp-202508.patch`

### Phase 5: Build Fixes ✅

Key changes made:
- C++20 now required (was C++17) for `requires` clauses
- Added missing files: `string_format.h`, Environment API files, `clock.cc`
- Fixed namespace changes (`rtc::` → `webrtc::` for ArrayView, scoped_refptr, split)
- Fixed trace_event.h perfetto includes
- Added WEBRTC_APM_DEBUG_DUMP=0 flag
- Changed to `link_whole` for static libraries (symbol export fix)
- Fixed vtable anchor for BuiltinAudioProcessingBuilder
- Replaced `flat_map` with `std::map` in field_trials
- Added `field_trials.cc`, `field_trials_registry.cc` to API library
- Added `unused.h` header

### Phase 6: Test Fixes ✅

- Regenerated `output_data_float.pb` for M145 algorithm changes
- Modified `CreateLpResidualAndPitchInfoReader()` to use ARM64-specific reference data

---

## Breaking Changes (API)

1. **C++20 Required** - Previously C++17
2. **New Environment API**:
   ```cpp
   // Before (M131):
   auto apm = webrtc::AudioProcessingBuilder()
       .SetConfig(config)
       .Create();

   // After (M145):
   webrtc::Environment env = webrtc::CreateEnvironment();
   auto apm = webrtc::BuiltinAudioProcessingBuilder(config).Build(env);
   ```
3. **Namespace consolidation** - Some types moved from `rtc::` to `webrtc::`

---

## Commits (13 total)

1. `sync: webrtc/api from upstream M145`
2. `sync: webrtc/rtc_base from upstream M145`
3. `sync: webrtc/common_audio from upstream M145`
4. `sync: webrtc/modules/audio_processing from upstream M145`
5. `sync: webrtc/system_wrappers from upstream M145`
6. `sync: tests from upstream M145`
7. `patch: apply 4 working patches`
8. `patch: apply BSD, MinGW, and MSVC fixes for M145`
9. `fix: build issues for M145 upgrade`
10. `fix: test build issues for M145 upgrade`
11. `release: upgrade to WebRTC M145 (version 3.0)`
12. `fix: update test reference data and ARM64 compatibility`

---

## Test Results

- **2432 tests total**
- **2432 tests pass (100%)**
- **0 failures**

---

## Files Modified

Key files:
- `meson.build` - C++20, version 3.0
- `NEWS` - Release notes with migration guide
- `webrtc/api/meson.build` - New source files
- `webrtc/modules/audio_processing/meson.build` - link_whole, new sources
- `tests/meson.build` - WEBRTC_ENABLE_PROTOBUF, new test utils
- `tests/test_utils/rnn_vad_test_utils.cc` - ARM64 reference data selection
- Multiple header fixes for include paths and namespaces

---

## Lessons Learned

1. **Commit frequently** - Each sync step should be committed immediately
2. **link_whole vs dependencies** - Static libraries need `link_whole` for proper symbol export
3. **Platform-specific reference data** - ARM64 produces different floating-point results
4. **Namespace changes** - Major version upgrades often consolidate namespaces
5. **Test utilities** - Many test helpers need to be copied/adapted from upstream
