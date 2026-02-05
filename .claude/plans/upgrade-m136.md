# Plan: Upgrade WebRTC Audio Processing to Latest Version

## Overview

Upgrade webrtc-audio-processing from WebRTC M131 (version 2.1) to M136 (version 2.2), with M145+ as a potential follow-up release.

**Current State:**
- WebRTC M131 (branch-heads/6778)
- Project version 2.1
- 10 patches in `patches/` directory
- **456 unit tests passing** (expanded from original 87)

### Test Porting Progress (Pre-Upgrade Work)

Before upgrading, we've been expanding test coverage by porting upstream M131 tests. This ensures we have comprehensive test coverage to validate the upgrade.

**Tests Added:**
- AGC2 tests: limiter_db_gain_curve, interpolated_gain_curve, clipping_predictor_level_buffer, saturation_protector_buffer, speech_probability_buffer
- AEC3 tests: echo_path_variability, block_delay_buffer, aec3_fft, alignment_mixer, api_call_jitter_metrics, multi_channel_content_detector, block_framer, frame_blocker, comfort_noise_generator, suppression_gain, erle_estimator, render_signal_analyzer, suppression_filter, residual_echo_estimator, reverb_model_estimator, subtractor, signal_dependent_erle_estimator, echo_remover, render_delay_buffer, adaptive_fir_filter, block_processor

**Test Infrastructure Added:**
- `tests/test_utils/echo_canceller_test_tools.h/cc` - RandomizeSampleVector, DelayBuffer utilities
- `tests/test_utils/mock/mock_echo_remover.h/cc` - Mock for EchoRemover
- `tests/test_utils/mock/mock_render_delay_buffer.h/cc` - Mock for RenderDelayBuffer  
- `tests/test_utils/mock/mock_render_delay_controller.h/cc` - Mock for RenderDelayController
- Metrics enabled in test builds for metrics-dependent tests

**Remaining Tests to Port:**
- AGC2: clipping_predictor, input_volume_controller, input_volume_stats_reporter (3 tests)
- AEC3: matched_filter, echo_canceller3, aec_state, and others (~10-15 more)
- NS/VAD tests

**Target:**
- WebRTC M136 (branch-heads/7103) - Recommended first step
- Lower risk, ~5 milestone versions of changes
- M145 can be evaluated afterward for version 3.0

## Phase 1: Preparation

### 1.1 Get the exact WebRTC commit from Chromium DEPS

The code must be synced against the exact revision Chromium uses, not just a branch head.

```bash
# Get Chromium M136 DEPS file
curl -s "https://chromium.googlesource.com/chromium/src/+/refs/branch-heads/6778/DEPS?format=TEXT" | base64 -d | grep webrtc_revision

# This gives you the exact commit hash, e.g.:
# 'webrtc_revision': 'abc123def456...'
```

Then checkout that specific commit:
```bash
git clone https://webrtc.googlesource.com/src webrtc-upstream
cd webrtc-upstream
git checkout <exact-commit-hash-from-deps>
```

**Important:** Using the exact commit ensures reproducibility and matches Chromium's tested configuration.

### 1.2 Create upgrade branch

```bash
cd /path/to/webrtc-audio-processing
git checkout -b upgrade-m136 master
git tag pre-m136-upgrade
```

### 1.3 Document current state for comparison

```bash
# Save current API
cp webrtc/api/audio/audio_processing.h /tmp/api-m131.h

# Save current library ABI for later comparison
mkdir -p /tmp/abi-check
cp builddir/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.so /tmp/abi-check/old.so
```

## Phase 2: Code Synchronization

### 2.1 Compare directories with Meld

Per UPDATING.md, compare against the Chromium third_party path:
```bash
meld webrtc-audio-processing/webrtc chromium/third_party/webrtc
```

**Note:** Work directory-by-directory through the `webrtc-audio-processing` tree, finding corresponding code in the Chromium tree.

### 2.2 Sync order (by priority)

| Directory | Priority | Notes |
|-----------|----------|-------|
| `webrtc/api/audio/` | HIGH | Public API - check for breaking changes |
| `webrtc/modules/audio_processing/` | HIGH | Core processing code |
| `webrtc/common_audio/` | MEDIUM | Utilities |
| `webrtc/rtc_base/` | MEDIUM | Base utilities |
| `webrtc/system_wrappers/` | LOW | Platform abstractions |
| `webrtc/third_party/` | LOW | External libs (pffft, rnnoise) |

### 2.3 Identifying file changes

**New files added upstream:**
```bash
# Compare file lists to find new files
diff <(cd webrtc-audio-processing/webrtc && find . -name "*.cc" -o -name "*.h" | sort) \
     <(cd chromium/third_party/webrtc && find . -name "*.cc" -o -name "*.h" | sort) \
     | grep "^>" | head -50
```

**Files removed upstream:**
```bash
diff <(cd webrtc-audio-processing/webrtc && find . -name "*.cc" -o -name "*.h" | sort) \
     <(cd chromium/third_party/webrtc && find . -name "*.cc" -o -name "*.h" | sort) \
     | grep "^<"
```

For each new file needed, add it to the appropriate `meson.build`.

### 2.4 Sync workflow (correct order)

1. **First:** Copy all upstream changes to get clean upstream code
2. **Second:** Attempt to build (will likely fail)
3. **Third:** Apply/adapt patches to fix build
4. **Fourth:** Build again and iterate

**Do NOT try to preserve patches during sync** - they are applied afterward.

### 2.5 Files to skip

- `*_unittest.cc`, `*_test.cc` - test files (not built, we port separately)
- `BUILD.gn` - copy for reference only, not used by Meson
- `DEPS`, `OWNERS` - Chromium-specific metadata

## Phase 3: Patch Reapplication

### 3.1 Patch risk assessment

**Low Risk (apply first):**
- `0001-arch.h-Add-s390x-support.patch`
- `0001-AECM-MIPS-Use-uintptr_t-for-pointer-arithmetic.patch`
- `0001-common_audio-Add-MIPS_DSP_R1_LE-guard-for-vector-sca.patch`
- `0001-Fix-compilation-with-gcc-15.patch`
- `0001-Some-fixes-for-MinGW.patch`

**Medium Risk:**
- `0001-Add-support-for-BSD-systems.patch`
- `0001-Fix-up-XMM-intrinsics-usage-on-MSVC.patch`
- `0001-meson-Fixes-for-MSVC-build.patch`

**High Risk (may need rework or removal):**
- `0001-Allow-disabling-inline-SSE.patch` - touches 9 files
- `0001-Fix-build-with-abseil-cpp-202508.patch` - check if upstream fixed this

### 3.2 Apply and regenerate patches

```bash
for patch in patches/*.patch; do
  git apply --check "$patch" || echo "FAILED: $patch"
done
```

For failed patches, manually adapt and regenerate:
```bash
# After manual fixes
git diff > patches/0001-Updated-patch-name.patch
```

## Phase 4: Build System Updates

### 4.1 Update meson.build files

Check for new/removed source files in:
- `webrtc/meson.build`
- `webrtc/modules/audio_processing/meson.build`
- `webrtc/common_audio/meson.build`

Cross-reference with upstream `BUILD.gn` files to identify:
- New source files to add
- Removed source files to delete
- New subdirectories needing their own `meson.build`

### 4.2 Verify abseil-cpp requirements

Check what abseil version M136 actually requires:
```bash
# In upstream WebRTC, check for abseil usage
grep -r "absl::" webrtc-upstream/modules/audio_processing/ | head -20
# Check Chromium's DEPS for abseil version
```

Update `meson.build` if needed:
```meson
dependency('absl_base', version: '>=REQUIRED_VERSION')
```

### 4.3 Update version and soversion

```meson
project('webrtc-audio-processing', 'c', 'cpp',
  version : '2.2',
  ...
)
```

### 4.4 ABI compatibility check

Use `abidiff` to detect breaking ABI changes:
```bash
# Build new library
ninja -C builddir

# Compare ABIs
abidiff /tmp/abi-check/old.so \
        builddir/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.so

# If incompatible, bump soversion according to libtool rules:
# - Interfaces removed/changed: increment current, reset revision, reset age
# - Interfaces added: increment current, reset revision, increment age
# - No interface change: increment revision only
```

## Phase 5: API Change Assessment

### 5.1 Compare audio_processing.h

```bash
diff /tmp/api-m131.h webrtc/api/audio/audio_processing.h
```

### 5.2 Key areas to check

- `AudioProcessing::Config` structure changes
- `HighPassFilter` configuration
- `GainController1` / `GainController2` settings
- `EchoCanceller3` configuration
- Any removed/deprecated features

### 5.3 Update NEWS file

Document all changes, API differences, and migration notes.

## Phase 6: Update Test Reference Values

**Critical:** Test reference values are version-specific. The bit-exact tests contain expected output values that must match the implementation version.

### 6.1 Extract test reference values from upstream M136

For each test file in `tests/unit/`, fetch the matching upstream test:
```bash
# Example for high_pass_filter
curl -s "https://webrtc.googlesource.com/src/+/refs/branch-heads/7103/modules/audio_processing/high_pass_filter_unittest.cc?format=TEXT" \
  | base64 -d > /tmp/upstream_hpf_test.cc

# Compare reference arrays
diff tests/unit/high_pass_filter_unittest.cc /tmp/upstream_hpf_test.cc
```

### 6.2 Update reference values in test files

Update `kReference` arrays in:
- `tests/unit/high_pass_filter_unittest.cc`
- `tests/unit/splitting_filter_unittest.cc`
- Any other bit-exact tests

The reference values must come from the **same WebRTC version** as the implementation, otherwise tests will fail.

### 6.3 Port any new upstream tests

Check for new test coverage upstream that should be added.

## Phase 7: Build and Test

### 7.1 Build and run unit tests

```bash
meson setup builddir -Dtests=enabled
ninja -C builddir
meson test -C builddir -v
```

### 7.2 Integration test with PulseAudio

```bash
# Install locally
meson setup builddir -Dprefix=$(pwd)/install
ninja -C builddir install

# Build PulseAudio against it
cd /path/to/pulseaudio
meson setup build -Dpkg_config_path=/path/to/install/lib64/pkgconfig/
ninja -C build

# Test module-echo-cancel
pulseaudio --start
pactl load-module module-echo-cancel
# Run audio tests
```

### 7.3 Platform testing matrix

| Platform | Priority |
|----------|----------|
| x86_64 Linux | HIGH |
| aarch64 Linux | HIGH |
| 32-bit ARM | MEDIUM |
| FreeBSD | MEDIUM |
| Windows MSVC | MEDIUM |

## Files to Modify

| File | Changes |
|------|---------|
| `meson.build` | Update version to 2.2, check deps |
| `webrtc/meson.build` | Add/remove source files |
| `webrtc/modules/audio_processing/meson.build` | Add/remove source files |
| `NEWS` | Add release notes for 2.2 |
| `patches/*.patch` | Regenerate as needed |
| `webrtc/**/*.cc` | Sync with upstream M136 |
| `webrtc/**/*.h` | Sync with upstream M136 |

## Rollback Strategy

If issues arise:
```bash
git checkout pre-m136-upgrade
# Or revert specific files
git checkout pre-m136-upgrade -- webrtc/path/to/file
```

## Verification Checklist

```bash
# 1. Full build with tests
meson setup builddir -Dtests=enabled
ninja -C builddir

# 2. Run all unit tests (should pass)
meson test -C builddir -v

# 3. Check library exports
nm -D builddir/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.so | grep AudioProcessing

# 4. Test with example application
./builddir/examples/run-offline input.wav output.wav

# 5. Verify ABI compatibility (if soversion unchanged)
abidiff /tmp/abi-check/old.so \
        builddir/webrtc/modules/audio_processing/libwebrtc-audio-processing-2.so

# 6. Verify patches are tracked
ls patches/*.patch  # All patches should be updated/regenerated
```

## Success Criteria

- [x] Comprehensive test coverage ported from upstream (456 tests, up from 87)
- [x] Test infrastructure (mocks, utilities) in place
- [ ] All unit tests pass after upgrade
- [ ] Library builds on x86_64 Linux
- [ ] All 10 patches applied (or removed if fixed upstream)
- [ ] NEWS file updated with changes
- [ ] Version bumped to 2.2
- [ ] Soversion updated if ABI changed
- [ ] Test reference values match M136 upstream
