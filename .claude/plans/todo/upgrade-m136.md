# Plan: Upgrade WebRTC Audio Processing to M136

**Status: 🔄 TODO**

## Overview

Upgrade webrtc-audio-processing from WebRTC M131 (version 2.1) to M136 (version 2.2), with M145+ as a potential follow-up release.

**Current State:**
- WebRTC M131 (branch-heads/6778)
- Project version 2.1
- 10 patches in `patches/` directory
- **2458 unit tests passing** (comprehensive coverage)

**Target:**
- WebRTC M136 (branch-heads/7103)
- Lower risk, ~5 milestone versions of changes
- M145 can be evaluated afterward for version 3.0

---

## Prerequisites (Complete)

✅ Comprehensive test coverage ported from upstream M131
✅ Test infrastructure (mocks, utilities, protobuf) in place
✅ Platform-specific reference files for ARM64 Mac
✅ NEON optimizations enabled on ARM64

---

## Phase 1: Preparation

### 1.1 Get the exact WebRTC commit from Chromium DEPS

```bash
# Get Chromium M136 DEPS file
curl -s "https://chromium.googlesource.com/chromium/src/+/refs/branch-heads/7103/DEPS?format=TEXT" | base64 -d | grep webrtc_revision
```

### 1.2 Create upgrade branch

```bash
git checkout -b upgrade-m136 main
git tag pre-m136-upgrade
```

### 1.3 Document current state

```bash
cp webrtc/api/audio/audio_processing.h /tmp/api-m131.h
```

---

## Phase 2: Code Synchronization

### 2.1 Sync order (by priority)

| Directory | Priority | Notes |
|-----------|----------|-------|
| `webrtc/api/audio/` | HIGH | Public API - check for breaking changes |
| `webrtc/modules/audio_processing/` | HIGH | Core processing code |
| `webrtc/common_audio/` | MEDIUM | Utilities |
| `webrtc/rtc_base/` | MEDIUM | Base utilities |
| `webrtc/system_wrappers/` | LOW | Platform abstractions |

### 2.2 Sync workflow

1. Copy upstream changes
2. Attempt build (will fail)
3. Apply/adapt patches
4. Iterate

---

## Phase 3: Patch Reapplication

### Patch risk assessment

**Low Risk:**
- `0001-arch.h-Add-s390x-support.patch`
- `0001-AECM-MIPS-Use-uintptr_t-for-pointer-arithmetic.patch`
- `0001-common_audio-Add-MIPS_DSP_R1_LE-guard-for-vector-sca.patch`
- `0001-Fix-compilation-with-gcc-15.patch`
- `0001-Some-fixes-for-MinGW.patch`

**Medium Risk:**
- `0001-Add-support-for-BSD-systems.patch`
- `0001-Fix-up-XMM-intrinsics-usage-on-MSVC.patch`
- `0001-meson-Fixes-for-MSVC-build.patch`

**High Risk:**
- `0001-Allow-disabling-inline-SSE.patch`
- `0001-Fix-build-with-abseil-cpp-202508.patch`

---

## Phase 4: Build System Updates

- Update `meson.build` files for new/removed sources
- Verify abseil-cpp requirements
- Update version to 2.2
- Check ABI compatibility

---

## Phase 5: Test Updates

- Update test reference values for M136
- Regenerate platform-specific reference files
- Port any new upstream tests

---

## Phase 6: Verification

```bash
meson setup builddir -Dtests=enabled
ninja -C builddir
meson test -C builddir -v
```

---

## Success Criteria

- [ ] All 2458+ unit tests pass after upgrade
- [ ] Library builds on x86_64 Linux
- [ ] All patches applied (or removed if fixed upstream)
- [ ] NEWS file updated
- [ ] Version bumped to 2.2
- [ ] Test reference values match M136 upstream
