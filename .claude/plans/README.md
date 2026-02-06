# Plans Overview

This directory contains implementation plans for webrtc-audio-processing development.

## Directory Structure

- `todo/` - Upcoming plans awaiting implementation
- `done/` - Completed plans for reference

## Current Plans

### In Progress / Todo

| Plan | Description | Status |
|------|-------------|--------|
| [rust-port.md](todo/rust-port.md) | Full Rust port master plan (overview) | Planning complete |
| [rust-port/](todo/rust-port/) | Per-phase detailed plans (10 phases, ~101 commits) | Planning complete |

#### Rust Port Phases

| Phase | Plan | Duration | Status |
|-------|------|----------|--------|
| 1 | [Foundation Infrastructure](todo/rust-port/phase-01-foundation.md) | 2-3 weeks | Not Started |
| 2 | [Common Audio Primitives](todo/rust-port/phase-02-common-audio.md) | 3-4 weeks | Not Started |
| 3 | [Voice Activity Detection](todo/rust-port/phase-03-vad.md) | 2 weeks | Not Started |
| 4 | [Automatic Gain Control](todo/rust-port/phase-04-agc.md) | 4-5 weeks | Not Started |
| 5 | [Noise Suppression](todo/rust-port/phase-05-noise-suppression.md) | 2-3 weeks | Not Started |
| 6 | [Echo Cancellation (AEC3)](todo/rust-port/phase-06-echo-cancellation.md) | 6-8 weeks | Not Started |
| 7 | [Mobile Echo Control (AECM)](todo/rust-port/phase-07-aecm.md) | 1-2 weeks | Not Started |
| 8 | [Audio Processing Integration](todo/rust-port/phase-08-integration.md) | 3-4 weeks | Not Started |
| 9 | [C API & Final Integration](todo/rust-port/phase-09-c-api.md) | 2-3 weeks | Not Started |
| 10 | [Documentation & Release](todo/rust-port/phase-10-docs-release.md) | 1-2 weeks | Not Started |

### Completed

| Plan | Description | Completed |
|------|-------------|-----------|
| [upgrade-m145.md](done/upgrade-m145.md) | Upgrade from WebRTC M131 to M145 (version 3.0) | 2025-02 |
| [improve-test-coverage.md](done/improve-test-coverage.md) | Port upstream tests, expand from 87 to 2458 tests | 2025 |

## Plan Format

Each plan should include:

1. **Overview** - Brief description and goals
2. **Current State** - Starting point and context
3. **Phases** - Ordered steps with clear tasks
4. **Files to Modify** - List of affected files
5. **Verification Checklist** - How to confirm success
6. **Success Criteria** - Definition of done

## Project Status

- **Current Version**: 3.0 (WebRTC M145)
- **Test Count**: 2432 passing, 185 disabled
- **Branch**: main
