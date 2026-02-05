# Plans Overview

This directory contains implementation plans for webrtc-audio-processing development.

## Directory Structure

- `todo/` - Upcoming plans awaiting implementation
- `done/` - Completed plans for reference

## Current Plans

### In Progress / Todo

| Plan | Description | Status |
|------|-------------|--------|
| [rust-port.md](todo/rust-port.md) | Full Rust port with C API, property tests, SIMD optimizations | Planning complete |
| [upgrade-m136.md](todo/upgrade-m136.md) | Upgrade from WebRTC M131 to M136 | Ready to start |

### Completed

| Plan | Description | Completed |
|------|-------------|-----------|
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

- **Current Version**: 2.1 (WebRTC M131)
- **Test Count**: 2458 passing, 37 skipped, 185 disabled
- **Next Milestone**: Version 2.2 (WebRTC M136)
