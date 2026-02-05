/*
 *  Copyright 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef TESTS_TEST_UTILS_FAKE_CLOCK_H_
#define TESTS_TEST_UTILS_FAKE_CLOCK_H_

#include <stdint.h>

#include "webrtc/api/units/time_delta.h"
#include "webrtc/api/units/timestamp.h"
#include "webrtc/rtc_base/synchronization/mutex.h"
#include "webrtc/rtc_base/thread_annotations.h"
#include "webrtc/rtc_base/time_utils.h"

namespace rtc {

// Fake clock for use with unit tests, which does not tick on its own.
// Starts at time 0.
class FakeClock : public webrtc::ClockInterface {
 public:
  FakeClock() = default;
  FakeClock(const FakeClock&) = delete;
  FakeClock& operator=(const FakeClock&) = delete;
  ~FakeClock() override = default;

  // webrtc::ClockInterface implementation.
  int64_t TimeNanos() const override;

  // Methods that can be used by the test to control the time.
  void SetTime(webrtc::Timestamp new_time);
  void AdvanceTime(webrtc::TimeDelta delta);

 private:
  mutable webrtc::Mutex lock_;
  int64_t time_ns_ RTC_GUARDED_BY(lock_) = 0;
};

// Simplified version for tests - no thread message queue processing.
// In the full WebRTC, this would call ThreadManager::ProcessAllMessageQueuesForTesting()
// but we don't have that infrastructure, and the smoothing_filter tests don't need it.
class ScopedFakeClock : public FakeClock {
 public:
  ScopedFakeClock();
  ~ScopedFakeClock() override;

 private:
  webrtc::ClockInterface* prev_clock_;
};

}  // namespace rtc

#endif  // TESTS_TEST_UTILS_FAKE_CLOCK_H_
