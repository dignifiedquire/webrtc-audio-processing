/*
 *  Copyright 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "tests/test_utils/fake_clock.h"

#include "webrtc/rtc_base/checks.h"

namespace rtc {

int64_t FakeClock::TimeNanos() const {
  webrtc::MutexLock lock(&lock_);
  return time_ns_;
}

void FakeClock::SetTime(webrtc::Timestamp new_time) {
  webrtc::MutexLock lock(&lock_);
  RTC_DCHECK(new_time.us() * 1000 >= time_ns_);
  time_ns_ = new_time.us() * 1000;
}

void FakeClock::AdvanceTime(webrtc::TimeDelta delta) {
  webrtc::MutexLock lock(&lock_);
  time_ns_ += delta.ns();
}

ScopedFakeClock::ScopedFakeClock() {
  prev_clock_ = SetClockForTesting(this);
}

ScopedFakeClock::~ScopedFakeClock() {
  SetClockForTesting(prev_clock_);
}

}  // namespace rtc
