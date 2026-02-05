/*
 *  Copyright (c) 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef TESTS_TEST_UTILS_PERFORMANCE_TIMER_H_
#define TESTS_TEST_UTILS_PERFORMANCE_TIMER_H_

#include <chrono>
#include <optional>
#include <vector>

namespace webrtc {
namespace test {

class PerformanceTimer {
 public:
  explicit PerformanceTimer(int num_frames_to_process);
  ~PerformanceTimer();

  void StartTimer();
  void StopTimer();

  double GetDurationAverage() const;
  double GetDurationStandardDeviation() const;

  // These methods are the same as those above, but they ignore the first
  // `number_of_warmup_samples` measurements.
  double GetDurationAverage(size_t number_of_warmup_samples) const;
  double GetDurationStandardDeviation(size_t number_of_warmup_samples) const;

 private:
  std::optional<std::chrono::steady_clock::time_point> start_timestamp_;
  std::vector<int64_t> timestamps_us_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TESTS_TEST_UTILS_PERFORMANCE_TIMER_H_
