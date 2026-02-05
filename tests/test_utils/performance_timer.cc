/*
 *  Copyright (c) 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "tests/test_utils/performance_timer.h"

#include <cmath>
#include <numeric>

namespace webrtc {
namespace test {

PerformanceTimer::PerformanceTimer(int num_frames_to_process)
    : timestamps_us_() {
  timestamps_us_.reserve(num_frames_to_process);
}

PerformanceTimer::~PerformanceTimer() = default;

void PerformanceTimer::StartTimer() {
  start_timestamp_ = std::chrono::steady_clock::now();
}

void PerformanceTimer::StopTimer() {
  auto end = std::chrono::steady_clock::now();
  if (start_timestamp_) {
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - *start_timestamp_).count();
    timestamps_us_.push_back(duration_us);
  }
}

double PerformanceTimer::GetDurationAverage() const {
  return GetDurationAverage(0);
}

double PerformanceTimer::GetDurationStandardDeviation() const {
  return GetDurationStandardDeviation(0);
}

double PerformanceTimer::GetDurationAverage(
    size_t number_of_warmup_samples) const {
  if (timestamps_us_.size() <= number_of_warmup_samples) {
    return 0.0;
  }
  const size_t num_samples = timestamps_us_.size() - number_of_warmup_samples;
  return static_cast<double>(
             std::accumulate(timestamps_us_.begin() + number_of_warmup_samples,
                             timestamps_us_.end(), static_cast<int64_t>(0))) /
         num_samples;
}

double PerformanceTimer::GetDurationStandardDeviation(
    size_t number_of_warmup_samples) const {
  if (timestamps_us_.size() <= number_of_warmup_samples) {
    return 0.0;
  }
  const double average = GetDurationAverage(number_of_warmup_samples);
  double variance = 0.0;
  for (size_t i = number_of_warmup_samples; i < timestamps_us_.size(); ++i) {
    const double diff = timestamps_us_[i] - average;
    variance += diff * diff;
  }
  const size_t num_samples = timestamps_us_.size() - number_of_warmup_samples;
  return std::sqrt(variance / num_samples);
}

}  // namespace test
}  // namespace webrtc
