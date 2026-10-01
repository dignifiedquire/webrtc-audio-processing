
/*
 *  Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "modules/audio_processing/aec3/moving_average.h"

#include <algorithm>
#include <cstddef>
#include <functional>

#include "api/array_view.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace aec3 {

MovingAverage::MovingAverage(size_t num_elem, size_t mem_len)
    : num_elem_(num_elem),
      mem_len_(mem_len - 1),
      memory_(num_elem * mem_len_, 0.f),
      mem_index_(0),
      number_updates_(0) {
  RTC_DCHECK(num_elem_ > 0);
  RTC_DCHECK(mem_len > 0);
}

MovingAverage::~MovingAverage() = default;

void MovingAverage::Average(ArrayView<const float> input,
                            ArrayView<float> output) {
  RTC_DCHECK(input.size() == num_elem_);
  RTC_DCHECK(output.size() == num_elem_);

  // Sum all contributions.
  std::copy(input.begin(), input.end(), output.begin());
  for (auto i = memory_.begin(); i < memory_.end(); i += num_elem_) {
    std::transform(i, i + num_elem_, output.begin(), output.begin(),
                   std::plus<float>());
  }

  // Divide by the number of points used to compute the average.
  const float scaling = 1.0f / static_cast<float>(number_updates_ + 1);
  for (float& o : output) {
    o *= scaling;
  }

  // Update memory.
  if (mem_len_ > 0) {
    std::copy(input.begin(), input.end(),
              memory_.begin() + mem_index_ * num_elem_);
    mem_index_ = (mem_index_ + 1) % mem_len_;
  }
  number_updates_ = std::min(static_cast<int>(mem_len_), number_updates_ + 1);
}

void MovingAverage::UpdateMemoryLength(size_t mem_len) {
  if (mem_len_ + 1 == mem_len) {
    return;
  }
  RTC_DCHECK(mem_len > 0);
  mem_len_ = mem_len - 1;
  memory_.resize(num_elem_ * mem_len_);
  std::fill(memory_.begin(), memory_.end(), 0.0f);
  mem_index_ = 0;
  number_updates_ = 0;
}

}  // namespace aec3
}  // namespace webrtc
