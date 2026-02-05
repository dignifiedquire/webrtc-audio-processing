/*
 *  Copyright (c) 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef TESTS_TEST_UTILS_BITEXACTNESS_TOOLS_H_
#define TESTS_TEST_UTILS_BITEXACTNESS_TOOLS_H_

#include <cmath>
#include <cstddef>
#include <vector>

namespace webrtc {
namespace test {

// Verifies that two deinterleaved arrays are element-wise within a tolerance.
// Returns true if all elements match within the specified error bound.
inline bool VerifyDeinterleavedArray(size_t samples_per_channel,
                                     size_t num_channels,
                                     const std::vector<float>& reference,
                                     const std::vector<float>& output,
                                     float element_error_bound) {
  if (reference.size() != output.size()) {
    return false;
  }
  
  if (reference.size() != samples_per_channel * num_channels) {
    return false;
  }

  for (size_t i = 0; i < reference.size(); ++i) {
    if (std::abs(reference[i] - output[i]) > element_error_bound) {
      return false;
    }
  }
  return true;
}

// Verifies that two interleaved arrays are element-wise within a tolerance.
inline bool VerifyInterleavedArray(size_t length,
                                   const std::vector<float>& reference,
                                   const std::vector<float>& output,
                                   float element_error_bound) {
  if (reference.size() != output.size() || reference.size() != length) {
    return false;
  }

  for (size_t i = 0; i < length; ++i) {
    if (std::abs(reference[i] - output[i]) > element_error_bound) {
      return false;
    }
  }
  return true;
}

}  // namespace test
}  // namespace webrtc

#endif  // TESTS_TEST_UTILS_BITEXACTNESS_TOOLS_H_
