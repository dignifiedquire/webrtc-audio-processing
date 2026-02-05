/*
 *  Copyright (c) 2024 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <gtest/gtest.h>

#include "absl/flags/parse.h"
#include "system_wrappers/include/metrics.h"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  
  // Parse abseil flags after gtest (gtest modifies argc/argv)
  absl::ParseCommandLine(argc, argv);
  
  // Enable metrics collection for tests that require it
  webrtc::metrics::Enable();
  
  return RUN_ALL_TESTS();
}
