/*
 *  Copyright (c) 2017 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef TESTS_TEST_UTILS_MOCK_MOCK_ECHO_REMOVER_H_
#define TESTS_TEST_UTILS_MOCK_MOCK_ECHO_REMOVER_H_

#include <optional>
#include <vector>

#include "webrtc/modules/audio_processing/aec3/echo_path_variability.h"
#include "webrtc/modules/audio_processing/aec3/echo_remover.h"
#include "webrtc/modules/audio_processing/aec3/render_buffer.h"
#include <gmock/gmock.h>

namespace webrtc {
namespace test {

class MockEchoRemover : public EchoRemover {
 public:
  MockEchoRemover();
  virtual ~MockEchoRemover();

  MOCK_METHOD(void,
              ProcessCapture,
              (EchoPathVariability echo_path_variability,
               bool capture_signal_saturation,
               const std::optional<DelayEstimate>& delay_estimate,
               RenderBuffer* render_buffer,
               Block* linear_output,
               Block* capture),
              (override));
  MOCK_METHOD(void,
              UpdateEchoLeakageStatus,
              (bool leakage_detected),
              (override));
  MOCK_METHOD(void,
              GetMetrics,
              (EchoControl::Metrics * metrics),
              (const, override));
  MOCK_METHOD(void,
              SetCaptureOutputUsage,
              (bool capture_output_used),
              (override));
};

}  // namespace test
}  // namespace webrtc

#endif  // TESTS_TEST_UTILS_MOCK_MOCK_ECHO_REMOVER_H_
