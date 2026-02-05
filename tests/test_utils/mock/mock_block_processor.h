/*
 *  Copyright (c) 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef TESTS_TEST_UTILS_MOCK_MOCK_BLOCK_PROCESSOR_H_
#define TESTS_TEST_UTILS_MOCK_MOCK_BLOCK_PROCESSOR_H_

#include <vector>

#include "webrtc/modules/audio_processing/aec3/block_processor.h"
#include <gmock/gmock.h>

namespace webrtc {
namespace test {

class MockBlockProcessor : public BlockProcessor {
 public:
  MockBlockProcessor();
  virtual ~MockBlockProcessor();

  MOCK_METHOD(void,
              ProcessCapture,
              (bool level_change,
               bool saturated_microphone_signal,
               Block* linear_output,
               Block* capture_block),
              (override));
  MOCK_METHOD(void, BufferRender, (const Block& block), (override));
  MOCK_METHOD(void,
              UpdateEchoLeakageStatus,
              (bool leakage_detected),
              (override));
  MOCK_METHOD(void,
              GetMetrics,
              (EchoControl::Metrics * metrics),
              (const, override));
  MOCK_METHOD(void, SetAudioBufferDelay, (int delay_ms), (override));
  MOCK_METHOD(void,
              SetCaptureOutputUsage,
              (bool capture_output_used),
              (override));
};

}  // namespace test
}  // namespace webrtc

#endif  // TESTS_TEST_UTILS_MOCK_MOCK_BLOCK_PROCESSOR_H_
