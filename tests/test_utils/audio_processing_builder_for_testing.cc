/*
 *  Copyright (c) 2020 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "tests/test_utils/audio_processing_builder_for_testing.h"

#include <memory>
#include <utility>

#include "api/audio/builtin_audio_processing_builder.h"
#include "api/environment/environment_factory.h"
#include "api/make_ref_counted.h"
#include "webrtc/modules/audio_processing/audio_processing_impl.h"

namespace webrtc {

AudioProcessingBuilderForTesting::AudioProcessingBuilderForTesting() = default;
AudioProcessingBuilderForTesting::~AudioProcessingBuilderForTesting() = default;

scoped_refptr<AudioProcessing> AudioProcessingBuilderForTesting::Create() {
  BuiltinAudioProcessingBuilder builder(config_);
  builder.SetCapturePostProcessing(std::move(capture_post_processing_));
  builder.SetRenderPreProcessing(std::move(render_pre_processing_));
  builder.SetEchoControlFactory(std::move(echo_control_factory_));
  builder.SetEchoDetector(std::move(echo_detector_));
  builder.SetCaptureAnalyzer(std::move(capture_analyzer_));
  return builder.Build(CreateEnvironment());
}

}  // namespace webrtc
