/*
 *  Copyright (c) 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "tests/test_utils/audio_buffer_tools.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "rtc_base/checks.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace webrtc {
namespace test {

void FillBufferWithSine(AudioBuffer* buffer,
                        size_t channel,
                        float frequency_hz,
                        float amplitude) {
  const size_t num_frames = buffer->num_frames();
  const int sample_rate = buffer->num_frames() * 100;  // 10ms frames
  float* data = buffer->channels()[channel];

  for (size_t i = 0; i < num_frames; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    data[i] = amplitude * std::sin(2.0f * static_cast<float>(M_PI) * frequency_hz * t);
  }
}

void FillBufferWithNoise(AudioBuffer* buffer,
                         size_t channel,
                         float amplitude) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-amplitude, amplitude);

  const size_t num_frames = buffer->num_frames();
  float* data = buffer->channels()[channel];

  for (size_t i = 0; i < num_frames; ++i) {
    data[i] = dis(gen);
  }
}

void FillBufferWithSilence(AudioBuffer* buffer, size_t channel) {
  const size_t num_frames = buffer->num_frames();
  float* data = buffer->channels()[channel];

  for (size_t i = 0; i < num_frames; ++i) {
    data[i] = 0.0f;
  }
}

void FillBufferWithSilence(AudioBuffer* buffer) {
  for (size_t ch = 0; ch < buffer->num_channels(); ++ch) {
    FillBufferWithSilence(buffer, ch);
  }
}

float ComputeRms(const AudioBuffer& buffer, size_t channel) {
  const size_t num_frames = buffer.num_frames();
  const float* data = buffer.channels_const()[channel];

  float sum_sq = 0.0f;
  for (size_t i = 0; i < num_frames; ++i) {
    sum_sq += data[i] * data[i];
  }

  return std::sqrt(sum_sq / num_frames);
}

bool BuffersApproximatelyEqual(const AudioBuffer& a,
                               const AudioBuffer& b,
                               float tolerance) {
  if (a.num_channels() != b.num_channels() ||
      a.num_frames() != b.num_frames()) {
    return false;
  }

  for (size_t ch = 0; ch < a.num_channels(); ++ch) {
    const float* data_a = a.channels_const()[ch];
    const float* data_b = b.channels_const()[ch];
    for (size_t i = 0; i < a.num_frames(); ++i) {
      if (std::abs(data_a[i] - data_b[i]) > tolerance) {
        return false;
      }
    }
  }

  return true;
}

namespace {

// Helper to set up a frame with stacked channel pointers
void SetupFrame(const StreamConfig& stream_config,
                std::vector<float*>* frame,
                std::vector<float>* frame_samples) {
  frame_samples->resize(stream_config.num_channels() *
                        stream_config.num_frames());
  frame->resize(stream_config.num_channels());
  for (size_t ch = 0; ch < stream_config.num_channels(); ++ch) {
    (*frame)[ch] = &(*frame_samples)[ch * stream_config.num_frames()];
  }
}

}  // namespace

void CopyVectorToAudioBuffer(const StreamConfig& stream_config,
                             const std::vector<float>& source,
                             AudioBuffer* destination) {
  std::vector<float*> input;
  std::vector<float> input_samples;

  SetupFrame(stream_config, &input, &input_samples);

  RTC_CHECK_EQ(input_samples.size(), source.size());
  memcpy(input_samples.data(), source.data(),
         source.size() * sizeof(source[0]));

  destination->CopyFrom(&input[0], stream_config);
}

void ExtractVectorFromAudioBuffer(const StreamConfig& stream_config,
                                  AudioBuffer* source,
                                  std::vector<float>* destination) {
  std::vector<float*> output;

  SetupFrame(stream_config, &output, destination);

  source->CopyTo(stream_config, &output[0]);
}

}  // namespace test
}  // namespace webrtc
