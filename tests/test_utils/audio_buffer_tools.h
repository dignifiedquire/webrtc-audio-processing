/*
 *  Copyright (c) 2024 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#ifndef TESTS_TEST_UTILS_AUDIO_BUFFER_TOOLS_H_
#define TESTS_TEST_UTILS_AUDIO_BUFFER_TOOLS_H_

#include <cmath>
#include <memory>
#include <vector>

#include "api/audio/audio_processing.h"
#include "modules/audio_processing/audio_buffer.h"

namespace webrtc {
namespace test {

// Fill an AudioBuffer channel with a sine wave
void FillBufferWithSine(AudioBuffer* buffer,
                        size_t channel,
                        float frequency_hz,
                        float amplitude);

// Fill an AudioBuffer channel with white noise
void FillBufferWithNoise(AudioBuffer* buffer,
                         size_t channel,
                         float amplitude);

// Fill an AudioBuffer channel with silence (zeros)
void FillBufferWithSilence(AudioBuffer* buffer, size_t channel);

// Fill all channels with silence
void FillBufferWithSilence(AudioBuffer* buffer);

// Compute the RMS of a buffer channel
float ComputeRms(const AudioBuffer& buffer, size_t channel);

// Check if two buffers are approximately equal
bool BuffersApproximatelyEqual(const AudioBuffer& a,
                               const AudioBuffer& b,
                               float tolerance);

// Copy interleaved float vector data into an AudioBuffer.
// The input vector should be interleaved: [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]
void CopyVectorToAudioBuffer(const StreamConfig& stream_config,
                             const std::vector<float>& input,
                             AudioBuffer* audio_buffer);

// Extract data from an AudioBuffer into an interleaved float vector.
void ExtractVectorFromAudioBuffer(const StreamConfig& stream_config,
                                  AudioBuffer* audio_buffer,
                                  std::vector<float>* output);

}  // namespace test
}  // namespace webrtc

#endif  // TESTS_TEST_UTILS_AUDIO_BUFFER_TOOLS_H_
