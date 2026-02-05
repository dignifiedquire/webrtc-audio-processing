/*
 *  Copyright (c) 2016 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "modules/audio_processing/audio_buffer.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "api/audio/audio_processing.h"
#include "api/audio/audio_view.h"
#include "rtc_base/checks.h"

#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace webrtc {

namespace {

void ExpectNumChannels(const AudioBuffer& ab, size_t num_channels) {
  EXPECT_EQ(ab.num_channels(), num_channels);
}

void FillChannelWith100HzSine(int channel, float amplitude, AudioBuffer& ab) {
  float sample_rate_hz;
  if (ab.num_frames() == 160) {
    sample_rate_hz = 16000.0f;
  } else if (ab.num_frames() == 320) {
    sample_rate_hz = 32000.0f;
  } else {
    sample_rate_hz = 48000.0f;
  }

  constexpr float kFrequencyHz = 100.0f;
  for (size_t i = 0; i < ab.num_frames(); ++i) {
    ab.channels()[channel][i] =
        amplitude * std::sin(2.0f * static_cast<float>(M_PI) * kFrequencyHz / sample_rate_hz * i);
  }
}

void FillChannelWith100HzSine(int sample_rate_hz,
                              int channel,
                              float amplitude,
                              float* const* stacked_data) {
  int num_samples_per_channel;
  if (sample_rate_hz == 16000) {
    num_samples_per_channel = 160;
  } else if (sample_rate_hz == 32000) {
    num_samples_per_channel = 320;
  } else {
    num_samples_per_channel = 480;
  }

  constexpr float kFrequencyHz = 100.0f;
  for (int i = 0; i < num_samples_per_channel; ++i) {
    stacked_data[channel][i] =
        amplitude * std::sin(2.0f * static_cast<float>(M_PI) * kFrequencyHz / sample_rate_hz * i);
  }
}

void FillChannelWith100HzSine(int sample_rate_hz,
                              int num_channels,
                              int channel,
                              float amplitude,
                              int16_t* const interleaved_data) {
  int num_samples_per_channel;
  if (sample_rate_hz == 16000) {
    num_samples_per_channel = 160;
  } else if (sample_rate_hz == 32000) {
    num_samples_per_channel = 320;
  } else {
    num_samples_per_channel = 480;
  }

  constexpr float kFrequencyHz = 100.0f;
  for (int i = 0; i < num_samples_per_channel; ++i) {
    interleaved_data[channel + i * num_channels] = static_cast<int16_t>(
        amplitude * std::sin(2.0f * static_cast<float>(M_PI) * kFrequencyHz / sample_rate_hz * i));
  }
}

}  // namespace

// Parametrized test for channel count and two sample rates
class AudioBufferChannelCountAndTwoRatesTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<int, int, int>> {
 protected:
  int Rate1() const { return std::get<0>(GetParam()); }
  int Rate2() const { return std::get<1>(GetParam()); }
  int NumChannels() const { return std::get<2>(GetParam()); }
};

INSTANTIATE_TEST_SUITE_P(
    AudioBufferTests,
    AudioBufferChannelCountAndTwoRatesTest,
    ::testing::Combine(::testing::Values(16000, 32000, 48000),
                       ::testing::Values(16000, 32000, 48000),
                       ::testing::Values(1, 2)));

// Parametrized test for channel count and one sample rate
class AudioBufferChannelCountAndOneRateTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<int, int>> {
 protected:
  int Rate() const { return std::get<0>(GetParam()); }
  int NumChannels() const { return std::get<1>(GetParam()); }
};

INSTANTIATE_TEST_SUITE_P(
    AudioBufferTests,
    AudioBufferChannelCountAndOneRateTest,
    ::testing::Combine(::testing::Values(16000, 32000, 48000),
                       ::testing::Values(1, 2)));

TEST(AudioBufferTest, SetNumChannelsSetsChannelBuffersNumChannels) {
  constexpr size_t kSampleRateHz = 48000u;
  AudioBuffer ab(kSampleRateHz, 2, kSampleRateHz, 2, kSampleRateHz, 2);
  ExpectNumChannels(ab, 2);
  ab.set_num_channels(1);
  ExpectNumChannels(ab, 1);
}

TEST(AudioBufferTest, ConstructorSetsCorrectDimensions) {
  AudioBuffer ab(16000, 1, 16000, 1, 16000, 1);
  EXPECT_EQ(1u, ab.num_channels());
  EXPECT_EQ(160u, ab.num_frames());  // 10ms at 16kHz
  EXPECT_EQ(1u, ab.num_bands());
}

TEST(AudioBufferTest, MultiChannelConstruction) {
  AudioBuffer ab(48000, 2, 48000, 2, 48000, 2);
  EXPECT_EQ(2u, ab.num_channels());
  EXPECT_EQ(480u, ab.num_frames());  // 10ms at 48kHz
}

TEST(AudioBufferTest, ThreeBandSplittingAt48kHz) {
  AudioBuffer ab(48000, 1, 48000, 1, 48000, 1);
  EXPECT_EQ(3u, ab.num_bands());
  EXPECT_EQ(160u, ab.num_frames_per_band());
}

TEST(AudioBufferTest, TwoBandSplittingAt32kHz) {
  AudioBuffer ab(32000, 1, 32000, 1, 32000, 1);
  EXPECT_EQ(2u, ab.num_bands());
}

TEST(AudioBufferTest, OneBandAt16kHz) {
  AudioBuffer ab(16000, 1, 16000, 1, 16000, 1);
  EXPECT_EQ(1u, ab.num_bands());
}

TEST_P(AudioBufferChannelCountAndOneRateTest, CopyWithoutResampling) {
  AudioBuffer ab1(Rate(), NumChannels(), Rate(), NumChannels(), Rate(),
                  NumChannels());
  AudioBuffer ab2(Rate(), NumChannels(), Rate(), NumChannels(), Rate(),
                  NumChannels());
  // Fill first buffer.
  for (size_t ch = 0; ch < ab1.num_channels(); ++ch) {
    for (size_t i = 0; i < ab1.num_frames(); ++i) {
      ab1.channels()[ch][i] = static_cast<float>(i + ch);
    }
  }
  // Copy to second buffer.
  ab1.CopyTo(&ab2);
  // Verify content of second buffer.
  for (size_t ch = 0; ch < ab2.num_channels(); ++ch) {
    for (size_t i = 0; i < ab2.num_frames(); ++i) {
      EXPECT_EQ(ab2.channels()[ch][i], static_cast<float>(i + ch));
    }
  }
}

TEST_P(AudioBufferChannelCountAndTwoRatesTest, CopyWithResampling) {
  AudioBuffer ab1(Rate1(), NumChannels(), Rate1(), NumChannels(), Rate2(),
                  NumChannels());
  AudioBuffer ab2(Rate2(), NumChannels(), Rate2(), NumChannels(), Rate2(),
                  NumChannels());
  float energy_ab1 = 0.0f;
  float energy_ab2 = 0.0f;
  // Put a sine and compute energy of first buffer.
  for (size_t ch = 0; ch < ab1.num_channels(); ++ch) {
    FillChannelWith100HzSine(static_cast<int>(ch), 1.0f, ab1);

    for (size_t i = 0; i < ab1.num_frames(); ++i) {
      energy_ab1 += ab1.channels()[ch][i] * ab1.channels()[ch][i];
    }
  }
  // Copy to second buffer.
  ab1.CopyTo(&ab2);
  // Compute energy of second buffer.
  for (size_t ch = 0; ch < ab2.num_channels(); ++ch) {
    for (size_t i = 0; i < ab2.num_frames(); ++i) {
      energy_ab2 += ab2.channels()[ch][i] * ab2.channels()[ch][i];
    }
  }
  // Verify that energies match.
  EXPECT_NEAR(energy_ab1, energy_ab2 * Rate1() / Rate2(), .04f * energy_ab1);
}

TEST_P(AudioBufferChannelCountAndOneRateTest, DeinterleavedView) {
  AudioBuffer ab(Rate(), NumChannels(), Rate(), NumChannels(), Rate(),
                 NumChannels());
  // Fill the buffer with data.
  for (size_t ch = 0; ch < ab.num_channels(); ++ch) {
    FillChannelWith100HzSine(static_cast<int>(ch), 1.0f, ab);
  }

  // Verify that the DeinterleavedView correctly maps to channels.
  DeinterleavedView<float> view = ab.view();
  ASSERT_EQ(view.num_channels(), ab.num_channels());
  float* const* channels = ab.channels();
  for (size_t c = 0; c < view.num_channels(); ++c) {
    MonoView<float> channel = view[c];
    EXPECT_EQ(SamplesPerChannel(channel), ab.num_frames());
    for (size_t s = 0; s < SamplesPerChannel(channel); ++s) {
      ASSERT_EQ(channel[s], channels[c][s]);
    }
  }
}

TEST_P(AudioBufferChannelCountAndOneRateTest, CopyFromInterleaved) {
  const int rate = Rate();
  const int num_channels = NumChannels();
  const int num_frames = rate / 100;  // 10ms
  
  AudioBuffer ab(rate, num_channels, rate, num_channels, rate, num_channels);
  
  // Create interleaved input data
  // Use amplitude in int16 range (0.7 * 32767 = ~22936)
  std::vector<int16_t> interleaved_data(num_frames * num_channels);
  for (int ch = 0; ch < num_channels; ++ch) {
    FillChannelWith100HzSine(rate, num_channels, ch, 0.7f * 32767.0f, interleaved_data.data());
  }
  
  // Copy to audio buffer
  StreamConfig stream_config(rate, num_channels);
  ab.CopyFrom(interleaved_data.data(), stream_config);
  
  // Verify that the channel count is correct
  EXPECT_EQ(ab.num_channels(), static_cast<size_t>(num_channels));
  
  // Verify that data was copied (non-zero energy)
  float energy = 0.0f;
  for (size_t ch = 0; ch < ab.num_channels(); ++ch) {
    for (size_t i = 0; i < ab.num_frames(); ++i) {
      energy += ab.channels()[ch][i] * ab.channels()[ch][i];
    }
  }
  EXPECT_GT(energy, 0.0f);
}

TEST_P(AudioBufferChannelCountAndOneRateTest, CopyFromStacked) {
  const int rate = Rate();
  const int num_channels = NumChannels();
  const int num_frames = rate / 100;  // 10ms
  
  AudioBuffer ab(rate, num_channels, rate, num_channels, rate, num_channels);
  
  // Create stacked input data
  std::vector<float> audio_data(num_frames * num_channels);
  std::vector<float*> channel_ptrs(num_channels);
  for (int ch = 0; ch < num_channels; ++ch) {
    channel_ptrs[ch] = &audio_data[ch * num_frames];
  }
  
  for (int ch = 0; ch < num_channels; ++ch) {
    FillChannelWith100HzSine(rate, ch, 0.7f, channel_ptrs.data());
  }
  
  // Copy to audio buffer
  StreamConfig stream_config(rate, num_channels);
  ab.CopyFrom(channel_ptrs.data(), stream_config);
  
  // Verify that the channel count is correct
  EXPECT_EQ(ab.num_channels(), static_cast<size_t>(num_channels));
  
  // Verify that data was copied (non-zero energy)
  float energy = 0.0f;
  for (size_t ch = 0; ch < ab.num_channels(); ++ch) {
    for (size_t i = 0; i < ab.num_frames(); ++i) {
      energy += ab.channels()[ch][i] * ab.channels()[ch][i];
    }
  }
  EXPECT_GT(energy, 0.0f);
}

TEST_P(AudioBufferChannelCountAndOneRateTest, CopyToInterleaved) {
  const int rate = Rate();
  const int num_channels = NumChannels();
  const int num_frames = rate / 100;  // 10ms
  
  AudioBuffer ab(rate, num_channels, rate, num_channels, rate, num_channels);
  
  // Fill the buffer with data
  for (size_t ch = 0; ch < ab.num_channels(); ++ch) {
    FillChannelWith100HzSine(static_cast<int>(ch), 0.7f, ab);
  }
  
  // Copy to interleaved output
  std::vector<int16_t> interleaved_data(num_frames * num_channels, 0);
  StreamConfig stream_config(rate, num_channels);
  ab.CopyTo(stream_config, interleaved_data.data());
  
  // Verify that data was copied (non-zero values)
  int64_t sum = 0;
  for (size_t i = 0; i < interleaved_data.size(); ++i) {
    sum += std::abs(interleaved_data[i]);
  }
  EXPECT_GT(sum, 0);
}

TEST_P(AudioBufferChannelCountAndOneRateTest, CopyToStacked) {
  const int rate = Rate();
  const int num_channels = NumChannels();
  const int num_frames = rate / 100;  // 10ms
  
  AudioBuffer ab(rate, num_channels, rate, num_channels, rate, num_channels);
  
  // Fill the buffer with data
  for (size_t ch = 0; ch < ab.num_channels(); ++ch) {
    FillChannelWith100HzSine(static_cast<int>(ch), 0.7f, ab);
  }
  
  // Copy to stacked output
  std::vector<float> audio_data(num_frames * num_channels, 0.0f);
  std::vector<float*> channel_ptrs(num_channels);
  for (int ch = 0; ch < num_channels; ++ch) {
    channel_ptrs[ch] = &audio_data[ch * num_frames];
  }
  
  StreamConfig stream_config(rate, num_channels);
  ab.CopyTo(stream_config, channel_ptrs.data());
  
  // Verify that data was copied (non-zero energy)
  float energy = 0.0f;
  for (size_t i = 0; i < audio_data.size(); ++i) {
    energy += audio_data[i] * audio_data[i];
  }
  EXPECT_GT(energy, 0.0f);
}

TEST(AudioBufferTest, SplitAndMergeFrequencyBands) {
  // Use 48kHz which has 3 bands
  AudioBuffer ab(48000, 1, 48000, 1, 48000, 1);
  
  // Fill with a test signal
  FillChannelWith100HzSine(0, 1.0f, ab);
  
  // Compute energy before split
  float energy_before = 0.0f;
  for (size_t i = 0; i < ab.num_frames(); ++i) {
    energy_before += ab.channels()[0][i] * ab.channels()[0][i];
  }
  
  // Split into bands
  ab.SplitIntoFrequencyBands();
  
  // Verify we have 3 bands
  EXPECT_EQ(3u, ab.num_bands());
  
  // Merge bands back
  ab.MergeFrequencyBands();
  
  // Compute energy after merge
  float energy_after = 0.0f;
  for (size_t i = 0; i < ab.num_frames(); ++i) {
    energy_after += ab.channels()[0][i] * ab.channels()[0][i];
  }
  
  // Energy should be preserved (within tolerance)
  // Note: The splitting filter introduces some loss, so we use a more relaxed tolerance
  EXPECT_NEAR(energy_before, energy_after, 0.20f * energy_before);
}

TEST(AudioBufferTest, DownmixByAveraging) {
  // Create stereo buffer that downmixes to mono
  AudioBuffer ab(48000, 2, 48000, 1, 48000, 1);
  ab.set_downmixing_by_averaging();
  
  // Create stereo input where ch0 = 1000, ch1 = 3000
  // Average should be 2000
  std::vector<int16_t> stereo_input(480 * 2);
  for (int i = 0; i < 480; ++i) {
    stereo_input[i * 2] = 1000;      // ch0
    stereo_input[i * 2 + 1] = 3000;  // ch1
  }
  
  StreamConfig input_config(48000, 2);
  ab.CopyFrom(stereo_input.data(), input_config);
  
  // Check that we have mono output
  EXPECT_EQ(1u, ab.num_channels());
  
  // The averaged value should be close to 2000
  float avg = 0.0f;
  for (size_t i = 0; i < ab.num_frames(); ++i) {
    avg += ab.channels()[0][i];
  }
  avg /= ab.num_frames();
  EXPECT_NEAR(avg, 2000.0f, 100.0f);
}

TEST(AudioBufferTest, DownmixToSpecificChannel) {
  // Create stereo buffer that downmixes to mono by selecting channel 1
  AudioBuffer ab(48000, 2, 48000, 1, 48000, 1);
  ab.set_downmixing_to_specific_channel(1);
  
  // Create stereo input where ch0 = 1000, ch1 = 3000
  std::vector<int16_t> stereo_input(480 * 2);
  for (int i = 0; i < 480; ++i) {
    stereo_input[i * 2] = 1000;      // ch0
    stereo_input[i * 2 + 1] = 3000;  // ch1
  }
  
  StreamConfig input_config(48000, 2);
  ab.CopyFrom(stereo_input.data(), input_config);
  
  // Check that we have mono output
  EXPECT_EQ(1u, ab.num_channels());
  
  // The value should be close to channel 1's value (3000)
  float avg = 0.0f;
  for (size_t i = 0; i < ab.num_frames(); ++i) {
    avg += ab.channels()[0][i];
  }
  avg /= ab.num_frames();
  EXPECT_NEAR(avg, 3000.0f, 100.0f);
}

}  // namespace webrtc
