/*
 *  Copyright (c) 2014 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "modules/audio_processing/splitting_filter.h"

#include <cmath>
#include <cstddef>
#include <cstring>

#include "common_audio/channel_buffer.h"

#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace webrtc {
namespace {

constexpr size_t kSamplesPer16kHzChannel = 160;
constexpr size_t kSamplesPer48kHzChannel = 480;

}  // namespace

// Generates a signal from presence or absence of sine waves of different
// frequencies.
// Splits into 3 bands and checks their presence or absence.
// Recombines the bands.
// Calculates the delay.
// Checks that the cross correlation of input and output is high enough at the
// calculated delay.
TEST(SplittingFilterTest, SplitsIntoThreeBandsAndReconstructs) {
  static const int kChannels = 1;
  static const int kSampleRateHz = 48000;
  static const size_t kNumBands = 3;
  static const int kFrequenciesHz[kNumBands] = {1000, 12000, 18000};
  static const float kAmplitude = 8192.f;
  static const size_t kChunks = 8;
  
  SplittingFilter splitting_filter(kChannels, kNumBands,
                                   kSamplesPer48kHzChannel);
  ChannelBuffer<float> in_data(kSamplesPer48kHzChannel, kChannels, kNumBands);
  ChannelBuffer<float> bands(kSamplesPer48kHzChannel, kChannels, kNumBands);
  ChannelBuffer<float> out_data(kSamplesPer48kHzChannel, kChannels, kNumBands);
  
  for (size_t i = 0; i < kChunks; ++i) {
    // Input signal generation.
    bool is_present[kNumBands];
    memset(in_data.channels()[0], 0,
           kSamplesPer48kHzChannel * sizeof(in_data.channels()[0][0]));
    for (size_t j = 0; j < kNumBands; ++j) {
      is_present[j] = i & (static_cast<size_t>(1) << j);
      float amplitude = is_present[j] ? kAmplitude : 0.f;
      for (size_t k = 0; k < kSamplesPer48kHzChannel; ++k) {
        in_data.channels()[0][k] +=
            amplitude * std::sin(2.f * static_cast<float>(M_PI) * kFrequenciesHz[j] *
                                (i * kSamplesPer48kHzChannel + k) / kSampleRateHz);
      }
    }
    
    // Three band splitting filter.
    splitting_filter.Analysis(&in_data, &bands);
    
    // Energy calculation.
    float energy[kNumBands];
    for (size_t j = 0; j < kNumBands; ++j) {
      energy[j] = 0.f;
      for (size_t k = 0; k < kSamplesPer16kHzChannel; ++k) {
        energy[j] += bands.channels(j)[0][k] * bands.channels(j)[0][k];
      }
      energy[j] /= kSamplesPer16kHzChannel;
      if (is_present[j]) {
        EXPECT_GT(energy[j], kAmplitude * kAmplitude / 4);
      } else {
        EXPECT_LT(energy[j], kAmplitude * kAmplitude / 4);
      }
    }
    
    // Three band merge.
    splitting_filter.Synthesis(&bands, &out_data);
    
    // Delay and cross correlation estimation.
    float xcorr = 0.f;
    for (size_t delay = 0; delay < kSamplesPer48kHzChannel; ++delay) {
      float tmpcorr = 0.f;
      for (size_t j = delay; j < kSamplesPer48kHzChannel; ++j) {
        tmpcorr += in_data.channels()[0][j - delay] * out_data.channels()[0][j];
      }
      tmpcorr /= kSamplesPer48kHzChannel;
      if (tmpcorr > xcorr) {
        xcorr = tmpcorr;
      }
    }
    
    // High cross correlation check.
    bool any_present = false;
    for (size_t j = 0; j < kNumBands; ++j) {
      any_present |= is_present[j];
    }
    if (any_present) {
      EXPECT_GT(xcorr, kAmplitude * kAmplitude / 4);
    }
  }
}

// Test two-band splitting at 32kHz
TEST(SplittingFilterTest, SplitsIntoTwoBandsAndReconstructs) {
  static const int kChannels = 1;
  static const int kSampleRateHz = 32000;
  static const size_t kNumBands = 2;
  static const size_t kSamplesPerChannel = 320;  // 10ms at 32kHz
  static const int kFrequenciesHz[kNumBands] = {1000, 12000};
  static const float kAmplitude = 8192.f;
  static const size_t kChunks = 8;
  
  SplittingFilter splitting_filter(kChannels, kNumBands, kSamplesPerChannel);
  ChannelBuffer<float> in_data(kSamplesPerChannel, kChannels, kNumBands);
  ChannelBuffer<float> bands(kSamplesPerChannel, kChannels, kNumBands);
  ChannelBuffer<float> out_data(kSamplesPerChannel, kChannels, kNumBands);
  
  for (size_t i = 0; i < kChunks; ++i) {
    // Input signal generation.
    bool is_present[kNumBands];
    memset(in_data.channels()[0], 0,
           kSamplesPerChannel * sizeof(in_data.channels()[0][0]));
    for (size_t j = 0; j < kNumBands; ++j) {
      is_present[j] = i & (static_cast<size_t>(1) << j);
      float amplitude = is_present[j] ? kAmplitude : 0.f;
      for (size_t k = 0; k < kSamplesPerChannel; ++k) {
        in_data.channels()[0][k] +=
            amplitude * std::sin(2.f * static_cast<float>(M_PI) * kFrequenciesHz[j] *
                                (i * kSamplesPerChannel + k) / kSampleRateHz);
      }
    }
    
    // Two band splitting filter.
    splitting_filter.Analysis(&in_data, &bands);
    
    // Two band merge.
    splitting_filter.Synthesis(&bands, &out_data);
    
    // Cross correlation check - verify reconstruction preserves signal
    float xcorr = 0.f;
    for (size_t delay = 0; delay < kSamplesPerChannel; ++delay) {
      float tmpcorr = 0.f;
      for (size_t j = delay; j < kSamplesPerChannel; ++j) {
        tmpcorr += in_data.channels()[0][j - delay] * out_data.channels()[0][j];
      }
      tmpcorr /= kSamplesPerChannel;
      if (tmpcorr > xcorr) {
        xcorr = tmpcorr;
      }
    }
    
    bool any_present = false;
    for (size_t j = 0; j < kNumBands; ++j) {
      any_present |= is_present[j];
    }
    if (any_present) {
      EXPECT_GT(xcorr, kAmplitude * kAmplitude / 4);
    }
  }
}

// Test multi-channel splitting
TEST(SplittingFilterTest, MultiChannelSplitting) {
  static const int kChannels = 2;
  static const size_t kNumBands = 3;
  static const float kAmplitude = 8192.f;
  
  SplittingFilter splitting_filter(kChannels, kNumBands, kSamplesPer48kHzChannel);
  ChannelBuffer<float> in_data(kSamplesPer48kHzChannel, kChannels, kNumBands);
  ChannelBuffer<float> bands(kSamplesPer48kHzChannel, kChannels, kNumBands);
  ChannelBuffer<float> out_data(kSamplesPer48kHzChannel, kChannels, kNumBands);
  
  // Fill both channels with different frequencies
  for (int ch = 0; ch < kChannels; ++ch) {
    float freq = (ch == 0) ? 1000.f : 12000.f;
    for (size_t k = 0; k < kSamplesPer48kHzChannel; ++k) {
      in_data.channels()[ch][k] =
          kAmplitude * std::sin(2.f * static_cast<float>(M_PI) * freq * k / 48000.f);
    }
  }
  
  // Split
  splitting_filter.Analysis(&in_data, &bands);
  
  // Merge
  splitting_filter.Synthesis(&bands, &out_data);
  
  // Verify both channels have non-zero output
  for (int ch = 0; ch < kChannels; ++ch) {
    float energy = 0.f;
    for (size_t k = 0; k < kSamplesPer48kHzChannel; ++k) {
      energy += out_data.channels()[ch][k] * out_data.channels()[ch][k];
    }
    EXPECT_GT(energy, 0.f);
  }
}

}  // namespace webrtc
