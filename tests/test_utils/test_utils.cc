/*
 *  Copyright (c) 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "tests/test_utils/test_utils.h"

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "webrtc/rtc_base/checks.h"
#include "webrtc/rtc_base/system/arch.h"

namespace webrtc {

// ChannelBufferWavReader and ChannelBufferWavWriter are stubbed out
// because wav_file.h is not available in this project.
// These classes are not used by the tests we're running.

ChannelBufferWavReader::ChannelBufferWavReader(std::unique_ptr<WavReader> file) {
  RTC_FATAL() << "ChannelBufferWavReader not available - wav_file.h not in project";
}

ChannelBufferWavReader::~ChannelBufferWavReader() = default;

bool ChannelBufferWavReader::Read(ChannelBuffer<float>* buffer) {
  RTC_FATAL() << "ChannelBufferWavReader not available - wav_file.h not in project";
  return false;
}

ChannelBufferWavWriter::ChannelBufferWavWriter(std::unique_ptr<WavWriter> file) {
  RTC_FATAL() << "ChannelBufferWavWriter not available - wav_file.h not in project";
}

ChannelBufferWavWriter::~ChannelBufferWavWriter() = default;

void ChannelBufferWavWriter::Write(const ChannelBuffer<float>& buffer) {
  RTC_FATAL() << "ChannelBufferWavWriter not available - wav_file.h not in project";
}

ChannelBufferVectorWriter::ChannelBufferVectorWriter(std::vector<float>* output)
    : output_(output) {
  RTC_DCHECK(output_);
}

ChannelBufferVectorWriter::~ChannelBufferVectorWriter() = default;

void ChannelBufferVectorWriter::Write(const ChannelBuffer<float>& buffer) {
  // Account for sample rate changes throughout a simulation.
  interleaved_buffer_.resize(buffer.size());
  InterleavedView<float> view(&interleaved_buffer_[0], buffer.num_frames(),
                              buffer.num_channels());
  Interleave(buffer.channels(), buffer.num_frames(), buffer.num_channels(),
             view);
  size_t old_size = output_->size();
  output_->resize(old_size + interleaved_buffer_.size());
  FloatToFloatS16(interleaved_buffer_.data(), interleaved_buffer_.size(),
                  output_->data() + old_size);
}

FILE* OpenFile(absl::string_view filename, absl::string_view mode) {
  std::string filename_str(filename);
  FILE* file = fopen(filename_str.c_str(), std::string(mode).c_str());
  if (!file) {
    printf("Unable to open file %s\n", filename_str.c_str());
    exit(1);
  }
  return file;
}

void SetFrameSampleRate(Int16FrameData* frame, int sample_rate_hz) {
  frame->sample_rate_hz = sample_rate_hz;
  frame->samples_per_channel =
      AudioProcessing::kChunkSizeMs * sample_rate_hz / 1000;
}

}  // namespace webrtc
