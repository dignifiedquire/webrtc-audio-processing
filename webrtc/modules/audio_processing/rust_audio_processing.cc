// C++ wrapper implementation delegating to the Rust C API (wap_* functions).

#include "modules/audio_processing/rust_audio_processing.h"

#include <cstring>

#include "wap_audio_processing.h"

namespace webrtc {

namespace {

WapStreamConfig ToWapStreamConfig(const StreamConfig& sc) {
  WapStreamConfig wsc;
  wsc.sample_rate_hz = sc.sample_rate_hz();
  wsc.num_channels = static_cast<int32_t>(sc.num_channels());
  return wsc;
}

// Map WapError to C++ AudioProcessing error codes.
int WapErrorToInt(WapError err) {
  switch (err) {
    case WapError::None:
      return AudioProcessing::kNoError;
    case WapError::NullPointer:
      return AudioProcessing::kNullPointerError;
    case WapError::Internal:
      return AudioProcessing::kUnspecifiedError;
    case WapError::BadSampleRate:
      return AudioProcessing::kBadSampleRateError;
    case WapError::BadNumberChannels:
      return AudioProcessing::kBadNumberChannelsError;
    case WapError::BadStreamParameter:
      return AudioProcessing::kBadStreamParameterWarning;
    case WapError::BadDataLength:
      return AudioProcessing::kBadDataLengthError;
    default:
      return AudioProcessing::kUnspecifiedError;
  }
}

}  // namespace

// --- Construction / destruction ---

RustAudioProcessing::RustAudioProcessing() : handle_(wap_create()) {}

RustAudioProcessing::RustAudioProcessing(
    const AudioProcessing::Config& config) {
  WapConfig wc = ConfigToWap(config);
  handle_ = wap_create_with_config(wc);
}

RustAudioProcessing::~RustAudioProcessing() {
  wap_destroy(handle_);
  handle_ = nullptr;
}

// --- Initialization ---

int RustAudioProcessing::Initialize() {
  // Rust auto-initializes; nothing to do.
  return kNoError;
}

int RustAudioProcessing::Initialize(
    const ProcessingConfig& /*processing_config*/) {
  // Rust auto-reinitializes on format changes.
  return kNoError;
}

void RustAudioProcessing::ApplyConfig(const Config& config) {
  WapConfig wc = ConfigToWap(config);
  wap_apply_config(handle_, wc);
}

// --- Stream properties ---

int RustAudioProcessing::proc_sample_rate_hz() const {
  return proc_sample_rate_;
}

int RustAudioProcessing::proc_split_sample_rate_hz() const {
  // Split band processing rate: <= 16kHz uses the full rate,
  // > 16kHz uses 16kHz per band.
  return proc_sample_rate_ <= 16000 ? proc_sample_rate_ : 16000;
}

size_t RustAudioProcessing::num_input_channels() const {
  return num_input_channels_;
}

size_t RustAudioProcessing::num_proc_channels() const {
  return num_output_channels_;
}

size_t RustAudioProcessing::num_output_channels() const {
  return num_output_channels_;
}

size_t RustAudioProcessing::num_reverse_channels() const {
  return num_reverse_channels_;
}

// --- Output control ---

void RustAudioProcessing::set_output_will_be_muted(bool muted) {
  wap_set_capture_output_used(handle_, !muted);
}

// --- Runtime settings ---

void RustAudioProcessing::SetRuntimeSetting(RuntimeSetting setting) {
  switch (setting.type()) {
    case RuntimeSetting::Type::kCapturePreGain: {
      float value;
      setting.GetFloat(&value);
      wap_set_capture_pre_gain(handle_, value);
      break;
    }
    case RuntimeSetting::Type::kCapturePostGain: {
      float value;
      setting.GetFloat(&value);
      wap_set_capture_post_gain(handle_, value);
      break;
    }
    case RuntimeSetting::Type::kCaptureFixedPostGain: {
      float value;
      setting.GetFloat(&value);
      wap_set_capture_fixed_post_gain(handle_, value);
      break;
    }
    case RuntimeSetting::Type::kPlayoutVolumeChange: {
      int value;
      setting.GetInt(&value);
      wap_set_playout_volume(handle_, value);
      break;
    }
    case RuntimeSetting::Type::kPlayoutAudioDeviceChange: {
      RuntimeSetting::PlayoutAudioDeviceInfo info;
      setting.GetPlayoutAudioDeviceInfo(&info);
      wap_set_playout_audio_device(handle_, info.id, info.max_volume);
      break;
    }
    case RuntimeSetting::Type::kCaptureOutputUsed: {
      bool value;
      setting.GetBool(&value);
      wap_set_capture_output_used(handle_, value);
      break;
    }
    default:
      // kCaptureCompressionGain, kCustomRenderProcessingRuntimeSetting:
      // Not supported by the Rust backend.
      break;
  }
}

bool RustAudioProcessing::PostRuntimeSetting(RuntimeSetting setting) {
  SetRuntimeSetting(std::move(setting));
  return true;
}

// --- Audio processing ---

int RustAudioProcessing::ProcessStream(const int16_t* const src,
                                       const StreamConfig& input_config,
                                       const StreamConfig& output_config,
                                       int16_t* const dest) {
  proc_sample_rate_ = input_config.sample_rate_hz();
  num_input_channels_ = input_config.num_channels();
  num_output_channels_ = output_config.num_channels();

  int32_t src_len =
      static_cast<int32_t>(input_config.num_frames() *
                           input_config.num_channels());
  int32_t dest_len =
      static_cast<int32_t>(output_config.num_frames() *
                           output_config.num_channels());

  WapError err = wap_process_stream_i16(
      handle_, src, src_len, ToWapStreamConfig(input_config),
      ToWapStreamConfig(output_config), dest, dest_len);
  return WapErrorToInt(err);
}

int RustAudioProcessing::ProcessStream(const float* const* src,
                                       const StreamConfig& input_config,
                                       const StreamConfig& output_config,
                                       float* const* dest) {
  proc_sample_rate_ = input_config.sample_rate_hz();
  num_input_channels_ = input_config.num_channels();
  num_output_channels_ = output_config.num_channels();

  WapError err = wap_process_stream_f32(
      handle_, src, ToWapStreamConfig(input_config),
      ToWapStreamConfig(output_config), dest);
  return WapErrorToInt(err);
}

int RustAudioProcessing::ProcessReverseStream(
    const int16_t* const src,
    const StreamConfig& input_config,
    const StreamConfig& output_config,
    int16_t* const dest) {
  num_reverse_channels_ = input_config.num_channels();

  int32_t src_len =
      static_cast<int32_t>(input_config.num_frames() *
                           input_config.num_channels());
  int32_t dest_len =
      static_cast<int32_t>(output_config.num_frames() *
                           output_config.num_channels());

  WapError err = wap_process_reverse_stream_i16(
      handle_, src, src_len, ToWapStreamConfig(input_config),
      ToWapStreamConfig(output_config), dest, dest_len);
  return WapErrorToInt(err);
}

int RustAudioProcessing::ProcessReverseStream(
    const float* const* src,
    const StreamConfig& input_config,
    const StreamConfig& output_config,
    float* const* dest) {
  num_reverse_channels_ = input_config.num_channels();

  WapError err = wap_process_reverse_stream_f32(
      handle_, src, ToWapStreamConfig(input_config),
      ToWapStreamConfig(output_config), dest);
  return WapErrorToInt(err);
}

int RustAudioProcessing::AnalyzeReverseStream(
    const float* const* data,
    const StreamConfig& reverse_config) {
  // Analyze is equivalent to ProcessReverseStream with the same output config
  // and a throwaway output buffer.
  size_t num_channels = reverse_config.num_channels();
  size_t num_frames = reverse_config.num_frames();
  std::vector<std::vector<float>> dest_bufs(num_channels,
                                            std::vector<float>(num_frames));
  std::vector<float*> dest_ptrs(num_channels);
  for (size_t i = 0; i < num_channels; ++i) {
    dest_ptrs[i] = dest_bufs[i].data();
  }

  return ProcessReverseStream(data, reverse_config, reverse_config,
                              dest_ptrs.data());
}

// --- Linear AEC output ---

bool RustAudioProcessing::GetLinearAecOutput(
    ArrayView<std::array<float, 160>> /*linear_output*/) const {
  // Not exposed in the Rust C API.
  return false;
}

// --- Analog level ---

void RustAudioProcessing::set_stream_analog_level(int level) {
  wap_set_stream_analog_level(handle_, level);
}

int RustAudioProcessing::recommended_stream_analog_level() const {
  return wap_recommended_stream_analog_level(handle_);
}

// --- Stream delay ---

int RustAudioProcessing::set_stream_delay_ms(int delay) {
  WapError err = wap_set_stream_delay_ms(handle_, delay);
  return WapErrorToInt(err);
}

int RustAudioProcessing::stream_delay_ms() const {
  return wap_stream_delay_ms(handle_);
}

// --- Key press ---

void RustAudioProcessing::set_stream_key_pressed(bool /*key_pressed*/) {
  // Not used by the modern pipeline.
}

// --- AEC dump (not supported) ---

bool RustAudioProcessing::CreateAndAttachAecDump(
    absl::string_view /*file_name*/,
    int64_t /*max_log_size_bytes*/,
    TaskQueueBase* /*worker_queue*/) {
  return false;
}

bool RustAudioProcessing::CreateAndAttachAecDump(
    FILE* /*handle*/,
    int64_t /*max_log_size_bytes*/,
    TaskQueueBase* /*worker_queue*/) {
  return false;
}

void RustAudioProcessing::AttachAecDump(
    std::unique_ptr<AecDump> /*aec_dump*/) {}

void RustAudioProcessing::DetachAecDump() {}

// --- Statistics ---

AudioProcessingStats RustAudioProcessing::GetStatistics() {
  WapStats wap_stats;
  wap_get_statistics(handle_, &wap_stats);

  AudioProcessingStats stats;
  if (wap_stats.has_echo_return_loss)
    stats.echo_return_loss = wap_stats.echo_return_loss;
  if (wap_stats.has_echo_return_loss_enhancement)
    stats.echo_return_loss_enhancement =
        wap_stats.echo_return_loss_enhancement;
  if (wap_stats.has_divergent_filter_fraction)
    stats.divergent_filter_fraction = wap_stats.divergent_filter_fraction;
  if (wap_stats.has_delay_median_ms)
    stats.delay_median_ms = wap_stats.delay_median_ms;
  if (wap_stats.has_delay_standard_deviation_ms)
    stats.delay_standard_deviation_ms =
        wap_stats.delay_standard_deviation_ms;
  if (wap_stats.has_residual_echo_likelihood)
    stats.residual_echo_likelihood = wap_stats.residual_echo_likelihood;
  if (wap_stats.has_residual_echo_likelihood_recent_max)
    stats.residual_echo_likelihood_recent_max =
        wap_stats.residual_echo_likelihood_recent_max;
  if (wap_stats.has_delay_ms)
    stats.delay_ms = wap_stats.delay_ms;
  return stats;
}

AudioProcessingStats RustAudioProcessing::GetStatistics(
    bool /*has_remote_tracks*/) {
  return GetStatistics();
}

AudioProcessing::Config RustAudioProcessing::GetConfig() const {
  WapConfig wc;
  wap_get_config(handle_, &wc);
  return WapToConfig(wc);
}

// --- Config conversion ---

WapConfig RustAudioProcessing::ConfigToWap(
    const AudioProcessing::Config& config) {
  WapConfig wc = wap_config_default();

  // Pipeline.
  wc.pipeline_maximum_internal_processing_rate =
      config.pipeline.maximum_internal_processing_rate;
  wc.pipeline_multi_channel_render = config.pipeline.multi_channel_render;
  wc.pipeline_multi_channel_capture = config.pipeline.multi_channel_capture;
  wc.pipeline_capture_downmix_method =
      config.pipeline.capture_downmix_method ==
              AudioProcessing::Config::Pipeline::DownmixMethod::
                  kUseFirstChannel
          ? WapDownmixMethod::UseFirstChannel
          : WapDownmixMethod::AverageChannels;

  // Pre-amplifier.
  wc.pre_amplifier_enabled = config.pre_amplifier.enabled;
  wc.pre_amplifier_fixed_gain_factor =
      config.pre_amplifier.fixed_gain_factor;

  // Capture level adjustment.
  wc.capture_level_adjustment_enabled =
      config.capture_level_adjustment.enabled;
  wc.capture_level_adjustment_pre_gain_factor =
      config.capture_level_adjustment.pre_gain_factor;
  wc.capture_level_adjustment_post_gain_factor =
      config.capture_level_adjustment.post_gain_factor;
  wc.analog_mic_gain_emulation_enabled =
      config.capture_level_adjustment.analog_mic_gain_emulation.enabled;
  wc.analog_mic_gain_emulation_initial_level =
      config.capture_level_adjustment.analog_mic_gain_emulation.initial_level;

  // High-pass filter.
  wc.high_pass_filter_enabled = config.high_pass_filter.enabled;
  wc.high_pass_filter_apply_in_full_band =
      config.high_pass_filter.apply_in_full_band;

  // Echo canceller.
  wc.echo_canceller_enabled = config.echo_canceller.enabled;
  wc.echo_canceller_enforce_high_pass_filtering =
      config.echo_canceller.enforce_high_pass_filtering;

  // Noise suppression.
  wc.noise_suppression_enabled = config.noise_suppression.enabled;
  switch (config.noise_suppression.level) {
    case AudioProcessing::Config::NoiseSuppression::kLow:
      wc.noise_suppression_level = WapNoiseSuppressionLevel::Low;
      break;
    case AudioProcessing::Config::NoiseSuppression::kModerate:
      wc.noise_suppression_level = WapNoiseSuppressionLevel::Moderate;
      break;
    case AudioProcessing::Config::NoiseSuppression::kHigh:
      wc.noise_suppression_level = WapNoiseSuppressionLevel::High;
      break;
    case AudioProcessing::Config::NoiseSuppression::kVeryHigh:
      wc.noise_suppression_level = WapNoiseSuppressionLevel::VeryHigh;
      break;
  }
  wc.noise_suppression_analyze_linear_aec_output_when_available =
      config.noise_suppression.analyze_linear_aec_output_when_available;

  // Gain controller 2.
  wc.gain_controller2_enabled = config.gain_controller2.enabled;
  wc.gain_controller2_fixed_digital_gain_db =
      config.gain_controller2.fixed_digital.gain_db;
  wc.gain_controller2_adaptive_digital_enabled =
      config.gain_controller2.adaptive_digital.enabled;
  wc.gain_controller2_adaptive_digital_headroom_db =
      config.gain_controller2.adaptive_digital.headroom_db;
  wc.gain_controller2_adaptive_digital_max_gain_db =
      config.gain_controller2.adaptive_digital.max_gain_db;
  wc.gain_controller2_adaptive_digital_initial_gain_db =
      config.gain_controller2.adaptive_digital.initial_gain_db;
  wc.gain_controller2_adaptive_digital_max_gain_change_db_per_second =
      config.gain_controller2.adaptive_digital.max_gain_change_db_per_second;
  wc.gain_controller2_adaptive_digital_max_output_noise_level_dbfs =
      config.gain_controller2.adaptive_digital.max_output_noise_level_dbfs;
  wc.gain_controller2_input_volume_controller_enabled =
      config.gain_controller2.input_volume_controller.enabled;

  return wc;
}

AudioProcessing::Config RustAudioProcessing::WapToConfig(
    const WapConfig& wc) {
  AudioProcessing::Config config;

  // Pipeline.
  config.pipeline.maximum_internal_processing_rate =
      wc.pipeline_maximum_internal_processing_rate;
  config.pipeline.multi_channel_render = wc.pipeline_multi_channel_render;
  config.pipeline.multi_channel_capture = wc.pipeline_multi_channel_capture;
  config.pipeline.capture_downmix_method =
      wc.pipeline_capture_downmix_method ==
              WapDownmixMethod::UseFirstChannel
          ? AudioProcessing::Config::Pipeline::DownmixMethod::
                kUseFirstChannel
          : AudioProcessing::Config::Pipeline::DownmixMethod::
                kAverageChannels;

  // Pre-amplifier.
  config.pre_amplifier.enabled = wc.pre_amplifier_enabled;
  config.pre_amplifier.fixed_gain_factor =
      wc.pre_amplifier_fixed_gain_factor;

  // Capture level adjustment.
  config.capture_level_adjustment.enabled =
      wc.capture_level_adjustment_enabled;
  config.capture_level_adjustment.pre_gain_factor =
      wc.capture_level_adjustment_pre_gain_factor;
  config.capture_level_adjustment.post_gain_factor =
      wc.capture_level_adjustment_post_gain_factor;
  config.capture_level_adjustment.analog_mic_gain_emulation.enabled =
      wc.analog_mic_gain_emulation_enabled;
  config.capture_level_adjustment.analog_mic_gain_emulation.initial_level =
      wc.analog_mic_gain_emulation_initial_level;

  // High-pass filter.
  config.high_pass_filter.enabled = wc.high_pass_filter_enabled;
  config.high_pass_filter.apply_in_full_band =
      wc.high_pass_filter_apply_in_full_band;

  // Echo canceller.
  config.echo_canceller.enabled = wc.echo_canceller_enabled;
  config.echo_canceller.enforce_high_pass_filtering =
      wc.echo_canceller_enforce_high_pass_filtering;

  // Noise suppression.
  config.noise_suppression.enabled = wc.noise_suppression_enabled;
  switch (wc.noise_suppression_level) {
    case WapNoiseSuppressionLevel::Low:
      config.noise_suppression.level =
          AudioProcessing::Config::NoiseSuppression::kLow;
      break;
    case WapNoiseSuppressionLevel::Moderate:
      config.noise_suppression.level =
          AudioProcessing::Config::NoiseSuppression::kModerate;
      break;
    case WapNoiseSuppressionLevel::High:
      config.noise_suppression.level =
          AudioProcessing::Config::NoiseSuppression::kHigh;
      break;
    case WapNoiseSuppressionLevel::VeryHigh:
      config.noise_suppression.level =
          AudioProcessing::Config::NoiseSuppression::kVeryHigh;
      break;
  }
  config.noise_suppression.analyze_linear_aec_output_when_available =
      wc.noise_suppression_analyze_linear_aec_output_when_available;

  // Gain controller 2.
  config.gain_controller2.enabled = wc.gain_controller2_enabled;
  config.gain_controller2.fixed_digital.gain_db =
      wc.gain_controller2_fixed_digital_gain_db;
  config.gain_controller2.adaptive_digital.enabled =
      wc.gain_controller2_adaptive_digital_enabled;
  config.gain_controller2.adaptive_digital.headroom_db =
      wc.gain_controller2_adaptive_digital_headroom_db;
  config.gain_controller2.adaptive_digital.max_gain_db =
      wc.gain_controller2_adaptive_digital_max_gain_db;
  config.gain_controller2.adaptive_digital.initial_gain_db =
      wc.gain_controller2_adaptive_digital_initial_gain_db;
  config.gain_controller2.adaptive_digital.max_gain_change_db_per_second =
      wc.gain_controller2_adaptive_digital_max_gain_change_db_per_second;
  config.gain_controller2.adaptive_digital.max_output_noise_level_dbfs =
      wc.gain_controller2_adaptive_digital_max_output_noise_level_dbfs;
  config.gain_controller2.input_volume_controller.enabled =
      wc.gain_controller2_input_volume_controller_enabled;

  return config;
}

}  // namespace webrtc
