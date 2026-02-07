// C++ wrapper that implements the AudioProcessing interface by delegating
// to the Rust C API (wap_* functions). Used for validating the Rust backend
// against existing C++ test suites.

#ifndef WEBRTC_MODULES_AUDIO_PROCESSING_RUST_AUDIO_PROCESSING_H_
#define WEBRTC_MODULES_AUDIO_PROCESSING_RUST_AUDIO_PROCESSING_H_

#include "webrtc/api/audio/audio_processing.h"
#include "webrtc/api/ref_counted_base.h"

// Forward declaration of the opaque Rust handle.
struct WapAudioProcessing;
struct WapConfig;
struct WapStreamConfig;

namespace webrtc {

// Implements the AudioProcessing abstract interface by delegating all calls
// to the Rust C API. Unsupported features (AGC1, AECM, AecDump) return
// kUnsupportedFunctionError or no-op.
class RustAudioProcessing : public AudioProcessing,
                            public RefCountedBase {
 public:
  RustAudioProcessing();
  explicit RustAudioProcessing(const AudioProcessing::Config& config);
  ~RustAudioProcessing() override;

  // RefCountInterface (via RefCountedBase).
  void AddRef() const override { RefCountedBase::AddRef(); }
  RefCountReleaseStatus Release() const override {
    return RefCountedBase::Release();
  }

  // --- Initialization ---
  int Initialize() override;
  int Initialize(const ProcessingConfig& processing_config) override;
  void ApplyConfig(const Config& config) override;

  // --- Stream properties ---
  int proc_sample_rate_hz() const override;
  int proc_split_sample_rate_hz() const override;
  size_t num_input_channels() const override;
  size_t num_proc_channels() const override;
  size_t num_output_channels() const override;
  size_t num_reverse_channels() const override;

  // --- Output control ---
  void set_output_will_be_muted(bool muted) override;

  // --- Runtime settings ---
  void SetRuntimeSetting(RuntimeSetting setting) override;
  bool PostRuntimeSetting(RuntimeSetting setting) override;

  // --- Audio processing ---
  int ProcessStream(const int16_t* const src,
                    const StreamConfig& input_config,
                    const StreamConfig& output_config,
                    int16_t* const dest) override;
  int ProcessStream(const float* const* src,
                    const StreamConfig& input_config,
                    const StreamConfig& output_config,
                    float* const* dest) override;
  int ProcessReverseStream(const int16_t* const src,
                           const StreamConfig& input_config,
                           const StreamConfig& output_config,
                           int16_t* const dest) override;
  int ProcessReverseStream(const float* const* src,
                           const StreamConfig& input_config,
                           const StreamConfig& output_config,
                           float* const* dest) override;
  int AnalyzeReverseStream(const float* const* data,
                           const StreamConfig& reverse_config) override;

  // --- Linear AEC output ---
  bool GetLinearAecOutput(
      ArrayView<std::array<float, 160>> linear_output) const override;

  // --- Analog level (AGC) ---
  void set_stream_analog_level(int level) override;
  int recommended_stream_analog_level() const override;

  // --- Stream delay ---
  int set_stream_delay_ms(int delay) override;
  int stream_delay_ms() const override;

  // --- Key press ---
  void set_stream_key_pressed(bool key_pressed) override;

  // --- AEC dump (not supported) ---
  bool CreateAndAttachAecDump(absl::string_view file_name,
                              int64_t max_log_size_bytes,
                              TaskQueueBase* worker_queue) override;
  bool CreateAndAttachAecDump(FILE* handle,
                              int64_t max_log_size_bytes,
                              TaskQueueBase* worker_queue) override;
  void AttachAecDump(std::unique_ptr<AecDump> aec_dump) override;
  void DetachAecDump() override;

  // --- Statistics ---
  AudioProcessingStats GetStatistics() override;
  AudioProcessingStats GetStatistics(bool has_remote_tracks) override;
  AudioProcessing::Config GetConfig() const override;

 private:
  // Convert between C++ and C API types.
  static WapConfig ConfigToWap(const AudioProcessing::Config& config);
  static AudioProcessing::Config WapToConfig(const WapConfig& wap);

  WapAudioProcessing* handle_;  // Owned; destroyed in destructor.

  // Cached stream properties (updated on each ProcessStream call).
  mutable int proc_sample_rate_ = 16000;
  mutable size_t num_input_channels_ = 1;
  mutable size_t num_output_channels_ = 1;
  mutable size_t num_reverse_channels_ = 1;
};

}  // namespace webrtc

#endif  // WEBRTC_MODULES_AUDIO_PROCESSING_RUST_AUDIO_PROCESSING_H_
