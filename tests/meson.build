# Include directories for tests - include project root for "tests/test_utils/..." includes
tests_inc = include_directories('..')

# Path to test resources directory
test_resources_dir = meson.current_source_dir() / 'resources'

# Protobuf support for audio_processing_unittest
protobuf_dep = dependency('protobuf-lite', required: false)
have_protobuf = protobuf_dep.found()

if have_protobuf
  protoc = find_program('protoc', required: true)

  # Generate protobuf sources
  proto_gen = generator(protoc,
    output: ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
    arguments: [
      '--proto_path=@CURRENT_SOURCE_DIR@/proto',
      '--cpp_out=@BUILD_DIR@',
      '@INPUT@'
    ]
  )

  proto_sources = proto_gen.process(
    'proto/debug.proto',
    'proto/unittest.proto',
  )

  protobuf_utils_sources = files(
    'test_utils/protobuf_utils.cc',
  )
else
  proto_sources = []
  protobuf_utils_sources = []
endif

# Test utilities
test_utils_sources = files(
  'test_utils/audio_buffer_tools.cc',
  'test_utils/bitexactness_tools.cc',
  'test_utils/create_test_field_trials.cc',
  'test_utils/file_utils.cc',
  'test_utils/input_audio_file.cc',
  'test_utils/echo_canceller_test_tools.cc',
  'test_utils/fake_clock.cc',
  'test_utils/field_trial.cc',
  'test_utils/performance_timer.cc',
  'test_utils/rnn_vad_test_utils.cc',
  'test_utils/test_utils.cc',
  'test_utils/audio_processing_builder_for_testing.cc',
)

# Mock classes for testing
mock_sources = files(
  'test_utils/mock/mock_echo_remover.cc',
  'test_utils/mock/mock_render_delay_buffer.cc',
  'test_utils/mock/mock_render_delay_controller.cc',
  'test_utils/mock/mock_block_processor.cc',
)

test_utils_lib_deps = [
  gtest_dep,
  gmock_dep,
  audio_processing_dep,
]
if have_protobuf
  test_utils_lib_deps += protobuf_dep
endif

test_utils_cxxflags = common_cxxflags + ['-DWEBRTC_TEST_RESOURCES_DIR="@0@"'.format(test_resources_dir), '-DWEBRTC_APM_DEBUG_DUMP=0', '-DWEBRTC_AUDIOPROC_FLOAT_PROFILE']
if have_protobuf
  test_utils_cxxflags += '-DWEBRTC_ENABLE_PROTOBUF=1'
endif

test_utils_lib = static_library('test_utils',
  test_utils_sources + mock_sources + protobuf_utils_sources + proto_sources,
  dependencies: test_utils_lib_deps,
  include_directories: [webrtc_inc, tests_inc],
  cpp_args: test_utils_cxxflags,
)

test_utils_dep = declare_dependency(
  link_with: test_utils_lib,
  include_directories: tests_inc,
  sources: proto_sources,  # Includes generated headers path
)

# Additional sources needed for tests (not exported from library due to hidden visibility)
common_audio_test_sources = files(
  '../webrtc/api/audio/echo_detector_creator.cc',
  '../webrtc/rtc_base/cpu_info.cc',
  '../webrtc/rtc_base/system/file_wrapper.cc',
  '../webrtc/system_wrappers/source/cpu_features.cc',
  '../webrtc/common_audio/channel_buffer.cc',
  '../webrtc/common_audio/wav_file.cc',
  '../webrtc/common_audio/wav_header.cc',
  '../webrtc/common_audio/fir_filter_c.cc',
  '../webrtc/common_audio/fir_filter_factory.cc',
  '../webrtc/common_audio/fir_filter_neon.cc',
  '../webrtc/common_audio/resampler/sinusoidal_linear_chirp_source.cc',
  '../webrtc/common_audio/smoothing_filter.cc',
  # Signal processing C sources for signal_processing tests
  '../webrtc/common_audio/signal_processing/auto_correlation.c',
  '../webrtc/common_audio/signal_processing/auto_corr_to_refl_coef.c',
  '../webrtc/common_audio/signal_processing/complex_bit_reverse.c',
  '../webrtc/common_audio/signal_processing/complex_fft.c',
  '../webrtc/common_audio/signal_processing/copy_set_operations.c',
  '../webrtc/common_audio/signal_processing/cross_correlation.c',
  '../webrtc/common_audio/signal_processing/cross_correlation_neon.c',
  '../webrtc/common_audio/signal_processing/division_operations.c',
  '../webrtc/common_audio/signal_processing/dot_product_with_scale.cc',
  '../webrtc/common_audio/signal_processing/downsample_fast.c',
  '../webrtc/common_audio/signal_processing/downsample_fast_neon.c',
  '../webrtc/common_audio/signal_processing/energy.c',
  '../webrtc/common_audio/signal_processing/filter_ar.c',
  '../webrtc/common_audio/signal_processing/filter_ar_fast_q12.c',
  '../webrtc/common_audio/signal_processing/filter_ma_fast_q12.c',
  '../webrtc/common_audio/signal_processing/get_hanning_window.c',
  '../webrtc/common_audio/signal_processing/get_scaling_square.c',
  '../webrtc/common_audio/signal_processing/ilbc_specific_functions.c',
  '../webrtc/common_audio/signal_processing/levinson_durbin.c',
  '../webrtc/common_audio/signal_processing/lpc_to_refl_coef.c',
  '../webrtc/common_audio/signal_processing/min_max_operations.c',
  '../webrtc/common_audio/signal_processing/min_max_operations_neon.c',
  '../webrtc/common_audio/signal_processing/randomization_functions.c',
  '../webrtc/common_audio/signal_processing/real_fft.c',
  '../webrtc/common_audio/signal_processing/refl_coef_to_lpc.c',
  '../webrtc/common_audio/signal_processing/resample.c',
  '../webrtc/common_audio/signal_processing/resample_48khz.c',
  '../webrtc/common_audio/signal_processing/resample_by_2.c',
  '../webrtc/common_audio/signal_processing/resample_by_2_internal.c',
  '../webrtc/common_audio/signal_processing/resample_fractional.c',
  '../webrtc/common_audio/signal_processing/spl_init.c',
  '../webrtc/common_audio/signal_processing/spl_inl.c',
  '../webrtc/common_audio/signal_processing/spl_sqrt.c',
  '../webrtc/common_audio/signal_processing/splitting_filter.c',
  '../webrtc/common_audio/signal_processing/sqrt_of_one_minus_x_squared.c',
  '../webrtc/common_audio/signal_processing/vector_scaling_operations.c',
)

# Unit test sources
unit_test_sources = files(
  'unit/rms_level_unittest.cc',
  'unit/audio_buffer_unittest.cc',
  'unit/splitting_filter_unittest.cc',
  'unit/high_pass_filter_unittest.cc',
  'unit/audio_frame_view_unittest.cc',
  'unit/residual_echo_detector_unittest.cc',
  'unit/gain_control_unittest.cc',
  'unit/gain_controller2_unittest.cc',
  'unit/echo_control_mobile_unittest.cc',
  'unit/echo_control_mobile_bit_exact_unittest.cc',
  'unit/audio_processing_impl_unittest.cc',
  'unit/audio_processing_unittest.cc',
  'unit/agc2/limiter_unittest.cc',
  'unit/agc2/gain_applier_unittest.cc',
  'unit/agc2/biquad_filter_unittest.cc',
  'unit/agc2/speech_level_estimator_unittest.cc',
  'unit/agc2/noise_level_estimator_unittest.cc',
  'unit/agc2/saturation_protector_unittest.cc',
  'unit/agc2/fixed_digital_level_estimator_unittest.cc',
  'unit/agc2/adaptive_digital_gain_controller_unittest.cc',
  'unit/agc2/limiter_db_gain_curve_unittest.cc',
  'unit/agc2/interpolated_gain_curve_unittest.cc',
  'unit/agc2/clipping_predictor_level_buffer_unittest.cc',
  'unit/agc2/saturation_protector_buffer_unittest.cc',
  'unit/agc2/speech_probability_buffer_unittest.cc',
  'unit/agc2/clipping_predictor_unittest.cc',
  'unit/agc2/input_volume_stats_reporter_unittest.cc',
  'unit/agc2/input_volume_controller_unittest.cc',
  'unit/agc2/agc2_testing_common_unittest.cc',
  'unit/agc2/vad_wrapper_unittest.cc',
  'unit/echo_canceller_test_tools_unittest.cc',
  'unit/aec3/fft_data_unittest.cc',
  'unit/aec3/moving_average_unittest.cc',
  'unit/aec3/vector_math_unittest.cc',
  'unit/aec3/decimator_unittest.cc',
  'unit/aec3/clockdrift_detector_unittest.cc',
  'unit/aec3/erl_estimator_unittest.cc',
  'unit/aec3/echo_path_variability_unittest.cc',
  'unit/aec3/block_delay_buffer_unittest.cc',
  'unit/aec3/aec3_fft_unittest.cc',
  'unit/aec3/alignment_mixer_unittest.cc',
  'unit/aec3/api_call_jitter_metrics_unittest.cc',
  'unit/aec3/multi_channel_content_detector_unittest.cc',
  'unit/aec3/block_framer_unittest.cc',
  'unit/aec3/frame_blocker_unittest.cc',
  'unit/aec3/comfort_noise_generator_unittest.cc',
  'unit/aec3/suppression_gain_unittest.cc',
  'unit/aec3/erle_estimator_unittest.cc',
  'unit/aec3/render_signal_analyzer_unittest.cc',
  'unit/aec3/suppression_filter_unittest.cc',
  'unit/aec3/residual_echo_estimator_unittest.cc',
  'unit/aec3/reverb_model_estimator_unittest.cc',
  'unit/aec3/subtractor_unittest.cc',
  'unit/aec3/signal_dependent_erle_estimator_unittest.cc',
  'unit/aec3/echo_remover_unittest.cc',
  'unit/aec3/render_delay_buffer_unittest.cc',
  'unit/aec3/adaptive_fir_filter_unittest.cc',
  'unit/aec3/block_processor_unittest.cc',
  'unit/aec3/matched_filter_unittest.cc',
  'unit/aec3/aec_state_unittest.cc',
  'unit/aec3/echo_canceller3_unittest.cc',
  'unit/aec3/adaptive_fir_filter_erl_unittest.cc',
  'unit/aec3/block_processor_metrics_unittest.cc',
  'unit/aec3/coarse_filter_update_gain_unittest.cc',
  'unit/aec3/config_selector_unittest.cc',
  'unit/aec3/echo_path_delay_estimator_unittest.cc',
  'unit/aec3/echo_remover_metrics_unittest.cc',
  'unit/aec3/filter_analyzer_unittest.cc',
  'unit/aec3/matched_filter_lag_aggregator_unittest.cc',
  'unit/aec3/refined_filter_update_gain_unittest.cc',
  'unit/aec3/render_buffer_unittest.cc',
  'unit/aec3/render_delay_controller_metrics_unittest.cc',
  'unit/aec3/render_delay_controller_unittest.cc',
  'unit/echo_detector/circular_buffer_unittest.cc',
  'unit/echo_detector/mean_variance_estimator_unittest.cc',
  'unit/echo_detector/moving_max_unittest.cc',
  'unit/echo_detector/normalized_covariance_estimator_unittest.cc',
  'unit/utility/cascaded_biquad_filter_unittest.cc',
  'unit/utility/delay_estimator_unittest.cc',
  'unit/utility/pffft_wrapper_unittest.cc',
  'unit/capture_levels_adjuster/audio_samples_scaler_unittest.cc',
  'unit/capture_levels_adjuster/capture_levels_adjuster_unittest.cc',
  'unit/ns/noise_suppressor_unittest.cc',
  'unit/vad/gmm_unittest.cc',
  'unit/vad/pitch_based_vad_unittest.cc',
  'unit/vad/pitch_internal_unittest.cc',
  'unit/vad/pole_zero_filter_unittest.cc',
  'unit/vad/standalone_vad_unittest.cc',
  'unit/vad/vad_audio_proc_unittest.cc',
  'unit/vad/vad_circular_buffer_unittest.cc',
  'unit/vad/voice_activity_detector_unittest.cc',
  'unit/agc/agc_manager_direct_unittest.cc',
  'unit/agc/loudness_histogram_unittest.cc',
  # RNN_VAD tests
  'unit/agc2/rnn_vad/auto_correlation_unittest.cc',
  'unit/agc2/rnn_vad/features_extraction_unittest.cc',
  'unit/agc2/rnn_vad/lp_residual_unittest.cc',
  'unit/agc2/rnn_vad/pitch_search_internal_unittest.cc',
  'unit/agc2/rnn_vad/pitch_search_unittest.cc',
  'unit/agc2/rnn_vad/ring_buffer_unittest.cc',
  'unit/agc2/rnn_vad/rnn_fc_unittest.cc',
  'unit/agc2/rnn_vad/rnn_gru_unittest.cc',
  'unit/agc2/rnn_vad/rnn_unittest.cc',
  'unit/agc2/rnn_vad/rnn_vad_unittest.cc',
  'unit/agc2/rnn_vad/sequence_buffer_unittest.cc',
  'unit/agc2/rnn_vad/spectral_features_internal_unittest.cc',
  'unit/agc2/rnn_vad/spectral_features_unittest.cc',
  'unit/agc2/rnn_vad/symmetric_matrix_buffer_unittest.cc',
  'unit/agc2/rnn_vad/vector_math_unittest.cc',
  # Common Audio tests
  'unit/common_audio/audio_converter_unittest.cc',
  'unit/common_audio/audio_util_unittest.cc',
  'unit/common_audio/channel_buffer_unittest.cc',
  'unit/common_audio/smoothing_filter_unittest.cc',
  'unit/common_audio/ring_buffer_unittest.cc',
  'unit/common_audio/fir_filter_unittest.cc',
  'unit/common_audio/resampler/push_resampler_unittest.cc',
  'unit/common_audio/resampler/push_sinc_resampler_unittest.cc',
  'unit/common_audio/resampler/resampler_unittest.cc',
  'unit/common_audio/resampler/sinc_resampler_unittest.cc',
  'unit/common_audio/vad/vad_core_unittest.cc',
  'unit/common_audio/vad/vad_filterbank_unittest.cc',
  'unit/common_audio/vad/vad_gmm_unittest.cc',
  'unit/common_audio/vad/vad_sp_unittest.cc',
  'unit/common_audio/vad/vad_unittest.cc',
  'unit/common_audio/signal_processing/real_fft_unittest.cc',
  'unit/common_audio/signal_processing/signal_processing_unittest.cc',
)

# Build test executable with custom main that enables metrics
absl_flags_parse_dep = dependency('absl_flags_parse', required: false)

apm_unit_tests_deps = [
  gtest_dep,
  gmock_dep,
  audio_processing_dep,
  test_utils_dep,
]
if have_protobuf
  apm_unit_tests_deps += protobuf_dep
endif
if absl_flags_parse_dep.found()
  apm_unit_tests_deps += absl_flags_parse_dep
endif
if use_rust_backend
  apm_unit_tests_deps += rust_apm_dep
endif

unit_tests_cxxflags = common_cxxflags + ['-DWEBRTC_APM_DEBUG_DUMP=0', '-DWEBRTC_AUDIOPROC_FLOAT_PROFILE']
if have_protobuf
  unit_tests_cxxflags += '-DWEBRTC_ENABLE_PROTOBUF=1'
endif

apm_unit_tests = executable('apm_unit_tests',
  unit_test_sources + common_audio_test_sources + files('test_main.cc'),
  dependencies: apm_unit_tests_deps,
  include_directories: webrtc_inc,
  cpp_args: unit_tests_cxxflags,
)

# Register tests with Meson
test('apm_unit_tests', apm_unit_tests,
  timeout: 120,
  protocol: 'gtest',
)
