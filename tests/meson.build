# Include directories for tests - include project root for "tests/test_utils/..." includes
tests_inc = include_directories('..')

# Test utilities
test_utils_sources = files(
  'test_utils/audio_buffer_tools.cc',
)

test_utils_lib = static_library('test_utils',
  test_utils_sources,
  dependencies: [
    gtest_dep,
    audio_processing_dep,
  ],
  include_directories: [webrtc_inc, tests_inc],
  cpp_args: common_cxxflags,
)

test_utils_dep = declare_dependency(
  link_with: test_utils_lib,
  include_directories: tests_inc,
)

# Unit test sources
unit_test_sources = files(
  'unit/rms_level_unittest.cc',
  'unit/audio_buffer_unittest.cc',
  'unit/splitting_filter_unittest.cc',
  'unit/high_pass_filter_unittest.cc',
)

# Build test executable
apm_unit_tests = executable('apm_unit_tests',
  unit_test_sources,
  dependencies: [
    gtest_dep,
    gtest_main_dep,
    gmock_dep,
    audio_processing_dep,
    test_utils_dep,
  ],
  include_directories: webrtc_inc,
  cpp_args: common_cxxflags,
)

# Register tests with Meson
test('apm_unit_tests', apm_unit_tests,
  timeout: 120,
  protocol: 'gtest',
)
