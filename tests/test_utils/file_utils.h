/*
 *  Copyright (c) 2024 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#ifndef TESTS_TEST_UTILS_FILE_UTILS_H_
#define TESTS_TEST_UTILS_FILE_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"

namespace webrtc {
namespace test {

// Returns the absolute path to a resource file.
// The path is constructed as: <project_root>/tests/resources/<name>.<extension>
// This function requires WEBRTC_TEST_RESOURCES_DIR to be defined at compile
// time, pointing to the test resources directory.
std::string ResourcePath(absl::string_view name, absl::string_view extension);

// Returns the path to test resource files used by audio processing tests.
// The name should be something like "far48_stereo" and extension "pcm".
// This is a convenience function that wraps ResourcePath.
std::string GetApmTestResourcePath(absl::string_view filename);

// Returns the path to a directory for output files.
std::string OutputPath();

// Creates a temporary filename with the given prefix in the given directory.
std::string TempFilename(absl::string_view dir, absl::string_view prefix);

}  // namespace test
}  // namespace webrtc

#endif  // TESTS_TEST_UTILS_FILE_UTILS_H_
