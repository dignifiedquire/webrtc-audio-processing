/*
 *  Copyright (c) 2024 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#include "tests/test_utils/file_utils.h"

#include <string>
#include <unistd.h>
#include <cstdlib>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

std::string ResourcePath(absl::string_view name, absl::string_view extension) {
#ifdef WEBRTC_TEST_RESOURCES_DIR
  std::string path = WEBRTC_TEST_RESOURCES_DIR;
  path += "/";
  path += std::string(name);
  path += ".";
  path += std::string(extension);
  return path;
#else
  RTC_CHECK(false) << "WEBRTC_TEST_RESOURCES_DIR is not defined";
  return "";
#endif
}

std::string GetApmTestResourcePath(absl::string_view filename) {
#ifdef WEBRTC_TEST_RESOURCES_DIR
  std::string path = WEBRTC_TEST_RESOURCES_DIR;
  path += "/";
  path += std::string(filename);
  return path;
#else
  RTC_CHECK(false) << "WEBRTC_TEST_RESOURCES_DIR is not defined";
  return "";
#endif
}

std::string OutputPath() {
  // Use /tmp for test output files
  return "/tmp/";
}

std::string TempFilename(absl::string_view dir, absl::string_view prefix) {
  std::string filename = std::string(dir);
  if (!filename.empty() && filename.back() != '/') {
    filename += '/';
  }
  filename += std::string(prefix);
  filename += "_XXXXXX";
  
  // Create a unique temp file, then delete it (we just want the unique name)
  char* temp = new char[filename.size() + 1];
  std::copy(filename.begin(), filename.end(), temp);
  temp[filename.size()] = '\0';
  int fd = mkstemp(temp);
  if (fd != -1) {
    close(fd);
    // Remove the file - caller will create it if needed
    unlink(temp);
  }
  std::string result(temp);
  delete[] temp;
  return result;
}

}  // namespace test
}  // namespace webrtc
