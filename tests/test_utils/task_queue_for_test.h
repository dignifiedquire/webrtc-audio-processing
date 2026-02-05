/*
 *  Copyright 2018 The WebRTC Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

// Stub implementation of TaskQueueForTest for tests that don't actually
// need async task execution.

#ifndef TESTS_TEST_UTILS_TASK_QUEUE_FOR_TEST_H_
#define TESTS_TEST_UTILS_TASK_QUEUE_FOR_TEST_H_

#include <string>

#include "absl/strings/string_view.h"
#include "webrtc/api/task_queue/task_queue_base.h"

namespace webrtc {

// Stub TaskQueueForTest - the tests that use this (DebugDump tests) are
// disabled when WEBRTC_AUDIOPROC_DEBUG_DUMP is not defined.
class TaskQueueForTest {
 public:
  explicit TaskQueueForTest(absl::string_view name = "TestQueue") {}
  TaskQueueForTest(const TaskQueueForTest&) = delete;
  TaskQueueForTest& operator=(const TaskQueueForTest&) = delete;
  ~TaskQueueForTest() = default;

  TaskQueueBase* Get() { return nullptr; }
};

}  // namespace webrtc

#endif  // TESTS_TEST_UTILS_TASK_QUEUE_FOR_TEST_H_
