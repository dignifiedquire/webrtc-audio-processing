/*
 * Tool to regenerate pitch_lp_res.dat for the current platform.
 * The reference data must be regenerated when the platform produces
 * different floating point results (e.g., ARM vs x86).
 *
 * This is compiled as a gtest binary to get FRIEND_TEST access.
 * The test name matches the one in pitch_search.h FRIEND_TEST_ALL_PREFIXES.
 */

#include <fstream>
#include <iostream>
#include <vector>
#include <array>

#include "modules/audio_processing/agc2/cpu_features.h"
#include "modules/audio_processing/agc2/rnn_vad/common.h"
#include "modules/audio_processing/agc2/rnn_vad/lp_residual.h"
#include "modules/audio_processing/agc2/rnn_vad/pitch_search.h"
#include <gtest/gtest.h>

using namespace webrtc::rnn_vad;

// Use exact test name that has FRIEND access via FRIEND_TEST_ALL_PREFIXES
// in pitch_search.h: FRIEND_TEST_ALL_PREFIXES(RnnVadTest, PitchSearchWithinTolerance)
TEST(RnnVadTest, PitchSearchWithinTolerance) {
    const char* input_file = "tests/resources/audio_processing/agc2/rnn_vad/pitch_buf_24k.dat";
    const char* output_file = "tests/resources/audio_processing/agc2/rnn_vad/pitch_lp_res_arm64.dat";
    
    std::ifstream in(input_file, std::ios::binary);
    ASSERT_TRUE(in) << "Cannot open input: " << input_file;
    
    std::ofstream out(output_file, std::ios::binary);
    ASSERT_TRUE(out) << "Cannot open output: " << output_file;
    
    std::vector<float> pitch_buffer(kBufSize24kHz);
    std::array<float, kNumLpcCoefficients> lpc;
    std::vector<float> lp_residual(kBufSize24kHz);
    
    // Get CPU features for pitch estimator
    const webrtc::AvailableCpuFeatures cpu_features = webrtc::GetAvailableCpuFeatures();
    PitchEstimator pitch_estimator(cpu_features);
    
    int frame_count = 0;
    while (in.read(reinterpret_cast<char*>(pitch_buffer.data()), kBufSize24kHz * sizeof(float))) {
        // Compute LP residual
        webrtc::rnn_vad::ComputeAndPostProcessLpcCoefficients(pitch_buffer, lpc);
        webrtc::rnn_vad::ComputeLpResidual(lpc, pitch_buffer, lp_residual);
        
        // Run pitch estimation on the LP residual
        webrtc::ArrayView<const float, kBufSize24kHz> lp_residual_view(
            lp_residual.data(), kBufSize24kHz);
        int pitch_period = pitch_estimator.Estimate(lp_residual_view);
        
        // Access pitch strength via FRIEND_TEST access
        float pitch_strength = pitch_estimator.GetLastPitchStrengthForTesting();
        
        // Write LP residual
        out.write(reinterpret_cast<const char*>(lp_residual.data()), kBufSize24kHz * sizeof(float));
        
        // Write pitch period and strength
        float pitch_info[2] = {static_cast<float>(pitch_period), pitch_strength};
        out.write(reinterpret_cast<const char*>(pitch_info), 2 * sizeof(float));
        
        frame_count++;
    }
    
    std::cout << "Generated " << frame_count << " frames to " << output_file << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
