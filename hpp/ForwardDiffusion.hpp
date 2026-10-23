#pragma once

#include "VPSchedule.hpp"

#include <cstdint>
#include <random>
#include <vector>

// Result of the forward diffusion process
struct NoisedSample
{
    std::vector<double> y_s;
    std::vector<double> epsilon;
    double s;
};

// Forward diffusion using a variance-preserving SDE
class ForwardDiffusion
{
private:
    VPSchedule schedule_;

    std::mt19937_64 generator_;
    std::normal_distribution<double> normal_;

public:
    // Constructor
    ForwardDiffusion(
        const VPSchedule& schedule,
        std::uint64_t seed = 42
    );

    // Apply forward diffusion with a given noise vector
    std::vector<double> applyNoise(
        const std::vector<double>& y0,
        double s,
        const std::vector<double>& epsilon
    ) const;

    // Generate Gaussian noise and apply forward diffusion
    NoisedSample sample(
        const std::vector<double>& y0,
        double s
    );
};