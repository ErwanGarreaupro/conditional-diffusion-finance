#pragma once

#include "NoisePredictor.hpp"
#include "VPSchedule.hpp"

#include <torch/torch.h>
#include <cstdint>


struct SamplingConfig
{
    std::int64_t n_steps = 1000;
    double delta = 1e-3;

    std::int64_t horizon = 60;
};

class ReverseDiffusion
{
private:
    VPSchedule schedule_;
    SamplingConfig config_;

public:
    ReverseDiffusion(
        const VPSchedule& schedule,
        const SamplingConfig& config
    );

    torch::Tensor sample(
        NoisePredictor& model,
        const torch::Tensor& context
    );
};