#pragma once
#include "NoisePredictor.hpp"
#include "VPSchedule.hpp"


struct TrainingConfig
{
    std::int64_t batch_size = 128;
    std::int64_t epochs = 100;

    double learning_rate = 1e-3;
    double weight_decay = 1e-4;

    double delta = 1e-4;
};

class DiffusionTrainer
{
private:
    VPSchedule schedule_;
    TrainingConfig config_;

public:
    DiffusionTrainer(
        const VPSchedule& schedule,
        const TrainingConfig& config
    );

    void train(
        NoisePredictor& model,
        const torch::Tensor& y0_data,
        const torch::Tensor& context_data
    );
};