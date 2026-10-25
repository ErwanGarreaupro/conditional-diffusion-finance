#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

struct NetworkConfig
{
    std::int64_t horizon = 60;
    std::int64_t context_size = 60;

    std::vector<std::int64_t> hidden_layers = {
        128, 128
    };
};

class NoisePredictorImpl : public torch::nn::Module
{
private:
    NetworkConfig config_;

    torch::nn::Sequential network_{nullptr};

public:
    explicit NoisePredictorImpl(
        const NetworkConfig& config
    );

    torch::Tensor forward(
        const torch::Tensor& y_s,
        const torch::Tensor& s,
        const torch::Tensor& context
    );
};

TORCH_MODULE(NoisePredictor);