#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

struct NetworkConfig
{
    std::int64_t horizon = 20;                   //Days of projection
    std::int64_t context_size = 30;              //Days of context

    std::vector<std::int64_t> hidden_layers = {
        130, 130
    };                                           //config of the neural network
};

class NoisePredictorImpl : public torch::nn::Module
{
private:
    NetworkConfig config_;
    torch::nn::Sequential network_{nullptr};

public:
    explicit NoisePredictorImpl(const NetworkConfig& config);

    torch::Tensor forward(
        const torch::Tensor& y_s,
        const torch::Tensor& s,
        const torch::Tensor& context           //param taking into account
    );
};

TORCH_MODULE(NoisePredictor);