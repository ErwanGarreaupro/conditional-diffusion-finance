#include "NoisePredictor.hpp"

#include <stdexcept>

NoisePredictorImpl::NoisePredictorImpl(
    const NetworkConfig& config
)
    : config_(config)
{
    if (config_.horizon <= 0 ||
        config_.context_size <= 0) {
        throw std::invalid_argument(
            "Invalid network dimensions"
        );
    }

    const std::int64_t input_size =
        config_.horizon + config_.context_size + 1;       //s the artificial time

    network_ = register_module(
        "network",
        torch::nn::Sequential()
    );

    std::int64_t previous_size = input_size;

    for (const auto hidden_size : config_.hidden_layers)
    {
        if (hidden_size <= 0) {
            throw std::invalid_argument(
                "Hidden layer size must be positive"
            );
        }

        network_->push_back(
            torch::nn::Linear(previous_size, hidden_size)
        );

        network_->push_back(torch::nn::SiLU());

        previous_size = hidden_size;
    }

    network_->push_back(
        torch::nn::Linear(previous_size, config_.horizon)
    );
}

torch::Tensor NoisePredictorImpl::forward(
    const torch::Tensor& y_s,
    const torch::Tensor& s,
    const torch::Tensor& context
)
{

    if (y_s.dim() != 2 ||
        s.dim() != 2 ||
        context.dim() != 2) {
        throw std::invalid_argument(
            "All inputs must be 2D tensors"
        );
    }

    const auto batch_size = y_s.size(0);                 //batch size_=number of s simulated for our trajectories

    if (y_s.size(1) != config_.horizon ||
        s.size(1) != 1 ||
        context.size(1) != config_.context_size ||
        s.size(0) != batch_size ||
        context.size(0) != batch_size) {
        throw std::invalid_argument(
            "Invalid input tensor dimensions"
        );
    }

    auto input = torch::cat(
        {y_s, s, context},
        1
    );

    return network_->forward(input);
}