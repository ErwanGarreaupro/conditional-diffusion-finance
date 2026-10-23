#include "DiffusionTrainer.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>

DiffusionTrainer::DiffusionTrainer(
    const VPSchedule& schedule,
    const TrainingConfig& config
)
    : schedule_(schedule), config_(config)
{
    // Validate training hyperparameters
    if (config_.batch_size <= 0 ||
        config_.epochs <= 0 ||
        config_.learning_rate <= 0.0 ||
        config_.delta <= 0.0 ||
        config_.delta >= 1.0)
    {
        throw std::invalid_argument(
            "Invalid training parameters"
        );
    }
}

void DiffusionTrainer::train(
    NoisePredictor& model,
    const torch::Tensor& y0_data,
    const torch::Tensor& context_data
)
{

    const auto n_samples = y0_data.size(0);

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(config_.learning_rate)
            .weight_decay(config_.weight_decay)
    );

    model->train();

    for (std::int64_t epoch = 0; epoch < config_.epochs; ++epoch)
    {
        auto permutation = torch::randperm(
            n_samples,
            torch::TensorOptions().dtype(torch::kLong)
        );

        double epoch_loss = 0.0;
        std::int64_t batches = 0;

        for (std::int64_t start = 0;
             start < n_samples;
             start += config_.batch_size)
        {
            //end of current mini batch
            const auto end = std::min(
                start + config_.batch_size,
                n_samples
            );

            auto indices = permutation.slice(0, start, end);

            auto y0 = y0_data.index_select(0, indices);
            auto context = context_data.index_select(0, indices);

            const auto current_batch = y0.size(0);

            auto s = config_.delta +
                (1.0 - config_.delta) *
                torch::rand({current_batch, 1});

            auto epsilon = torch::randn_like(y0);

            auto beta_integral = 0.0 * s;

            for (std::int64_t i = 0; i < current_batch; ++i)
            {
                const double si = s[i].item<double>();

                beta_integral[i] =
                    schedule_.integratedBeta(si);
            }

            auto alpha = torch::exp(
                -0.5 * beta_integral
            );

            auto sigma = torch::sqrt(
                1.0 - torch::exp(-beta_integral)
            );

            auto y_s =
                alpha * y0 + sigma * epsilon;

            auto epsilon_hat =
                model->forward(y_s, s, context);

            auto loss = torch::mse_loss(
                epsilon_hat,
                epsilon
            );

            optimizer.zero_grad();

            loss.backward();

            optimizer.step();

            epoch_loss += loss.item<double>();
            ++batches;
        }
    }
}