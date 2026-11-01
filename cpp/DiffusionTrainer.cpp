#include "DiffusionTrainer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

DiffusionTrainer::DiffusionTrainer(
    const VPSchedule& schedule,
    const TrainingConfig& config
)
    : schedule_(schedule), config_(config)
{
    if (config_.batch_size <= 0 ||
        config_.epochs <= 0 ||
        config_.learning_rate <= 0.0 ||
        config_.weight_decay < 0.0 ||
        config_.delta <= 0.0 ||
        config_.delta >= 1.0)
    {
        throw std::invalid_argument("Invalid training parameters");
    }
}

void DiffusionTrainer::train(
    NoisePredictor& model,
    const torch::Tensor& y0_train,
    const torch::Tensor& context_train,
    const torch::Tensor& y0_val,
    const torch::Tensor& context_val
)
{
    if (y0_train.dim() != 2 ||
        context_train.dim() != 2 ||
        y0_val.dim() != 2 ||
        context_val.dim() != 2 ||
        y0_train.size(0) == 0 ||
        y0_val.size(0) == 0 ||
        y0_train.size(0) != context_train.size(0) ||
        y0_val.size(0) != context_val.size(0) ||
        y0_train.size(1) != y0_val.size(1) ||
        context_train.size(1) != context_val.size(1))
    {
        throw std::invalid_argument("Invalid training or validation tensors");
    }

    const auto n_samples = y0_train.size(0);
    const auto n_validation = y0_val.size(0);

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(config_.learning_rate)
            .weight_decay(config_.weight_decay)
    );

    auto integratedBeta = [&](const torch::Tensor& s)
    {
        auto result = torch::empty_like(s);
        auto accessor = result.accessor<float, 2>();

        for (std::int64_t i = 0; i < s.size(0); ++i)
            accessor[i][0] = static_cast<float>(
                schedule_.integratedBeta(s[i][0].item<double>())
            );

        return result;
    };

    auto makeNoisyInput = [&](const torch::Tensor& y0,
                              const torch::Tensor& s,
                              const torch::Tensor& epsilon)
    {
        auto beta_integral = integratedBeta(s);
        auto alpha = torch::exp(-0.5 * beta_integral);
        auto sigma = torch::sqrt(
            torch::clamp_min(1.0 - torch::exp(-beta_integral), 0.0)
        );

        return alpha * y0 + sigma * epsilon;
    };

    torch::manual_seed(2025);

    auto s_val = config_.delta +
        (1.0 - config_.delta) *
        torch::rand({n_validation, 1});

    auto epsilon_val = torch::randn_like(y0_val);
    auto y_s_val = makeNoisyInput(y0_val, s_val, epsilon_val);

    double best_val_loss = std::numeric_limits<double>::infinity();
    std::int64_t best_epoch = 0;

    std::vector<torch::Tensor> best_parameters;

    for (std::int64_t epoch = 0; epoch < config_.epochs; ++epoch)
    {
        model->train();

        auto permutation = torch::randperm(
            n_samples,
            torch::TensorOptions().dtype(torch::kLong)
        );

        double train_loss_sum = 0.0;
        std::int64_t train_batches = 0;

        for (std::int64_t start = 0;
             start < n_samples;
             start += config_.batch_size)
        {
            const auto end = std::min(
                start + config_.batch_size,
                n_samples
            );

            auto indices = permutation.slice(0, start, end);
            auto y0 = y0_train.index_select(0, indices);
            auto context = context_train.index_select(0, indices);

            const auto current_batch = y0.size(0);

            auto s = config_.delta +
                (1.0 - config_.delta) *
                torch::rand({current_batch, 1});

            auto epsilon = torch::randn_like(y0);
            auto y_s = makeNoisyInput(y0, s, epsilon);

            auto epsilon_hat = model->forward(y_s, s, context);
            auto loss = torch::mse_loss(epsilon_hat, epsilon);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            train_loss_sum += loss.item<double>() * current_batch;
            ++train_batches;
        }

        const double train_loss =
            train_loss_sum / static_cast<double>(n_samples);

        model->eval();

        double val_loss = 0.0;

        {
            torch::NoGradGuard no_grad;

            for (std::int64_t start = 0;
                 start < n_validation;
                 start += config_.batch_size)
            {
                const auto end = std::min(
                    start + config_.batch_size,
                    n_validation
                );

                auto y_s = y_s_val.slice(0, start, end);
                auto s = s_val.slice(0, start, end);
                auto epsilon = epsilon_val.slice(0, start, end);
                auto context = context_val.slice(0, start, end);

                auto epsilon_hat = model->forward(y_s, s, context);
                auto loss = torch::mse_loss(epsilon_hat, epsilon);

                val_loss += loss.item<double>() * (end - start);
            }
        }

        val_loss /= static_cast<double>(n_validation);

        if (!std::isfinite(train_loss) || !std::isfinite(val_loss))
            throw std::runtime_error("Non-finite training or validation loss");

        if (val_loss < best_val_loss)
        {
            best_val_loss = val_loss;
            best_epoch = epoch + 1;
            best_parameters.clear();

            torch::NoGradGuard no_grad;

            for (const auto& parameter : model->parameters())
                best_parameters.push_back(parameter.detach().clone());
        }

        std::cout
            << "Epoch " << epoch + 1
            << "/" << config_.epochs
            << " | Train Loss: " << train_loss
            << " | Val Loss: " << val_loss
            << " | Best Val: " << best_val_loss
            << " (epoch " << best_epoch << ")"
            << std::endl;
    }

    {
        torch::NoGradGuard no_grad;

        auto parameters = model->parameters();

        if (parameters.size() != best_parameters.size())
            throw std::runtime_error("Model parameter count changed");

        for (std::size_t i = 0; i < parameters.size(); ++i)
            parameters[i].copy_(best_parameters[i]);
    }

    model->eval();

    std::cout
        << "\n========== BEST MODEL ==========\n"
        << "Best epoch: " << best_epoch << "\n"
        << "Best validation loss: " << best_val_loss << "\n"
        << "Best model parameters restored\n";
}