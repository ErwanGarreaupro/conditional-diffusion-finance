#include "ReverseDiffusion.hpp"

#include <cmath>
#include <stdexcept>

ReverseDiffusion::ReverseDiffusion(
    const VPSchedule& schedule,
    const SamplingConfig& config
)
    : schedule_(schedule), config_(config)
{
    if (config_.n_steps <= 0 ||
        config_.delta <= 0.0 ||
        config_.delta >= 1.0)
    {
        throw std::invalid_argument(
            "Invalid sampling parameters"
        );
    }
}

torch::Tensor ReverseDiffusion::sample(
    NoisePredictor& model,
    const torch::Tensor& context
)
{
    if (context.dim() != 2 ||
        context.size(0) <= 0)
    {
        throw std::invalid_argument(
            "Context must have shape [batch_size, context_size]"
        );
    }

    model->eval();

    torch::NoGradGuard no_grad;

    const auto batch_size = context.size(0);

    const std::int64_t horizon = config_.horizon;

    auto y = torch::randn(
        {batch_size, horizon},
        context.options()
    );

    const double h =
        (1.0 - config_.delta) / config_.n_steps;

    for (std::int64_t k = 0;
         k < config_.n_steps;
         ++k)
    {
        const double s =
            1.0 - k * h;

        const double beta_s = schedule_.beta(s);
        const double sigma_s = schedule_.sigma(s);

        auto s_tensor = torch::full(
            {batch_size, 1},
            s,
            context.options()
        );

        auto epsilon_hat =
            model->forward(y, s_tensor, context);

        auto drift =
            0.5 * beta_s * y
            - (beta_s / sigma_s) * epsilon_hat;

        auto z = torch::randn_like(y);

        y = y
            + drift * h
            + std::sqrt(beta_s * h) * z;
    }

    return y;
}