#include "ForwardDiffusion.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

ForwardDiffusion::ForwardDiffusion(
    const VPSchedule& schedule,
    std::uint64_t seed
)
    : schedule_(schedule),
      generator_(seed),
      normal_(0.0, 1.0)
{
}

std::vector<double> ForwardDiffusion::applyNoise(
    const std::vector<double>& y0,
    double s,
    const std::vector<double>& epsilon
) const
{
    if (y0.empty())
    {
        throw std::invalid_argument(
            "y0 cannot be empty"
        );
    }

    if (y0.size() != epsilon.size())
    {
        throw std::invalid_argument(
            "y0 and epsilon must have the same dimension"
        );
    }

    const double alpha_s = schedule_.alpha(s);
    const double sigma_s = schedule_.sigma(s);

    std::vector<double> y_s(y0.size());

    // Y_s = alpha(s) * Y_0 + sigma(s) * epsilon
    for (std::size_t i = 0; i < y0.size(); ++i)
    {
        if (!std::isfinite(y0[i]) ||
            !std::isfinite(epsilon[i]))
        {
            throw std::invalid_argument(
                "Inputs must contain finite values"
            );
        }

        y_s[i] =
            alpha_s * y0[i] +
            sigma_s * epsilon[i];
    }

    return y_s;
}



NoisedSample ForwardDiffusion::sample(
    const std::vector<double>& y0,
    double s
)
{
    std::vector<double> epsilon(y0.size());

    // Generate independent Gaussian noise
    for (double& value : epsilon)
    {
        value = normal_(generator_);
    }

    // Apply forward diffusion
    std::vector<double> y_s =
        applyNoise(y0, s, epsilon);

    return NoisedSample{
        std::move(y_s),
        std::move(epsilon),
        s
    };
}