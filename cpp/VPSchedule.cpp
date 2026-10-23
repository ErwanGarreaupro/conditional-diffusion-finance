#include "VPSchedule.hpp"
#include <cmath>
#include <stdexcept>

VPSchedule::VPSchedule(double beta_min, double beta_max): beta_min_(beta_min), beta_max_(beta_max)
{
    if (beta_min_ <= 0.0 ||beta_max_ < beta_min_) {
        throw std::invalid_argument("Invalid beta parameters");
    }
}

double VPSchedule::beta(double s) const             
{
    if (s < 0.0 || s > 1.0) {
        throw std::invalid_argument("s must be in [0, 1]");
    }
    return beta_min_ + s * (beta_max_ - beta_min_);
}

double VPSchedule::integratedBeta(double s) const
{
    if (s < 0.0 || s > 1.0) {
        throw std::invalid_argument("s must be in [0, 1]");
    }
    return beta_min_ * s
         + 0.5 * (beta_max_ - beta_min_) * s * s;
}

double VPSchedule::alpha(double s) const
{
    return std::exp(-0.5 * integratedBeta(s));
}

double VPSchedule::sigma(double s) const
{
    return std::sqrt(-std::exp(-integratedBeta(s))+1);
}