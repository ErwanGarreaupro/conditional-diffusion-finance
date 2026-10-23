#pragma once

class VPSchedule {
private:
    double beta_min_;
    double beta_max_;

public:
    VPSchedule(double beta_min, double beta_max);
    double beta(double s) const;
    double integratedBeta(double s) const;
    double alpha(double s) const;
    double sigma(double s) const;
};