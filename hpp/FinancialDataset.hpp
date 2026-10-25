#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct PreparedData
{
    torch::Tensor context_train;
    torch::Tensor y0_train;

    torch::Tensor context_val;
    torch::Tensor y0_val;

    double mean = 0.0;
    double std_dev = 1.0;
};

class FinancialDataset
{
private:
    std::string csv_path_;

    std::vector<std::string> dates_;
    std::vector<double> prices_;
    std::vector<double> returns_;

    double mean_ = 0.0;
    double std_dev_ = 1.0;

    bool normalization_ready_ = false;

    std::vector<float> buildContext(
        std::size_t end_index,
        std::size_t lookback
    ) const;

public:
    explicit FinancialDataset(const std::string& csv_path);

    void load();

    std::size_t findDateIndex(
        const std::string& date
    ) const;

    PreparedData prepareTrainingData(
        const std::string& prediction_date,
        std::int64_t lookback,
        std::int64_t horizon,
        double validation_fraction = 0.2
    );

    torch::Tensor getContext(
        const std::string& prediction_date,
        std::int64_t lookback
    ) const;

    double getPrice(
        const std::string& prediction_date
    ) const;

    torch::Tensor denormalize(
        const torch::Tensor& normalized_returns
    ) const;

    const std::vector<std::string>& dates() const;
    const std::vector<double>& prices() const;
    const std::vector<double>& returns() const;
};