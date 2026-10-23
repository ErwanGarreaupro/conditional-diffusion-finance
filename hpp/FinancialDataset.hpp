#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Prepared tensors for model training and validation
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

    // Historical observations
    std::vector<std::string> dates_;
    std::vector<double> prices_;
    std::vector<double> returns_;

    // Training normalization statistics
    double mean_ = 0.0;
    double std_dev_ = 1.0;

    bool normalization_ready_ = false;

public:
    explicit FinancialDataset(const std::string& csv_path);

    // Load CSV and compute daily log returns
    void load();

    // Find an exact trading date
    std::size_t findDateIndex(
        const std::string& date
    ) const;

    // Prepare chronological training and validation windows
    PreparedData prepareTrainingData(
        const std::string& prediction_date,
        std::int64_t context_size,
        std::int64_t horizon,
        double validation_fraction = 0.2
    );

    // Extract the normalized context at prediction date
    torch::Tensor getContext(
        const std::string& prediction_date,
        std::int64_t context_size
    ) const;

    // Get closing price at prediction date
    double getPrice(
        const std::string& prediction_date
    ) const;

    // Convert normalized returns to original log returns
    torch::Tensor denormalize(
        const torch::Tensor& normalized_returns
    ) const;

    // Historical data accessors
    const std::vector<std::string>& dates() const;
    const std::vector<double>& prices() const;
    const std::vector<double>& returns() const;
};