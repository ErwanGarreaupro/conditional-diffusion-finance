#include "FinancialDataset.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

// Constructor
FinancialDataset::FinancialDataset(
    const std::string& csv_path
)
    : csv_path_(csv_path)
{
}

// ============================================================
// Load historical closing prices and compute log returns
// ============================================================

void FinancialDataset::load()
{
    std::ifstream file(csv_path_);

    if (!file.is_open())
    {
        throw std::runtime_error(
            "Cannot open CSV file: " + csv_path_
        );
    }

    dates_.clear();
    prices_.clear();
    returns_.clear();

    mean_ = 0.0;
    std_dev_ = 1.0;
    normalization_ready_ = false;

    std::string line;

    // Skip CSV header
    if (!std::getline(file, line))
    {
        throw std::runtime_error(
            "CSV file is empty"
        );
    }

    // Read historical observations
    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream ss(line);

        std::string date;
        std::string open;
        std::string high;
        std::string low;
        std::string close;
        std::string volume;

        std::getline(ss, date, ',');
        std::getline(ss, open, ',');
        std::getline(ss, high, ',');
        std::getline(ss, low, ',');
        std::getline(ss, close, ',');
        std::getline(ss, volume, ',');

        if (date.empty() || close.empty())
        {
            throw std::runtime_error(
                "Missing date or closing price in CSV"
            );
        }

        double price = 0.0;

        try
        {
            price = std::stod(close);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                "Invalid closing price at date: " + date
            );
        }

        if (!std::isfinite(price) || price <= 0.0)
        {
            throw std::runtime_error(
                "Closing price must be positive at: " + date
            );
        }

        dates_.push_back(date);
        prices_.push_back(price);
    }

    if (prices_.size() < 2)
    {
        throw std::runtime_error(
            "Not enough prices to compute returns"
        );
    }

    // Check that dates are strictly increasing
    for (std::size_t i = 1; i < dates_.size(); ++i)
    {
        if (dates_[i] <= dates_[i - 1])
        {
            throw std::runtime_error(
                "CSV dates must be strictly increasing"
            );
        }
    }

    returns_.reserve(prices_.size() - 1);

    // Compute daily log returns
    for (std::size_t i = 1; i < prices_.size(); ++i)
    {
        const double r = std::log(
            prices_[i] / prices_[i - 1]
        );

        returns_.push_back(r);
    }
}

// ============================================================
// Find the index of an exact trading date
// ============================================================

std::size_t FinancialDataset::findDateIndex(
    const std::string& date
) const
{
    const auto it = std::lower_bound(
        dates_.begin(),
        dates_.end(),
        date
    );

    if (it == dates_.end() || *it != date)
    {
        throw std::invalid_argument(
            "Trading date not found: " + date
        );
    }

    return static_cast<std::size_t>(
        std::distance(dates_.begin(), it)
    );
}

// ============================================================
// Prepare normalized training and validation datasets
// ============================================================

PreparedData FinancialDataset::prepareTrainingData(
    const std::string& prediction_date,
    std::int64_t context_size,
    std::int64_t horizon,
    double validation_fraction
)
{
    if (context_size <= 0 || horizon <= 0)
    {
        throw std::invalid_argument(
            "Context size and horizon must be positive"
        );
    }

    if (validation_fraction <= 0.0 ||
        validation_fraction >= 1.0)
    {
        throw std::invalid_argument(
            "Validation fraction must be between 0 and 1"
        );
    }

    // Price index at the prediction date
    const std::size_t prediction_index =
        findDateIndex(prediction_date);

    // returns_[i] corresponds to dates_[i + 1]
    // Therefore, prediction_index returns are available
    const std::size_t n_available_returns =
        prediction_index;

    const std::size_t L =
        static_cast<std::size_t>(context_size);

    const std::size_t H =
        static_cast<std::size_t>(horizon);

    if (n_available_returns < L + H + 2)
    {
        throw std::runtime_error(
            "Not enough historical returns"
        );
    }

    // Chronological split
    const std::size_t train_end =
        static_cast<std::size_t>(
            (1.0 - validation_fraction) *
            static_cast<double>(n_available_returns)
        );

    if (train_end < L + H ||
        n_available_returns - train_end < H)
    {
        throw std::runtime_error(
            "Not enough data for training or validation"
        );
    }

    // Compute mean using training returns only
    const double sum = std::accumulate(
        returns_.begin(),
        returns_.begin() + train_end,
        0.0
    );

    mean_ = sum / static_cast<double>(train_end);

    // Compute training variance
    double variance = 0.0;

    for (std::size_t i = 0; i < train_end; ++i)
    {
        const double diff = returns_[i] - mean_;

        variance += diff * diff;
    }

    variance /= static_cast<double>(train_end);

    std_dev_ = std::sqrt(variance);

    if (!std::isfinite(std_dev_) || std_dev_ <= 0.0)
    {
        throw std::runtime_error(
            "Invalid training standard deviation"
        );
    }

    // Normalize returns with training statistics
    std::vector<double> normalized_returns(
        n_available_returns
    );

    for (std::size_t i = 0;
         i < n_available_returns;
         ++i)
    {
        normalized_returns[i] =
            (returns_[i] - mean_) / std_dev_;
    }

    std::vector<float> context_train_values;
    std::vector<float> y0_train_values;

    std::vector<float> context_val_values;
    std::vector<float> y0_val_values;

    std::int64_t n_train = 0;
    std::int64_t n_val = 0;

    // Build sliding windows
    for (std::size_t start = 0;
         start + L + H <= n_available_returns;
         ++start)
    {
        const std::size_t target_start = start + L;
        const std::size_t target_end = target_start + H;

        // Training target must be entirely before split
        const bool is_train =
            target_end <= train_end;

        // Validation target must be entirely after split
        const bool is_validation =
            target_start >= train_end &&
            target_end <= n_available_returns;

        // Skip windows crossing the boundary
        if (!is_train && !is_validation)
        {
            continue;
        }

        auto& context_values = is_train
            ? context_train_values
            : context_val_values;

        auto& target_values = is_train
            ? y0_train_values
            : y0_val_values;

        // Historical context
        for (std::size_t j = 0; j < L; ++j)
        {
            context_values.push_back(
                static_cast<float>(
                    normalized_returns[start + j]
                )
            );
        }

        // Future target
        for (std::size_t j = 0; j < H; ++j)
        {
            target_values.push_back(
                static_cast<float>(
                    normalized_returns[target_start + j]
                )
            );
        }

        if (is_train)
        {
            ++n_train;
        }
        else
        {
            ++n_val;
        }
    }

    if (n_train == 0 || n_val == 0)
    {
        throw std::runtime_error(
            "No training or validation windows generated"
        );
    }

    // Create independent tensors from flattened vectors
    auto makeTensor = [](
        const std::vector<float>& values,
        std::int64_t rows,
        std::int64_t cols
    ) -> torch::Tensor
    {
        return torch::from_blob(
            const_cast<float*>(values.data()),
            {rows, cols},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();
    };

    PreparedData result;

    result.context_train = makeTensor(
        context_train_values,
        n_train,
        context_size
    );

    result.y0_train = makeTensor(
        y0_train_values,
        n_train,
        horizon
    );

    result.context_val = makeTensor(
        context_val_values,
        n_val,
        context_size
    );

    result.y0_val = makeTensor(
        y0_val_values,
        n_val,
        horizon
    );

    result.mean = mean_;
    result.std_dev = std_dev_;

    normalization_ready_ = true;

    return result;
}

// ============================================================
// Extract normalized historical context
// ============================================================

torch::Tensor FinancialDataset::getContext(
    const std::string& prediction_date,
    std::int64_t context_size
) const
{
    if (!normalization_ready_)
    {
        throw std::runtime_error(
            "Call prepareTrainingData before getContext"
        );
    }

    if (context_size <= 0)
    {
        throw std::invalid_argument(
            "Context size must be positive"
        );
    }

    const std::size_t index =
        findDateIndex(prediction_date);

    const std::size_t L =
        static_cast<std::size_t>(context_size);

    if (index < L)
    {
        throw std::runtime_error(
            "Not enough historical returns for context"
        );
    }

    std::vector<float> values;
    values.reserve(L);

    // Last known return is returns_[index - 1]
    for (std::size_t i = index - L;
         i < index;
         ++i)
    {
        values.push_back(
            static_cast<float>(
                (returns_[i] - mean_) / std_dev_
            )
        );
    }

    return torch::from_blob(
        values.data(),
        {1, context_size},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
}

// ============================================================
// Get closing price at prediction date
// ============================================================

double FinancialDataset::getPrice(
    const std::string& prediction_date
) const
{
    return prices_[findDateIndex(prediction_date)];
}

// ============================================================
// Undo return normalization
// ============================================================

torch::Tensor FinancialDataset::denormalize(
    const torch::Tensor& normalized_returns
) const
{
    if (!normalization_ready_)
    {
        throw std::runtime_error(
            "Normalization statistics are not available"
        );
    }

    return normalized_returns * std_dev_ + mean_;
}

// ============================================================
// Historical data accessors
// ============================================================

const std::vector<std::string>&
FinancialDataset::dates() const
{
    return dates_;
}

const std::vector<double>&
FinancialDataset::prices() const
{
    return prices_;
}

const std::vector<double>&
FinancialDataset::returns() const
{
    return returns_;
}