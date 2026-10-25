#include "FinancialDataset.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

FinancialDataset::FinancialDataset(
    const std::string& csv_path
)
    : csv_path_(csv_path)
{
}

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

    if (!std::getline(file, line))
    {
        throw std::runtime_error("CSV file is empty");
    }

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

    for (std::size_t i = 1; i < prices_.size(); ++i)
    {
        returns_.push_back(
            std::log(prices_[i] / prices_[i - 1])
        );
    }
}

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

std::vector<float> FinancialDataset::buildContext(
    std::size_t end_index,
    std::size_t lookback
) const
{
    if (!normalization_ready_)
    {
        throw std::runtime_error(
            "Normalization statistics are not available"
        );
    }

    if (lookback == 0 ||
        end_index < lookback ||
        end_index > returns_.size())
    {
        throw std::runtime_error(
            "Invalid historical context window"
        );
    }

    std::vector<float> values;
    values.reserve(lookback);

    for (std::size_t i = end_index - lookback;
         i < end_index;
         ++i)
    {
        values.push_back(
            static_cast<float>(
                (returns_[i] - mean_) / std_dev_
            )
        );
    }

    return values;
}

PreparedData FinancialDataset::prepareTrainingData(
    const std::string& prediction_date,
    std::int64_t lookback,
    std::int64_t horizon,
    double validation_fraction
)
{
    if (lookback <= 0 || horizon <= 0)
    {
        throw std::invalid_argument(
            "Lookback and horizon must be positive"
        );
    }

    if (validation_fraction <= 0.0 ||
        validation_fraction >= 1.0)
    {
        throw std::invalid_argument(
            "Validation fraction must be between 0 and 1"
        );
    }

    const std::size_t prediction_index =
        findDateIndex(prediction_date);

    const std::size_t n_available_returns =
        prediction_index;

    const std::size_t L =
        static_cast<std::size_t>(lookback);

    const std::size_t H =
        static_cast<std::size_t>(horizon);

    if (n_available_returns < L + H + 2)
    {
        throw std::runtime_error(
            "Not enough historical returns"
        );
    }

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

    const double sum = std::accumulate(
        returns_.begin(),
        returns_.begin() + train_end,
        0.0
    );

    mean_ = sum / static_cast<double>(train_end);

    double variance = 0.0;

    for (std::size_t i = 0; i < train_end; ++i)
    {
        const double difference = returns_[i] - mean_;
        variance += difference * difference;
    }

    variance /= static_cast<double>(train_end);

    std_dev_ = std::sqrt(variance);

    if (!std::isfinite(std_dev_) || std_dev_ <= 0.0)
    {
        throw std::runtime_error(
            "Invalid training standard deviation"
        );
    }

    normalization_ready_ = true;

    std::vector<float> context_train_values;
    std::vector<float> y0_train_values;

    std::vector<float> context_val_values;
    std::vector<float> y0_val_values;

    std::int64_t n_train = 0;
    std::int64_t n_val = 0;

    for (std::size_t start = 0;
         start + L + H <= n_available_returns;
         ++start)
    {
        const std::size_t target_start = start + L;
        const std::size_t target_end = target_start + H;

        const bool is_train =
            target_end <= train_end;

        const bool is_validation =
            target_start >= train_end &&
            target_end <= n_available_returns;

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

        const auto context = buildContext(
            target_start,
            L
        );

        context_values.insert(
            context_values.end(),
            context.begin(),
            context.end()
        );

        for (std::size_t j = 0; j < H; ++j)
        {
            target_values.push_back(
                static_cast<float>(
                    (returns_[target_start + j] - mean_) / std_dev_
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
        lookback
    );

    result.y0_train = makeTensor(
        y0_train_values,
        n_train,
        horizon
    );

    result.context_val = makeTensor(
        context_val_values,
        n_val,
        lookback
    );

    result.y0_val = makeTensor(
        y0_val_values,
        n_val,
        horizon
    );

    result.mean = mean_;
    result.std_dev = std_dev_;

    return result;
}

torch::Tensor FinancialDataset::getContext(
    const std::string& prediction_date,
    std::int64_t lookback
) const
{
    if (lookback <= 0)
    {
        throw std::invalid_argument(
            "Lookback must be positive"
        );
    }

    const std::size_t index =
        findDateIndex(prediction_date);

    const auto values = buildContext(
        index,
        static_cast<std::size_t>(lookback)
    );

    return torch::from_blob(
        const_cast<float*>(values.data()),
        {1, lookback},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
}

double FinancialDataset::getPrice(
    const std::string& prediction_date
) const
{
    return prices_[findDateIndex(prediction_date)];
}

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