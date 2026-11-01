#include "FinancialDataset.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr std::size_t RTE_LOOKBACK = 14;

std::tm parseDate(const std::string& date) {
    if (date.size() != 10 || date[4] != '-' || date[7] != '-')
        throw std::runtime_error("Invalid date: " + date);

    std::tm tm{};
    tm.tm_year = std::stoi(date.substr(0, 4)) - 1900;
    tm.tm_mon = std::stoi(date.substr(5, 2)) - 1;
    tm.tm_mday = std::stoi(date.substr(8, 2));
    tm.tm_hour = 12;
    tm.tm_isdst = -1;

    const auto year = tm.tm_year;
    const auto month = tm.tm_mon;
    const auto day = tm.tm_mday;

    if (std::mktime(&tm) == static_cast<std::time_t>(-1) ||
        tm.tm_year != year ||
        tm.tm_mon != month ||
        tm.tm_mday != day)
        throw std::runtime_error("Invalid calendar date: " + date);

    return tm;
}

void appendCalendar(
    std::vector<float>& out,
    const std::string& date
) {
    const auto tm = parseDate(date);

    const double w =
        2.0 * PI * ((tm.tm_wday + 6) % 7) / 7.0;

    const double y =
        2.0 * PI * tm.tm_yday / 365.25;

    out.push_back(static_cast<float>(std::sin(w)));
    out.push_back(static_cast<float>(std::cos(w)));
    out.push_back(static_cast<float>(std::sin(y)));
    out.push_back(static_cast<float>(std::cos(y)));
}

std::vector<std::string> split(const std::string& line) {
    std::stringstream ss(line);
    std::vector<std::string> cells;
    std::string cell;

    while (std::getline(ss, cell, ','))
        cells.push_back(cell);

    return cells;
}

double number(
    const std::string& text,
    const std::string& date
) {
    try {
        std::size_t n = 0;
        const double x = std::stod(text, &n);

        if (n != text.size() || !std::isfinite(x))
            throw std::runtime_error("Invalid number");

        return x;
    }
    catch (...) {
        throw std::runtime_error(
            "Invalid numeric value at " + date + ": " + text
        );
    }
}

torch::Tensor makeTensor(
    const std::vector<float>& values,
    std::int64_t rows,
    std::int64_t cols
) {
    return torch::from_blob(
        const_cast<float*>(values.data()),
        {rows, cols},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
}

}

FinancialDataset::FinancialDataset(
    const std::string& csv_path,
    const std::string& rte_csv_path
)
    : csv_path_(csv_path),
      rte_csv_path_(rte_csv_path)
{
}

void FinancialDataset::load() {
    dates_.clear();
    prices_.clear();
    rte_.clear();
    normalization_ready_ = false;

    std::ifstream prices_file(csv_path_);

    if (!prices_file)
        throw std::runtime_error("Cannot open " + csv_path_);

    std::string line;

    if (!std::getline(prices_file, line))
        throw std::runtime_error("Empty price CSV");

    if (!line.empty() && line.back() == '\r')
        line.pop_back();

    if (line.size() >= 3 &&
        static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF)
    {
        line.erase(0, 3);
    }

    if (line != "Date,Price")
        throw std::runtime_error(
            "Unexpected price CSV header: [" + line + "]"
        );

    while (std::getline(prices_file, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty())
            continue;

        const auto fields = split(line);

        if (fields.size() != 2)
            throw std::runtime_error(
                "Invalid price CSV row: " + line
            );

        parseDate(fields[0]);

        if (!dates_.empty() && fields[0] <= dates_.back())
            throw std::runtime_error(
                "Price dates must increase"
            );

        dates_.push_back(fields[0]);
        prices_.push_back(number(fields[1], fields[0]));
    }

    if (prices_.size() < 2)
        throw std::runtime_error("Not enough prices");

    std::ifstream rte_file(rte_csv_path_);

    if (!rte_file)
        throw std::runtime_error("Cannot open " + rte_csv_path_);

    if (!std::getline(rte_file, line))
        throw std::runtime_error("Empty RTE CSV");

    if (!line.empty() && line.back() == '\r')
        line.pop_back();

    if (line.size() >= 3 &&
        static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF)
    {
        line.erase(0, 3);
    }

    while (!line.empty() &&
           (line.back() == ' ' || line.back() == '\t'))
    {
        line.pop_back();
    }

    if (line != "Date,Consommation,Eolien,Solaire,Nucleaire")
        throw std::runtime_error(
            "Unexpected RTE CSV header: [" + line + "]"
        );

    std::map<std::string, std::array<double, 4>> rte_by_date;

    while (std::getline(rte_file, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty())
            continue;

        const auto fields = split(line);

        if (fields.size() != 5)
            throw std::runtime_error(
                "Invalid RTE CSV row: " + line
            );

        parseDate(fields[0]);

        std::array<double, 4> values{};

        for (std::size_t k = 0; k < 4; ++k)
            values[k] = number(fields[k + 1], fields[0]);

        if (!rte_by_date.emplace(fields[0], values).second)
            throw std::runtime_error(
                "Duplicate RTE date: " + fields[0]
            );
    }

    for (const auto& date : dates_) {
        const auto it = rte_by_date.find(date);

        if (it == rte_by_date.end())
            throw std::runtime_error(
                "Missing RTE data for " + date
            );

        rte_.push_back(it->second);
    }
}

std::size_t FinancialDataset::findDateIndex(
    const std::string& date
) const {
    const auto it = std::lower_bound(
        dates_.begin(),
        dates_.end(),
        date
    );

    if (it == dates_.end() || *it != date)
        throw std::invalid_argument(
            "Date not found: " + date
        );

    return static_cast<std::size_t>(
        it - dates_.begin()
    );
}

std::vector<float> FinancialDataset::buildContext(
    std::size_t end_index,
    std::size_t lookback,
    std::size_t horizon
) const {
    if (!normalization_ready_)
        throw std::runtime_error(
            "Normalization unavailable"
        );

    if (lookback == 0 ||
        horizon == 0 ||
        end_index < lookback ||
        end_index < RTE_LOOKBACK ||
        end_index > prices_.size() ||
        horizon > prices_.size() - end_index)
        throw std::runtime_error(
            "Invalid context window"
        );

    std::vector<float> values;

    values.reserve(
        lookback + 4 * RTE_LOOKBACK + 4 * horizon
    );

    for (
        std::size_t i = end_index - lookback;
        i < end_index;
        ++i
    ) {
        values.push_back(
            static_cast<float>(
                (prices_[i] - mean_) / std_dev_
            )
        );
    }

    for (
        std::size_t i = end_index - RTE_LOOKBACK;
        i < end_index;
        ++i
    ) {
        for (std::size_t k = 0; k < 4; ++k) {
            values.push_back(
                static_cast<float>(
                    (rte_[i][k] - rte_mean_[k]) /
                    rte_std_[k]
                )
            );
        }
    }

    for (std::size_t j = 0; j < horizon; ++j)
        appendCalendar(
            values,
            dates_[end_index + j]
        );

    return values;
}

PreparedData FinancialDataset::prepareTrainingData(
    const std::string& prediction_date,
    std::int64_t lookback,
    std::int64_t horizon,
    double validation_fraction
) {
    if (lookback <= 0 ||
        horizon <= 0 ||
        !(validation_fraction > 0.0 &&
          validation_fraction < 1.0))
        throw std::invalid_argument(
            "Invalid training configuration"
        );

    const std::size_t available =
        findDateIndex(prediction_date);

    const auto L =
        static_cast<std::size_t>(lookback);

    const auto H =
        static_cast<std::size_t>(horizon);

    const std::size_t first_window =
        std::max(L, RTE_LOOKBACK);

    const std::size_t train_end =
        static_cast<std::size_t>(
            (1.0 - validation_fraction) * available
        );

    if (train_end < first_window + H ||
        available - train_end < H)
        throw std::runtime_error(
            "Insufficient train/validation data"
        );

    mean_ = std::accumulate(
        prices_.begin(),
        prices_.begin() + train_end,
        0.0
    ) / train_end;

    double variance = 0.0;

    for (std::size_t i = 0; i < train_end; ++i)
        variance += std::pow(
            prices_[i] - mean_,
            2
        );

    std_dev_ = std::sqrt(
        variance / train_end
    );

    if (!std::isfinite(std_dev_) ||
        std_dev_ <= 0.0)
        throw std::runtime_error(
            "Invalid price variance"
        );

    for (std::size_t k = 0; k < 4; ++k) {
        rte_mean_[k] = 0.0;

        for (std::size_t i = 0; i < train_end; ++i)
            rte_mean_[k] += rte_[i][k];

        rte_mean_[k] /= train_end;

        double var = 0.0;

        for (std::size_t i = 0; i < train_end; ++i)
            var += std::pow(
                rte_[i][k] - rte_mean_[k],
                2
            );

        rte_std_[k] = std::sqrt(
            var / train_end
        );

        if (!std::isfinite(rte_std_[k]) ||
            rte_std_[k] <= 0.0)
            throw std::runtime_error(
                "Invalid RTE variance"
            );
    }

    normalization_ready_ = true;

    std::vector<float> train_context;
    std::vector<float> train_targets;
    std::vector<float> val_context;
    std::vector<float> val_targets;

    std::int64_t n_train = 0;
    std::int64_t n_val = 0;

    for (
        std::size_t target_start = first_window;
        target_start + H <= available;
        ++target_start
    ) {
        const bool is_train =
            target_start + H <= train_end;

        const bool is_val =
            target_start >= train_end;

        if (!is_train && !is_val)
            continue;

        auto& contexts = is_train
            ? train_context
            : val_context;

        auto& targets = is_train
            ? train_targets
            : val_targets;

        const auto context = buildContext(
            target_start,
            L,
            H
        );

        contexts.insert(
            contexts.end(),
            context.begin(),
            context.end()
        );

        for (std::size_t j = 0; j < H; ++j) {
            targets.push_back(
                static_cast<float>(
                    (prices_[target_start + j] - mean_) /
                    std_dev_
                )
            );
        }

        if (is_train)
            ++n_train;
        else
            ++n_val;
    }

    if (n_train == 0 || n_val == 0)
        throw std::runtime_error(
            "No training or validation windows"
        );

    const std::int64_t context_size =
        lookback +
        4 * static_cast<std::int64_t>(RTE_LOOKBACK) +
        4 * horizon;

    PreparedData result;

    result.context_train = makeTensor(
        train_context,
        n_train,
        context_size
    );

    result.y0_train = makeTensor(
        train_targets,
        n_train,
        horizon
    );

    result.context_val = makeTensor(
        val_context,
        n_val,
        context_size
    );

    result.y0_val = makeTensor(
        val_targets,
        n_val,
        horizon
    );

    result.mean = mean_;
    result.std_dev = std_dev_;

    return result;
}

torch::Tensor FinancialDataset::getContext(
    const std::string& prediction_date,
    std::int64_t lookback,
    std::int64_t horizon
) const {
    if (lookback <= 0 || horizon <= 0)
        throw std::invalid_argument(
            "Invalid context dimensions"
        );

    const auto values = buildContext(
        findDateIndex(prediction_date),
        static_cast<std::size_t>(lookback),
        static_cast<std::size_t>(horizon)
    );

    const std::int64_t context_size =
        lookback +
        4 * static_cast<std::int64_t>(RTE_LOOKBACK) +
        4 * horizon;

    return makeTensor(
        values,
        1,
        context_size
    );
}

double FinancialDataset::getPrice(
    const std::string& prediction_date
) const {
    return prices_[findDateIndex(prediction_date)];
}

torch::Tensor FinancialDataset::denormalize(
    const torch::Tensor& normalized_prices
) const {
    if (!normalization_ready_)
        throw std::runtime_error(
            "Normalization unavailable"
        );

    return normalized_prices * std_dev_ + mean_;
}

const std::vector<std::string>& FinancialDataset::dates() const {
    return dates_;
}

const std::vector<double>& FinancialDataset::prices() const {
    return prices_;
}