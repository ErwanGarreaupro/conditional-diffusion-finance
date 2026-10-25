#include "FinancialDataset.hpp"
#include "VPSchedule.hpp"
#include "NoisePredictor.hpp"
#include "DiffusionTrainer.hpp"
#include "ReverseDiffusion.hpp"

#include <torch/torch.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <cmath>

double computeVolatility(
    const std::vector<double>& returns,
    std::size_t start,
    std::size_t length
)
{
    if (length < 2 || start + length > returns.size())
    {
        throw std::runtime_error(
            "Invalid volatility window"
        );
    }

    double mean = 0.0;

    for (std::size_t i = start; i < start + length; ++i)
    {
        mean += returns[i];
    }

    mean /= static_cast<double>(length);

    double variance = 0.0;

    for (std::size_t i = start; i < start + length; ++i)
    {
        const double difference = returns[i] - mean;
        variance += difference * difference;
    }

    variance /= static_cast<double>(length - 1);

    return std::sqrt(variance);
}

void saveEvaluation(
    const std::string& output_path,
    const torch::Tensor& scenarios,
    const std::vector<std::size_t>& selected_indices,
    const std::vector<std::string>& dates,
    const std::vector<double>& past_volatilities,
    const std::vector<double>& real_future_volatilities,
    std::int64_t scenarios_per_context,
    std::int64_t horizon
)
{
    const std::int64_t n_contexts =
        static_cast<std::int64_t>(selected_indices.size());

    const std::int64_t expected_rows =
        n_contexts * scenarios_per_context;

    auto paths = scenarios
        .to(torch::kCPU, torch::kFloat32)
        .contiguous();

    if (paths.dim() != 2 ||
        paths.size(0) != expected_rows ||
        paths.size(1) != horizon)
    {
        throw std::runtime_error(
            "Unexpected generated scenarios shape"
        );
    }

    std::ofstream output(output_path);

    if (!output.is_open())
    {
        throw std::runtime_error(
            "Cannot create output CSV file: " + output_path
        );
    }

    output << std::setprecision(12);

    output
        << "Date,Scenario,PastVolatility,RealFutureVolatility";

    for (std::int64_t day = 1; day <= horizon; ++day)
    {
        output << ",Day_" << day;
    }

    output << "\n";

    auto accessor = paths.accessor<float, 2>();

    for (std::int64_t i = 0; i < n_contexts; ++i)
    {
        const std::string& date =
            dates[selected_indices[i]];

        for (std::int64_t j = 0;
             j < scenarios_per_context;
             ++j)
        {
            const std::int64_t row =
                i * scenarios_per_context + j;

            output
                << date
                << "," << j + 1
                << "," << past_volatilities[i]
                << "," << real_future_volatilities[i];

            for (std::int64_t day = 0; day < horizon; ++day)
            {
                output << "," << accessor[row][day];
            }

            output << "\n";
        }
    }

    output.close();

    if (!output)
    {
        throw std::runtime_error(
            "Error writing CSV file: " + output_path
        );
    }

    std::cout
        << "\nSaved: " << output_path
        << "\nRows: " << expected_rows
        << "\nColumns: " << horizon + 4
        << "\n";
}

int main()
{
    try
    {
        const std::string csv_path = "aapl_us.csv";
        const std::string prediction_date = "2022-12-29";

        const std::int64_t lookback = 60;
        const std::int64_t n_contexts = 50;
        const std::int64_t scenarios_per_context = 100;

        const double validation_fraction = 0.2;

        NetworkConfig network_config;

        network_config.context_size = lookback;
        network_config.horizon = 60;
        network_config.hidden_layers = {128, 128};

        const std::size_t L =
            static_cast<std::size_t>(lookback);

        const std::size_t H =
            static_cast<std::size_t>(network_config.horizon);

        VPSchedule schedule(0.1, 20.0);

        TrainingConfig training_config;

        training_config.batch_size = 128;
        training_config.epochs = 100;
        training_config.learning_rate = 1e-3;
        training_config.weight_decay = 1e-4;
        training_config.delta = 1e-4;

        SamplingConfig sampling_config;

        sampling_config.n_steps = 1000;
        sampling_config.delta = 1e-3;
        sampling_config.horizon = network_config.horizon;

        FinancialDataset dataset(csv_path);

        dataset.load();

        auto data = dataset.prepareTrainingData(
            prediction_date,
            lookback,
            network_config.horizon,
            validation_fraction
        );

        if (data.context_train.size(1) != network_config.context_size ||
            data.context_val.size(1) != network_config.context_size)
        {
            throw std::runtime_error(
                "Dataset context dimension does not match network"
            );
        }

        std::cout
            << "\n========== DATASET ==========\n"
            << "Training mean: " << data.mean << "\n"
            << "Training std: " << data.std_dev << "\n"
            << "Training windows: " << data.y0_train.size(0) << "\n"
            << "Validation windows: " << data.y0_val.size(0) << "\n";

        const auto& dates = dataset.dates();
        const auto& returns = dataset.returns();

        const std::size_t prediction_index =
            dataset.findDateIndex(prediction_date);

        const std::size_t train_end =
            static_cast<std::size_t>(
                (1.0 - validation_fraction) *
                static_cast<double>(prediction_index)
            );

        const std::size_t first_context_end =
            train_end;

        if (prediction_index < H)
        {
            throw std::runtime_error(
                "Not enough data for future volatility evaluation"
            );
        }

        const std::size_t last_context_end =
            prediction_index - H;

        if (first_context_end < L ||
            last_context_end < first_context_end)
        {
            throw std::runtime_error(
                "Not enough validation data"
            );
        }

        const std::size_t available_contexts =
            last_context_end - first_context_end + 1;

        if (n_contexts < 2 ||
            available_contexts <
                static_cast<std::size_t>(n_contexts))
        {
            throw std::runtime_error(
                "Invalid number of validation contexts"
            );
        }

        std::vector<std::size_t> selected_indices;
        selected_indices.reserve(n_contexts);

        for (std::int64_t i = 0; i < n_contexts; ++i)
        {
            const std::size_t offset =
                static_cast<std::size_t>(
                    (
                        static_cast<double>(i) *
                        static_cast<double>(available_contexts - 1)
                    ) /
                    static_cast<double>(n_contexts - 1)
                );

            selected_indices.push_back(
                first_context_end + offset
            );
        }

        std::vector<torch::Tensor> context_tensors;
        context_tensors.reserve(n_contexts);

        std::vector<double> past_volatilities;
        std::vector<double> real_future_volatilities;

        past_volatilities.reserve(n_contexts);
        real_future_volatilities.reserve(n_contexts);

        for (const std::size_t index : selected_indices)
        {
            const std::string& date = dates[index];

            auto context = dataset.getContext(
                date,
                lookback
            );

            if (context.dim() != 2 ||
                context.size(1) != network_config.context_size)
            {
                throw std::runtime_error(
                    "Invalid generated context dimension"
                );
            }

            context_tensors.push_back(
                context.repeat({
                    scenarios_per_context,
                    1
                })
            );

            past_volatilities.push_back(
                computeVolatility(
                    returns,
                    index - L,
                    L
                )
            );

            real_future_volatilities.push_back(
                computeVolatility(
                    returns,
                    index,
                    H
                )
            );
        }

        auto all_contexts = torch::cat(
            context_tensors,
            0
        );

        auto zero_train_contexts = torch::zeros_like(
            data.context_train
        );

        auto zero_generation_contexts = torch::zeros_like(
            all_contexts
        );

        std::cout
            << "\n========== V1 MLP EVALUATION ==========\n"
            << "Architecture: Simple MLP\n"
            << "Historical lookback: " << lookback << "\n"
            << "Context features: "
            << network_config.context_size << "\n"
            << "Forecast horizon: "
            << network_config.horizon << "\n"
            << "Validation contexts: " << n_contexts << "\n"
            << "Scenarios per context: "
            << scenarios_per_context << "\n"
            << "Total scenarios per model: "
            << n_contexts * scenarios_per_context << "\n"
            << "First date: "
            << dates[selected_indices.front()] << "\n"
            << "Last date: "
            << dates[selected_indices.back()] << "\n";

        ReverseDiffusion sampler(
            schedule,
            sampling_config
        );

        {
            std::cout
                << "\n========== CONDITIONAL V1 MLP ==========\n";

            torch::manual_seed(42);

            NoisePredictor model(network_config);

            DiffusionTrainer trainer(
                schedule,
                training_config
            );

            trainer.train(
                model,
                data.y0_train,
                data.context_train
            );

            torch::manual_seed(123);

            auto normalized_scenarios = sampler.sample(
                model,
                all_contexts
            );

            auto scenarios = dataset.denormalize(
                normalized_scenarios
            );

            saveEvaluation(
                "conditional_v1_evaluation.csv",
                scenarios,
                selected_indices,
                dates,
                past_volatilities,
                real_future_volatilities,
                scenarios_per_context,
                network_config.horizon
            );
        }

        {
            std::cout
                << "\n========== UNCONDITIONAL V1 MLP ==========\n";

            torch::manual_seed(42);

            NoisePredictor model(network_config);

            DiffusionTrainer trainer(
                schedule,
                training_config
            );

            trainer.train(
                model,
                data.y0_train,
                zero_train_contexts
            );

            torch::manual_seed(123);

            auto normalized_scenarios = sampler.sample(
                model,
                zero_generation_contexts
            );

            auto scenarios = dataset.denormalize(
                normalized_scenarios
            );

            saveEvaluation(
                "unconditional_v1_evaluation.csv",
                scenarios,
                selected_indices,
                dates,
                past_volatilities,
                real_future_volatilities,
                scenarios_per_context,
                network_config.horizon
            );
        }

        std::cout
            << "\n========== V1 EVALUATION COMPLETED ==========\n"
            << "Conditional: conditional_v1_evaluation.csv\n"
            << "Unconditional: unconditional_v1_evaluation.csv\n";
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "\nError: "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}