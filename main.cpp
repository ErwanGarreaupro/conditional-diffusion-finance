#include "FinancialDataset.hpp"
#include "VPSchedule.hpp"
#include "NoisePredictor.hpp"
#include "DiffusionTrainer.hpp"
#include "ReverseDiffusion.hpp"

#include <torch/torch.h>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    try
    {
        const std::string csv_path = "france_day_ahead_daily.csv";
        const std::string rte_csv_path =
            "RTE_eCO2mix_moyennes_journalieres_2023_2025.csv";

        const std::string training_cutoff = "2025-01-01";
        const std::string backtest_end = "2025-12-31";
        const std::string output_path = "electricity_backtest_2025.csv";

        const std::int64_t lookback = 60;
        const std::int64_t rte_lookback = 7;
        const std::int64_t horizon = 7;
        const std::int64_t rte_features_per_day = 4;
        const std::int64_t calendar_features_per_day = 4;

        const std::int64_t context_size =
            lookback +
            rte_lookback * rte_features_per_day +
            calendar_features_per_day * horizon;

        const std::int64_t n_scenarios = 1000;
        const double validation_fraction = 0.2;

        NetworkConfig network_config;
        network_config.context_size = context_size;
        network_config.horizon = horizon;
        network_config.hidden_layers = {128, 128};

        VPSchedule schedule(0.1, 20.0);

        TrainingConfig training_config;
        training_config.batch_size = 128;
        training_config.epochs = 100;
        training_config.learning_rate = 1e-3;
        training_config.weight_decay = 1e-3;
        training_config.delta = 1e-4;

        SamplingConfig sampling_config;
        sampling_config.n_steps = 1000;
        sampling_config.delta = 1e-3;
        sampling_config.horizon = horizon;

        FinancialDataset dataset(
            csv_path,
            rte_csv_path
        );

        dataset.load();

        auto data = dataset.prepareTrainingData(
            training_cutoff,
            lookback,
            horizon,
            validation_fraction
        );

        if (data.context_train.dim() != 2 ||
            data.y0_train.dim() != 2 ||
            data.context_val.dim() != 2 ||
            data.y0_val.dim() != 2 ||
            data.context_train.size(1) != context_size ||
            data.context_val.size(1) != context_size ||
            data.y0_train.size(1) != horizon ||
            data.y0_val.size(1) != horizon ||
            data.context_train.size(0) != data.y0_train.size(0) ||
            data.context_val.size(0) != data.y0_val.size(0))
        {
            throw std::runtime_error(
                "Invalid electricity dataset dimensions"
            );
        }

        std::cout
            << "\n========== ELECTRICITY DATASET ==========\n"
            << "Market: France Day-Ahead\n"
            << "Unit: EUR/MWh\n"
            << "Price CSV: " << csv_path << "\n"
            << "RTE CSV: " << rte_csv_path << "\n"
            << "Training cutoff: " << training_cutoff << "\n"
            << "Price lookback: " << lookback << " days\n"
            << "RTE lookback: " << rte_lookback << " days\n"
            << "RTE lag: 0 days\n"
            << "Forecast horizon: " << horizon << " days\n"
            << "RTE features per historical day: "
            << rte_features_per_day << "\n"
            << "Calendar features per future day: "
            << calendar_features_per_day << "\n"
            << "Total context size: " << context_size << "\n"
            << "Network input size: "
            << context_size + horizon + 1 << "\n"
            << "Training mean: " << data.mean << "\n"
            << "Training std: " << data.std_dev << "\n"
            << "Training windows: "
            << data.y0_train.size(0) << "\n"
            << "Validation windows: "
            << data.y0_val.size(0) << "\n";

        torch::manual_seed(42);

        NoisePredictor model(network_config);
        DiffusionTrainer trainer(
            schedule,
            training_config
        );

        std::cout
            << "\n========== TRAINING WITH VALIDATION ==========\n";

        trainer.train(
            model,
            data.y0_train,
            data.context_train,
            data.y0_val,
            data.context_val
        );

        model->eval();

        torch::manual_seed(123);

        ReverseDiffusion sampler(
            schedule,
            sampling_config
        );

        const auto& all_dates = dataset.dates();

        const std::size_t first_index =
            dataset.findDateIndex(training_cutoff);

        const std::size_t last_index =
            dataset.findDateIndex(backtest_end);

        const std::size_t H =
            static_cast<std::size_t>(horizon);

        std::ofstream output(output_path);

        if (!output.is_open())
        {
            throw std::runtime_error(
                "Cannot create CSV file: " + output_path
            );
        }

        output << std::setprecision(12)
               << "PredictionDate,Scenario";

        for (std::int64_t day = 1; day <= horizon; ++day)
        {
            output << ",Day_" << day;
        }

        output << "\n";

        std::cout
            << "\n========== BACKTESTING 2025 ==========\n"
            << "Price lookback: " << lookback << " days\n"
            << "RTE lookback: " << rte_lookback << " days\n"
            << "Forecast horizon: " << horizon << " days\n"
            << "Context size: " << context_size << "\n"
            << "Scenarios per forecast: " << n_scenarios << "\n";

        std::size_t n_forecasts = 0;

        for (
            std::size_t index = first_index;
            index + H - 1 <= last_index;
            index += H
        )
        {
            const std::string current_date =
                all_dates[index];

            std::cout
                << "\n----------------------------------------\n"
                << "Forecast date: " << current_date << "\n";

            auto context = dataset.getContext(
                current_date,
                lookback,
                horizon
            );

            if (context.dim() != 2 ||
                context.size(0) != 1 ||
                context.size(1) != context_size)
            {
                throw std::runtime_error(
                    "Invalid prediction context shape"
                );
            }

            auto generation_contexts =
                context.repeat({n_scenarios, 1});

            torch::Tensor normalized_scenarios;

            {
                torch::NoGradGuard no_grad;

                normalized_scenarios = sampler.sample(
                    model,
                    generation_contexts
                );
            }

            auto scenarios = dataset.denormalize(
                normalized_scenarios
            );

            auto paths = scenarios
                .to(torch::kCPU, torch::kFloat32)
                .contiguous();

            if (paths.dim() != 2 ||
                paths.size(0) != n_scenarios ||
                paths.size(1) != horizon)
            {
                throw std::runtime_error(
                    "Unexpected scenario tensor dimensions"
                );
            }

            auto accessor = paths.accessor<float, 2>();

            for (
                std::int64_t scenario = 0;
                scenario < n_scenarios;
                ++scenario
            )
            {
                output << current_date
                       << ","
                       << scenario + 1;

                for (
                    std::int64_t day = 0;
                    day < horizon;
                    ++day
                )
                {
                    output << ","
                           << accessor[scenario][day];
                }

                output << "\n";
            }

            ++n_forecasts;

            std::cout
                << "Generated: "
                << n_scenarios
                << " scenarios\n"
                << "Forecast number: "
                << n_forecasts
                << "\n";
        }

        output.close();

        if (!output)
        {
            throw std::runtime_error(
                "Error writing backtesting CSV"
            );
        }

        std::cout
            << "\n========== BACKTEST COMPLETED ==========\n"
            << "Forecast dates: "
            << n_forecasts << "\n"
            << "Scenarios per forecast: "
            << n_scenarios << "\n"
            << "Total scenarios: "
            << n_forecasts * n_scenarios << "\n"
            << "Output: "
            << output_path << "\n";
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