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
#include <cstdint>
#include <exception>
#include <cmath>

int main()
{
    try
    {
        const std::string csv_path = "aapl_us.csv";
        const std::string prediction_date = "2025-12-31";
        const std::int64_t n_scenarios = 1000;
        const double validation_fraction = 0.2;
        torch::manual_seed(42);

        NetworkConfig network_config;

        network_config.context_size = 30;
        network_config.horizon = 20;
        network_config.hidden_layers = {128, 128};

        NoisePredictor model(network_config);


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



        FinancialDataset dataset(csv_path);

        dataset.load();

        auto data = dataset.prepareTrainingData(
            prediction_date,
            network_config.context_size,
            network_config.horizon,
            validation_fraction
        );


        DiffusionTrainer trainer(
            schedule,
            training_config
        );

        trainer.train(
            model,
            data.y0_train,
            data.context_train
        );


        auto context = dataset.getContext(
            prediction_date,
            network_config.context_size
        );

        // Repeat the same context for all scenarios
        auto contexts = context.repeat({
            n_scenarios,
            1
        });


        ReverseDiffusion sampler(
            schedule,
            sampling_config
        );

        auto normalized_scenarios = sampler.sample(
            model,
            contexts
        );

        auto scenarios = dataset.denormalize(
            normalized_scenarios
        );

        const double initial_price =
            dataset.getPrice(prediction_date);

        auto cumulative_returns = torch::cumsum(
            scenarios,
            1
        );

        auto future_prices =
            initial_price * torch::exp(cumulative_returns);

        auto initial_column = torch::full(
            {n_scenarios, 1},
            initial_price,
            future_prices.options()
        );

        auto price_paths = torch::cat(
            {initial_column, future_prices},
            1
        );


        const std::string output_path =
            "generated_apple_scenarios.csv";

        std::ofstream output(output_path);

        if (!output.is_open())
        {
            throw std::runtime_error(
                "Cannot create output CSV file"
            );
        }
  
        output << "Scenario";

        for (std::int64_t day = 0;
             day <= network_config.horizon;
             ++day)
        {
            output << ",Day_" << day;
        }

        output << "\n";


        auto paths = price_paths.contiguous();

        for (std::int64_t i = 0;
             i < n_scenarios;
             ++i)
        {
            output << i + 1;

            for (std::int64_t j = 0;
                 j <= network_config.horizon;
                 ++j)
            {
                output
                    << ","
                    << paths[i][j].item<double>();
            }

            output << "\n";
        }

        output.close();


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