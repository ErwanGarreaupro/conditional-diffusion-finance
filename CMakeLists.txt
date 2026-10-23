cmake_minimum_required(VERSION 3.18)

project(conditional_diffusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(conditional_diffusion

    main.cpp

    cpp/VPSchedule.cpp
    cpp/ForwardDiffusion.cpp
    cpp/NoisePredictor.cpp
    cpp/DiffusionTrainer.cpp
    cpp/ReverseDiffusion.cpp
    cpp/FinancialDataset.cpp

)

target_include_directories(conditional_diffusion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)

target_link_libraries(conditional_diffusion PRIVATE
    "${TORCH_LIBRARIES}"
)