#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>
namespace th=torch;


namespace starrygl{
    enum device_t{
        CPU,
        CUDA
    };
    inline const char *device_name(device_t device)
    {
        switch (device) {
        case CPU:
            return "CPU";
        case CUDA:
            return "CUDA";
        default:
            throw std::invalid_argument("invalid device");
        }
    }
    template <typename T, typename TS> class TemporalEdge;
    template <typename T, typename TS, device_t device> class Graph;
}
