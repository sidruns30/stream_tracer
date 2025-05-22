/*
    * global.hpp
    *
    * Define global variables and functions
*/
#ifndef GLOBAL_HPP
    #define GLOBAL_HPP

    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <pybind11/stl.h> 
    #include <vector>
    #include <string>
    #include <iostream>
    #include <stdexcept>
    #include <tuple>
    #include <functional>
    #include <omp.h>
    #include <cmath>

    namespace py = pybind11;
    // Minimum precesion required for the grid to be considered uniform
    const float relative_precision              = 0.01;
    // Number of iterations to display the progress
    const std::size_t Ndisplay                  = 20;
    // Stepsize in units of dx (or dr)
    constexpr float CFL                         = 10.0;
    // Integration method to use
    enum class IntegrationMethod {
                                Euler,
                                RK4
    };
    constexpr IntegrationMethod integration_method = IntegrationMethod::RK4;
    // Minimum radial distance from the center of the grid to terminate the streamlines
    const float inner_termination_radius        = 5.;
    // Minimum fraction of streamlines to integrate before terminating integration
    const float terminate_fraction              = 0.99;
    // OpenMP thread count - this should be set to the number of threads available
    constexpr int number_of_threads             = 128;
    // Minimum distance between two points to be considered the same
    const float softening_length                = 1e-5;
    // Phi coordinate limits for spherical coordinates (-PI, PI)
    const double phi_min                        = -3.141592653589793;
    const double phi_max                        = 3.141592653589793;

    template <typename T>
    T square(T x) 
    {   return x * x;   }

    template <typename T>
    std::size_t to_size_t(T x)
    {   return static_cast<std::size_t>(x);   }

#endif