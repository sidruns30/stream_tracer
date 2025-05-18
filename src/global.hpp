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
    const float relative_precision              = 0.01;
    const std::size_t Ndisplay                  = 20;
    constexpr float CFL                         = 0.25;
    const float inner_termination_radius        = 5.;
    const float terminate_fraction              = 0.99;
    constexpr int number_of_threads             = 128;
    const float softening_length                = 1e-5;
    const double phi_min                        = -3.141592653589793;
    const double phi_max                        = 3.141592653589793;

    template <typename T>
    T square(T x) 
    {   return x * x;   }

    template <typename T>
    std::size_t to_size_t(T x)
    {   return static_cast<std::size_t>(x);   }

#endif