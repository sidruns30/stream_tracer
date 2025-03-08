/*
    * coordinates.hpp
    *
    * Code to convert between different coordinate systems for vectors and positions
*/
#ifndef COORDINATES_HPP_
    #define COORDINATES_HPP_

    #include "global.hpp"
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    namespace py = pybind11;
    

    template <typename T>
    void SphericalToCartesianCoords(py::array_t<T> &points_spherical,
                                    py::array_t<T> &points_cartesian)
    {
        auto points_sphericalRef    = points_spherical.template unchecked<2>();
        auto points_cartesianRef    = points_cartesian.template mutable_unchecked<2>();
        #pragma omp parallel for schedule(static) num_threads(number_of_threads)
        for (std::size_t i=0; i<points_cartesian.shape(1); i++)
        {
            points_cartesianRef(0, i) = points_sphericalRef(0, i) * sin(points_sphericalRef(1, i)) * 
                                        cos(points_sphericalRef(2, i));
            points_cartesianRef(1, i) = points_sphericalRef(0, i) * sin(points_sphericalRef(1, i)) * 
                                        sin(points_sphericalRef(2, i));
            points_cartesianRef(2, i) = points_sphericalRef(0, i) * cos(points_sphericalRef(1, i));
        }
        return;
    }

    template <typename T>
    void CartesianToSphericalCoords(py::array_t<T> &points_cartesian,
                                    py::array_t<T> &points_spherical)
    {
        auto points_cartesianRef    = points_cartesian.template unchecked<2>();
        auto points_sphericalRef    = points_spherical.template mutable_unchecked<2>();
        #pragma omp parallel for schedule(static) num_threads(number_of_threads)
        for (std::size_t i=0; i<points_spherical.shape(1); i++)
        {
            points_sphericalRef(0, i) = sqrt(square(points_cartesianRef(0, i)) +
                                        square(points_cartesianRef(1, i)) +
                                        square(points_cartesianRef(2, i)));
            points_sphericalRef(1, i) = acos(points_cartesianRef(2, i) / points_sphericalRef(0, i));
            points_sphericalRef(2, i) = atan2(points_cartesianRef(1, i), points_cartesianRef(0, i));
        }
        return;
    }

    template <typename T>
    void SphericalToLogSphericalCoords(py::array_t<T> &points_spherical,
                                    py::array_t<T> &points_log_spherical)
    {
        auto points_sphericalRef    = points_spherical.template unchecked<2>();
        auto points_log_sphericalRef    = points_log_spherical.template mutable_unchecked<2>();
        #pragma omp parallel for schedule(static) num_threads(number_of_threads)
        for (std::size_t i=0; i<points_log_spherical.shape(1); i++)
        {
            points_log_sphericalRef(0, i) = log(points_sphericalRef(0, i));
            points_log_sphericalRef(1, i) = points_sphericalRef(1, i);
            points_log_sphericalRef(2, i) = points_sphericalRef(2, i);
        }
        return;
    }

    template <typename T>
    void LogSphericalToSphericalCoords(py::array_t<T> &points_log_spherical,
                                        py::array_t<T> &points_spherical)
    {
        auto points_log_sphericalRef    = points_log_spherical.template unchecked<2>();
        auto points_sphericalRef    = points_spherical.template mutable_unchecked<2>();
        #pragma omp parallel for schedule(static) num_threads(number_of_threads)
        for (std::size_t i=0; i<points_spherical.shape(1); i++)
        {
            points_sphericalRef(0, i) = exp(points_log_sphericalRef(0, i));
            points_sphericalRef(1, i) = points_log_sphericalRef(1, i);
            points_sphericalRef(2, i) = points_log_sphericalRef(2, i);
        }
        return;
    }

    template <typename T>
    void CartesianToLogSphericalCoords( py::array_t<T> &points_cartesian,
                                        py::array_t<T> &points_log_spherical)
    {
        auto points_cartesianRef            = points_cartesian.template unchecked<2>();
        auto points_log_sphericalRef        = points_log_spherical.template mutable_unchecked<2>();
        #pragma omp parallel for schedule(static) num_threads(number_of_threads)
        for (std::size_t i=0; i<points_log_spherical.shape(1); i++)
        {
            auto r                          = sqrt( square(points_cartesianRef(0, i)) + 
                                                square(points_cartesianRef(1, i)) +
                                                square(points_cartesianRef(2, i)));
            points_log_sphericalRef(0, i)   = log(r);
            points_log_sphericalRef(1, i)   = acos(points_cartesianRef(2, i) / r);
            points_log_sphericalRef(2, i)   = atan2(points_cartesianRef(1, i), points_cartesianRef(0, i));
        }
        return;
    }


    template <typename T>
    void LogSphericalToCartesianCoords(py::array_t<T> &points_log_spherical,
                                        py::array_t<T> &points_cartesian)
    {
        auto points_log_sphericalRef    = points_log_spherical.template unchecked<2>();
        auto points_cartesianRef        = points_cartesian.template mutable_unchecked<2>();
        #pragma omp parallel for schedule(static) num_threads(number_of_threads)
        for (std::size_t i=0; i<points_cartesian.shape(1); i++)
        {
            points_cartesianRef(0, i) = exp(points_log_sphericalRef(0, i)) * sin(points_log_sphericalRef(1, i)) * 
                                        cos(points_log_sphericalRef(2, i));
            points_cartesianRef(1, i) = exp(points_log_sphericalRef(0, i)) * sin(points_log_sphericalRef(1, i)) * 
                                        sin(points_log_sphericalRef(2, i));
            points_cartesianRef(2, i) = exp(points_log_sphericalRef(0, i)) * cos(points_log_sphericalRef(1, i));
        }
        return;
    }


    // Convert points from coordinates 'from' to coordianates 'to'
    template <typename T>
    void ConvertCoordiantes(py::array_t<T> &points,
                            std::string from,
                            std::string to,
                            py::array_t<T> &new_points)
    {
        auto pointsRef              = points.template mutable_unchecked<2>();
        const std::size_t Npoints   = points.shape(1);
        if (from == "cartesian" && to == "spherical")
        {   CartesianToSphericalCoords(points, new_points);   }
        else if (from == "spherical" && to == "cartesian")
        {   SphericalToCartesianCoords(points, new_points);   }
        else if (from == "spherical" && to == "log_spherical")
        {   SphericalToLogSphericalCoords(points, new_points);   }
        else if (from == "log_spherical" && to == "spherical")
        {   LogSphericalToSphericalCoords(points, new_points);   }
        else if (from == "cartesian" && to == "log_spherical")
        {   CartesianToLogSphericalCoords(points, new_points);   }
        else if (from == "log_spherical" && to == "cartesian")
        {   LogSphericalToCartesianCoords(points, new_points);   }
        else
        {
            throw std::invalid_argument(
                "Invalid point coordinate system. Must be 'cartesian', 'spherical' or 'log_spherical'");
        }
        return;
    }

#endif /* COORDINATES_HPP_ */