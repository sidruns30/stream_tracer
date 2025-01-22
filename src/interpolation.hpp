/*
    * interpolation.hpp
    *
    * Code to interpolate a field at a given position given the 
    * field values at the grid points
    * Interpolation supports a uniform grid with cartesian coordiantes
*/
#ifndef INTERPOLATION_HPP_
    #define INTERPOLATION_HPP_

    #include <omp.h>
    #include <vector>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include "grid.hpp"
    #include <functional>

    namespace py                = pybind11;
    namespace interpolate
    {
        /*
            Function to find distance between two points in 3D space
        */
        template <typename T>
        T ComputeDistanceCartesian(T x1,
                           T y1,
                           T z1,
                           T x2,
                           T y2,
                           T z2)
        {
            return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
        }

        template <typename T>
        T ComputeDistanceSpherical(T r1,
                           T theta1,
                           T phi1,
                           T r2,
                           T theta2,
                           T phi2)
        {
            return sqrt(r1*r1 + r2*r2 - 
                        2*r1*r2*(sin(theta1)*sin(theta2)*cos(phi1-phi2)
                        + cos(theta1)*cos(theta2)));
        }

        /*
            Function to interpolate the field at a given point
            using the field values at the grid points
        */
        template <typename T>
        void InterpolateField(py::array_t<T> &position_x,
                              py::array_t<T> &position_y,
                              py::array_t<T> &position_z,
                              std::vector<std::size_t> &indices_x,
                              std::vector<std::size_t> &indices_y,
                              std::vector<std::size_t> &indices_z,
                              py::array_t<T> &grid_x,
                              py::array_t<T> &grid_y,
                              py::array_t<T> &grid_z,
                              py::array_t<T> &field_values,
                              py::array_t<T> &interpolated_values,
                              const std::string &coordinateSystem)
        {
            auto position_x_ref = position_x.template unchecked<1>();
            auto position_y_ref = position_y.template unchecked<1>();
            auto position_z_ref = position_z.template unchecked<1>();
            auto grid_x_ref = grid_x.template unchecked<1>();
            auto grid_y_ref = grid_y.template unchecked<1>();
            auto grid_z_ref = grid_z.template unchecked<1>();
            auto field_values_ref = field_values.template unchecked<3>();
            const std::size_t nx = grid_x_ref.shape(0);
            const std::size_t ny = grid_y_ref.shape(0);
            const std::size_t nz = grid_z_ref.shape(0);
            const std::size_t Npoints = position_x_ref.shape(0);
            auto interpolated_values_ref = interpolated_values.template mutable_unchecked<1>();

            // Point to cartesian or spherical functions
            std::function<T(T, T, T, T, T, T)> ComputeDistance;
            if (coordinateSystem == "cartesian")
            {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceCartesian); }
            if (coordinateSystem == "spherical")
            {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceSpherical); }
            
            #pragma omp parallel for schedule(static)
            for (std::size_t i=0; i<Npoints; i++)
            {
                auto F111 = field_values_ref(indices_x[i], indices_y[i], indices_z[i]);
                auto F112 = field_values_ref(indices_x[i], indices_y[i], indices_z[i]+1);
                auto F121 = field_values_ref(indices_x[i], indices_y[i]+1, indices_z[i]);
                auto F122 = field_values_ref(indices_x[i], indices_y[i]+1, indices_z[i]+1);
                auto F211 = field_values_ref(indices_x[i]+1, indices_y[i], indices_z[i]);
                auto F212 = field_values_ref(indices_x[i]+1, indices_y[i], indices_z[i]+1);
                auto F221 = field_values_ref(indices_x[i]+1, indices_y[i]+1, indices_z[i]);
                auto F222 = field_values_ref(indices_x[i]+1, indices_y[i]+1, indices_z[i]+1);

                auto d111 = ComputeDistance(position_x_ref(i), position_y_ref(i), position_z_ref(i),
                                        grid_x_ref(indices_x[i]), grid_y_ref(indices_y[i]), grid_z_ref(indices_z[i]));
                auto d112 = ComputeDistance(position_x_ref(i), position_y_ref(i), position_z_ref(i),
                                            grid_x_ref(indices_x[i]), grid_y_ref(indices_y[i]), grid_z_ref(indices_z[i]+1));
                auto d121 = ComputeDistance(position_x_ref(i), position_y_ref(i), position_z_ref(i),
                                            grid_x_ref(indices_x[i]), grid_y_ref(indices_y[i]+1), grid_z_ref(indices_z[i]));
                auto d122 = ComputeDistance(position_x_ref(i), position_y_ref(i), position_z_ref(i),
                                            grid_x_ref(indices_x[i]), grid_y_ref(indices_y[i]+1), grid_z_ref(indices_z[i]+1));
                auto d211 = ComputeDistance(position_x_ref(i), position_y_ref(i), position_z_ref(i),
                                            grid_x_ref(indices_x[i]+1), grid_y_ref(indices_y[i]), grid_z_ref(indices_z[i]));
                auto d212 = ComputeDistance(position_x_ref(i), position_y_ref(i), position_z_ref(i),
                                            grid_x_ref(indices_x[i]+1), grid_y_ref(indices_y[i]), grid_z_ref(indices_z[i]+1));
                auto d221 = ComputeDistance(position_x_ref(i), position_y_ref(i), position_z_ref(i),
                                            grid_x_ref(indices_x[i]+1), grid_y_ref(indices_y[i]+1), grid_z_ref(indices_z[i]));
                auto d222 = ComputeDistance(position_x_ref(i), position_y_ref(i), position_z_ref(i),
                                            grid_x_ref(indices_x[i]+1), grid_y_ref(indices_y[i]+1), grid_z_ref(indices_z[i]+1));

                interpolated_values_ref(i) =  (F111/d111 + F112/d112 + F121/d121 + F122/d122 + F211/d211 + F212/d212 + F221/d221 +
                                            F222/d222) / (1/d111 + 1/d112 + 1/d121 + 1/d122 + 1/d211 + 1/d212 + 1/d221 + 1/d222);
            }
            return;
        }
    } // namespace interpolate

#endif