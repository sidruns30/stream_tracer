#ifndef _INTEGRATION_HPP_
#define _INTEGRATION_HPP_

#include "global.hpp"
#include "grid.hpp"
#include "interpolation.hpp"
#include "display.hpp"
#include "coordinates.hpp"
#include "payloads.hpp"

namespace Integrators
{
    template <typename T>
    float TakeStepEuler( py::array_t<T> &points,
                        const py::array_t<T> &field,
                        Grid<T> &Grid,
                        std::vector<bool> &should_terminate,
                        py::array_t<T> &current_quantity_values,
                        const std::vector<std::string> &payload_names,
                        const std::vector<py::array_t<T>> &payload_arrays,
                        Timers &timers)
    {
        auto pointsRef                  = points.template mutable_unchecked<2>();
        auto current_quantity_valuesRef = current_quantity_values.mutable_unchecked();
        const auto Npoints              = points.shape(1);
        std::size_t count               = 0;

        // Distance function based on the grid coordinate system
        std::function<T(T, T, T, T, T, T)> ComputeDistance;
        if (Grid.grid_coord_system == "cartesian")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceCartesian); }
        else if (Grid.grid_coord_system == "spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceSpherical); }
        else if (Grid.grid_coord_system == "log_spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistancesLogSpherical); }

        // Similarly generate the coordinate transformation functions
        std::function<void(const T&, const T&, const T&, T&, T&, T&)> TransformToCartesian;
        if (Grid.grid_coord_system == "cartesian")
        {  TransformToCartesian = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::CopyPoint); }
        else if (Grid.grid_coord_system == "spherical")
        {   TransformToCartesian = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::SphericalToCartesianPoint); }
        else if (Grid.grid_coord_system == "log_spherical")
        {   TransformToCartesian = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::LogSphericalToCartesianPoint); }

        std::function<void(const T&, const T&, const T&, T&, T&, T&)> TransformToGrid;
        if (Grid.grid_coord_system == "cartesian")
        {   TransformToGrid = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::CopyPoint); }
        else if (Grid.grid_coord_system == "spherical")
        {   TransformToGrid = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::CartesianToSphericalPoint); }
        else if (Grid.grid_coord_system == "log_spherical")
        {   TransformToGrid = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::CartesianToLogSphericalPoint); }

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads) reduction(+:count)
        for (std::size_t streamline_id=0; streamline_id < Npoints; streamline_id++)
        {
            if (should_terminate[streamline_id]){   count++;  continue;   }
            // Compute the index of the closet grid point
            std::size_t ix1, ix2, ix3;
            should_terminate[streamline_id] = Grid.ReturnClosestIndexUniformGrid( pointsRef(0, streamline_id),
            pointsRef(1, streamline_id),
            pointsRef(2, streamline_id),
            ix1,
            ix2,
            ix3);
            
            // Interpolate the field at the point
            T fieldx, fieldy, fieldz;
            InterpolateFieldAtPoint( field,
                Grid,
                ComputeDistance,
                pointsRef(0, streamline_id),
                pointsRef(1, streamline_id),
                pointsRef(2, streamline_id),
                ix1, ix2, ix3, fieldx, fieldy, fieldz);
                
            auto norm                   = sqrt(square(fieldx) +
                                            square(fieldy) +
                                            square(fieldz));
            T stepsize = Grid.dx1;
            if (Grid.grid_coord_system == "log_spherical")
            {  stepsize *= exp(Grid.gridx1Ref(ix1)); }
                
            T point_cartx, point_carty, point_cartz;
            TransformToCartesian( pointsRef(0, streamline_id),
                                    pointsRef(1, streamline_id),
                                    pointsRef(2, streamline_id),
                                    point_cartx,
                                    point_carty,
                                    point_cartz);
            point_cartx += fieldx * stepsize / norm;
            point_carty += fieldy * stepsize / norm;
            point_cartz += fieldz * stepsize / norm;
            if (sqrt(square(point_cartx) + 
                    square(point_carty) + 
                    square(point_cartz)) <= inner_termination_radius)
            {   should_terminate[streamline_id] = true;  }
            
            TransformToGrid( point_cartx,
                                point_carty,
                                point_cartz,
                                pointsRef(0, streamline_id),
                                pointsRef(1, streamline_id),
                                pointsRef(2, streamline_id));
            current_quantity_valuesRef(streamline_id) = norm;
        }
        return static_cast<float>(count) / Npoints;
    }

    template <typename T>
    float TakeStepRK4(  py::array_t<T> &points,
                        const py::array_t<T> &field,
                        Grid<T> &Grid,
                        std::vector<bool> &should_terminate,
                        py::array_t<T> &current_quantity_values,
                        const std::vector<std::string> &payload_names,
                        const std::vector<py::array_t<T>> &payload_arrays,
                        Timers &timers)
    {
        auto pointsRef                  = points.template mutable_unchecked<2>();
        auto current_quantity_valuesRef = current_quantity_values.mutable_unchecked();
        const auto Npoints              = points.shape(1);
        std::size_t count               = 0;

        // Distance function based on the grid coordinate system
        std::function<T(T, T, T, T, T, T)> ComputeDistance;
        if (Grid.grid_coord_system == "cartesian")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceCartesian); }
        else if (Grid.grid_coord_system == "spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceSpherical); }
        else if (Grid.grid_coord_system == "log_spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistancesLogSpherical); }

        // Similarly generate the coordinate transformation functions
        std::function<void(const T&, const T&, const T&, T&, T&, T&)> TransformToCartesian;
        if (Grid.grid_coord_system == "cartesian")
        {  TransformToCartesian = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::CopyPoint); }
        else if (Grid.grid_coord_system == "spherical")
        {   TransformToCartesian = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::SphericalToCartesianPoint); }
        else if (Grid.grid_coord_system == "log_spherical")
        {   TransformToCartesian = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::LogSphericalToCartesianPoint); }

        std::function<void(const T&, const T&, const T&, T&, T&, T&)> TransformToGrid;
        if (Grid.grid_coord_system == "cartesian")
        {   TransformToGrid = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::CopyPoint); }
        else if (Grid.grid_coord_system == "spherical")
        {   TransformToGrid = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::CartesianToSphericalPoint); }
        else if (Grid.grid_coord_system == "log_spherical")
        {   TransformToGrid = static_cast<void(*)(const T&, const T&, const T&, T&, T&, T&)>(Coordinates::CartesianToLogSphericalPoint); }

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads) reduction(+:count)
        for (std::size_t streamline_id=0; streamline_id < Npoints; streamline_id++)
        {
            if (should_terminate[streamline_id]){   count++;  continue;   }
            T stepsize = Grid.dx1;
            // Store cartesian vectors here for RK4
            T r1[3], r2[3], r3[3], r4[3];
            T k1[3], k2[3], k3[3], k4[3];
            T r1_grid[3], r2_grid[3], r3_grid[3], r4_grid[3];

            // STEP 1
            T point_cartx, point_carty, point_cartz;
            TransformToCartesian( pointsRef(0, streamline_id),
                                    pointsRef(1, streamline_id),
                                    pointsRef(2, streamline_id),
                                    point_cartx,
                                    point_carty,
                                    point_cartz);
            r1[0] = point_cartx; r1_grid[0] = pointsRef(0, streamline_id);
            r1[1] = point_carty; r1_grid[1] = pointsRef(1, streamline_id);
            r1[2] = point_cartz; r1_grid[2] = pointsRef(2, streamline_id);
            std::size_t ix1, ix2, ix3;
            should_terminate[streamline_id] = (should_terminate[streamline_id] || 
                                                Grid.ReturnClosestIndexUniformGrid( r1_grid[0],
                                                r1_grid[1],
                                                r1_grid[2],
                                                ix1,
                                                ix2,
                                                ix3));
            if (Grid.grid_coord_system == "log_spherical")
            {  stepsize *= exp(Grid.gridx1Ref(ix1)); }
            // Interpolate the field at the point
            T fieldx, fieldy, fieldz;
            InterpolateFieldAtPoint( field,
                                    Grid,
                                    ComputeDistance,
                                    r1_grid[0],
                                    r1_grid[1],
                                    r1_grid[2],
                                    ix1, ix2, ix3, fieldx, fieldy, fieldz);
            auto norm                   = sqrt(square(fieldx) +
                                            square(fieldy) +
                                            square(fieldz));
            k1[0] = fieldx * stepsize / norm;
            k1[1] = fieldy * stepsize / norm;
            k1[2] = fieldz * stepsize / norm;
            // STEP 2
            r2[0] = r1[0] + k1[0] / 2.0;
            r2[1] = r1[1] + k1[1] / 2.0;
            r2[2] = r1[2] + k1[2] / 2.0;
            TransformToGrid( r2[0],
                                r2[1],
                                r2[2],
                                r2_grid[0],
                                r2_grid[1],
                                r2_grid[2]);
            should_terminate[streamline_id] = (should_terminate[streamline_id] || 
                                                Grid.ReturnClosestIndexUniformGrid( r2_grid[0],
                                                r2_grid[1],
                                                r2_grid[2],
                                                ix1,
                                                ix2,
                                                ix3));
            InterpolateFieldAtPoint( field,
                                    Grid,
                                    ComputeDistance,
                                    r2_grid[0],
                                    r2_grid[1],
                                    r2_grid[2],
                                    ix1, ix2, ix3, fieldx, fieldy, fieldz);
            norm                   = sqrt(square(fieldx) +
                                            square(fieldy) +
                                            square(fieldz));
            k2[0] = fieldx * stepsize / norm;
            k2[1] = fieldy * stepsize / norm;
            k2[2] = fieldz * stepsize / norm;
            // STEP 3
            r3[0] = r1[0] + k2[0] / 2.0;
            r3[1] = r1[1] + k2[1] / 2.0;
            r3[2] = r1[2] + k2[2] / 2.0;
            TransformToGrid( r3[0],
                                r3[1],
                                r3[2],
                                r3_grid[0],
                                r3_grid[1],
                                r3_grid[2]);
            should_terminate[streamline_id] = (should_terminate[streamline_id] || 
                                                Grid.ReturnClosestIndexUniformGrid( r3_grid[0],
                                                r3_grid[1],
                                                r3_grid[2],
                                                ix1,
                                                ix2,
                                                ix3));
            InterpolateFieldAtPoint( field,
                                    Grid,
                                    ComputeDistance,
                                    r3_grid[0],
                                    r3_grid[1],
                                    r3_grid[2],
                                    ix1, ix2, ix3, fieldx, fieldy, fieldz);
            norm                   = sqrt(square(fieldx) +
                                            square(fieldy) +
                                            square(fieldz));
            k3[0] = fieldx * stepsize / norm;
            k3[1] = fieldy * stepsize / norm;
            k3[2] = fieldz * stepsize / norm;
            // STEP 4
            r4[0] = r1[0] + k3[0];
            r4[1] = r1[1] + k3[1];
            r4[2] = r1[2] + k3[2];
            TransformToGrid( r4[0],
                            r4[1],
                            r4[2],
                            r4_grid[0],
                            r4_grid[1],
                            r4_grid[2]);
            should_terminate[streamline_id] = (should_terminate[streamline_id] || 
                                                Grid.ReturnClosestIndexUniformGrid( r4_grid[0],
                                                r4_grid[1],
                                                r4_grid[2],
                                                ix1,
                                                ix2,
                                                ix3));
            InterpolateFieldAtPoint( field,
                                    Grid,
                                    ComputeDistance,
                                    r4_grid[0],
                                    r4_grid[1],
                                    r4_grid[2],
                                    ix1, ix2, ix3, fieldx, fieldy, fieldz);
            norm                   = sqrt(square(fieldx) +
                                            square(fieldy) +
                                            square(fieldz));
            k4[0] = fieldx * stepsize / norm;
            k4[1] = fieldy * stepsize / norm;
            k4[2] = fieldz * stepsize / norm;
            // Update the current positions
            for (int i=0; i<3; i++) 
            {   r1[i] +=  (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;    }

            if (sqrt(square(r1[0]) + square(r1[1]) +    square(r1[2])) <= inner_termination_radius)
            {   should_terminate[streamline_id] = true;  }

            TransformToGrid( r1[0],
                                r1[1],
                                r1[2],
                                pointsRef(0, streamline_id),
                                pointsRef(1, streamline_id),
                                pointsRef(2, streamline_id));
            current_quantity_valuesRef(streamline_id) = norm;
        }
        return static_cast<float>(count) / Npoints;
    }

} // namespace Integrators



#endif /* INTEGRATION_HPP_ */