/*
    * interpolation.hpp
    *
    * Code to interpolate a field at a given position given the 
    * field values at the grid points
    * Interpolation supports a uniform grid with cartesian coordiantes
*/
#ifndef INTERPOLATION_HPP_
    #define INTERPOLATION_HPP_

    #include "grid.hpp"
    #include "indices.hpp"

    namespace py                = pybind11;

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

    template <typename T>
    T ComputeDistancesLogSpherical(T lnr1,
                                    T theta1,
                                    T phi1,
                                    T lnr2,
                                    T theta2,
                                    T phi2)
    {
        return sqrt(exp(2*lnr1) + exp(2*lnr2) - 
                    2*exp(lnr1)*exp(lnr2)*(sin(theta1)*sin(theta2)*cos(phi1-phi2)
                    + cos(theta1)*cos(theta2)));
    }

    /*
        Function to interpolate the field at a given point
        using the field values at the grid points
        Function call overloaded with should_terminate flag
    */
    template <typename T>
    py::array_t<T> InterpolateField(  py::array_t<T> &points,
                            py::array_t<T> &field,
                            std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> &grid,
                            py::array_t<std::size_t> &indices,
                            std::vector<bool> &should_terminate,
                            std::string grid_coord_system)
    {
        auto pointsRef              = points.template unchecked<2>();
        const auto Npoints          = points.shape(1);
        auto gridx1                 = std::get<0>(grid);
        auto gridx2                 = std::get<1>(grid);
        auto gridx3                 = std::get<2>(grid);
        auto gridx1Ref              = gridx1.template unchecked<1>();
        auto gridx2Ref              = gridx2.template unchecked<1>();
        auto gridx3Ref              = gridx3.template unchecked<1>();
        auto indicesRef             = indices.template mutable_unchecked<2>();
        auto fieldRef               = field.template unchecked<4>();
        auto interpolated_field     = py::array_t<T>({3, static_cast<int>(Npoints)});
        auto interpolated_fieldRef  = interpolated_field.template mutable_unchecked<2>();

        // Function to compute distance between two points
        std::function<T(T, T, T, T, T, T)> ComputeDistance;
        if (grid_coord_system == "cartesian")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceCartesian); }
        else if (grid_coord_system == "spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceSpherical); }
        else if (grid_coord_system == "log_spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistancesLogSpherical); }

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   continue;   }

            // Compute the distance between the point and the grid points
            auto d111 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)), gridx2Ref(indicesRef(1, i)), 
                                        gridx3Ref(indicesRef(2, i)));
            auto d112 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)), gridx2Ref(indicesRef(1, i)), 
                                        gridx3Ref(indicesRef(2, i)+1));
            auto d121 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)), gridx2Ref(indicesRef(1, i)+1), 
                                        gridx3Ref(indicesRef(2, i)));
            auto d122 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)), gridx2Ref(indicesRef(1, i)+1), 
                                        gridx3Ref(indicesRef(2, i)+1));
            auto d211 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)+1), gridx2Ref(indicesRef(1, i)), 
                                        gridx3Ref(indicesRef(2, i)));
            auto d212 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)+1), gridx2Ref(indicesRef(1, i)), 
                                        gridx3Ref(indicesRef(2, i)+1));
            auto d221 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)+1), gridx2Ref(indicesRef(1, i)+1), 
                                        gridx3Ref(indicesRef(2, i)));
            auto d222 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)+1), gridx2Ref(indicesRef(1, i)+1), 
                                        gridx3Ref(indicesRef(2, i)+1));

            for (auto coord_id=0; coord_id<3; coord_id++)
            {
                auto F111 = fieldRef(coord_id, indicesRef(0, i), indicesRef(1, i), indicesRef(2, i));
                auto F112 = fieldRef(coord_id, indicesRef(0, i), indicesRef(1, i), indicesRef(2, i)+1);
                auto F121 = fieldRef(coord_id, indicesRef(0, i), indicesRef(1, i)+1, indicesRef(2, i));
                auto F122 = fieldRef(coord_id, indicesRef(0, i), indicesRef(1, i)+1, indicesRef(2, i)+1);
                auto F211 = fieldRef(coord_id, indicesRef(0, i)+1, indicesRef(1, i), indicesRef(2, i));
                auto F212 = fieldRef(coord_id, indicesRef(0, i)+1, indicesRef(1, i), indicesRef(2, i)+1);
                auto F221 = fieldRef(coord_id, indicesRef(0, i)+1, indicesRef(1, i)+1, indicesRef(2, i));
                auto F222 = fieldRef(coord_id, indicesRef(0, i)+1, indicesRef(1, i)+1, indicesRef(2, i)+1);
                // Interpolated field weighted by inverse distances
                interpolated_fieldRef(coord_id, i) =  (F111/d111 + F112/d112 + F121/d121 + F122/d122 + F211/d211 + F212/d212 + F221/d221 +
                                                    F222/d222) / (1/d111 + 1/d112 + 1/d121 + 1/d122 + 1/d211 + 1/d212 + 1/d221 + 1/d222);
            }
        }
        return interpolated_field;
    }

    template <typename T>
    py::array_t<T> InterpolateField(  py::array_t<T> &points,
                            py::array_t<T> &field,
                            std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> &grid,
                            py::array_t<std::size_t> &indices,
                            std::string grid_coord_system)
    {
        auto pointsRef              = points.template unchecked<2>();
        const auto Npoints          = points.shape(1);
        auto gridx1                 = std::get<0>(grid);
        auto gridx2                 = std::get<1>(grid);
        auto gridx3                 = std::get<2>(grid);
        auto gridx1Ref              = gridx1.template unchecked<1>();
        auto gridx2Ref              = gridx2.template unchecked<1>();
        auto gridx3Ref              = gridx3.template unchecked<1>();
        auto indicesRef             = indices.template mutable_unchecked<2>();
        auto fieldRef               = field.template unchecked<4>();
        auto interpolated_field     = py::array_t<T>({3, static_cast<int>(Npoints)});
        auto interpolated_fieldRef  = interpolated_field.template mutable_unchecked<2>();

        // Function to compute distance between two points
        std::function<T(T, T, T, T, T, T)> ComputeDistance;
        if (grid_coord_system == "cartesian")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceCartesian); }
        else if (grid_coord_system == "spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceSpherical); }
        else if (grid_coord_system == "log_spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistancesLogSpherical); }

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            // Compute the distance between the point and the grid points
            auto d111 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)), gridx2Ref(indicesRef(1, i)), 
                                        gridx3Ref(indicesRef(2, i)));
            auto d112 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)), gridx2Ref(indicesRef(1, i)), 
                                        gridx3Ref(indicesRef(2, i)+1));
            auto d121 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)), gridx2Ref(indicesRef(1, i)+1), 
                                        gridx3Ref(indicesRef(2, i)));
            auto d122 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)), gridx2Ref(indicesRef(1, i)+1), 
                                        gridx3Ref(indicesRef(2, i)+1));
            auto d211 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)+1), gridx2Ref(indicesRef(1, i)), 
                                        gridx3Ref(indicesRef(2, i)));
            auto d212 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)+1), gridx2Ref(indicesRef(1, i)), 
                                        gridx3Ref(indicesRef(2, i)+1));
            auto d221 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)+1), gridx2Ref(indicesRef(1, i)+1), 
                                        gridx3Ref(indicesRef(2, i)));
            auto d222 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        gridx1Ref(indicesRef(0, i)+1), gridx2Ref(indicesRef(1, i)+1), 
                                        gridx3Ref(indicesRef(2, i)+1));

            for (auto coord_id=0; coord_id<3; coord_id++)
            {
                auto F111 = fieldRef(coord_id, indicesRef(0, i), indicesRef(1, i), indicesRef(2, i));
                auto F112 = fieldRef(coord_id, indicesRef(0, i), indicesRef(1, i), indicesRef(2, i)+1);
                auto F121 = fieldRef(coord_id, indicesRef(0, i), indicesRef(1, i)+1, indicesRef(2, i));
                auto F122 = fieldRef(coord_id, indicesRef(0, i), indicesRef(1, i)+1, indicesRef(2, i)+1);
                auto F211 = fieldRef(coord_id, indicesRef(0, i)+1, indicesRef(1, i), indicesRef(2, i));
                auto F212 = fieldRef(coord_id, indicesRef(0, i)+1, indicesRef(1, i), indicesRef(2, i)+1);
                auto F221 = fieldRef(coord_id, indicesRef(0, i)+1, indicesRef(1, i)+1, indicesRef(2, i));
                auto F222 = fieldRef(coord_id, indicesRef(0, i)+1, indicesRef(1, i)+1, indicesRef(2, i)+1);
                // Interpolated field weighted by inverse distances
                interpolated_fieldRef(coord_id, i) =  (F111/d111 + F112/d112 + F121/d121 + F122/d122 + F211/d211 + F212/d212 + F221/d221 +
                                                    F222/d222) / (1/d111 + 1/d112 + 1/d121 + 1/d122 + 1/d211 + 1/d212 + 1/d221 + 1/d222);
            }
        }
        return interpolated_field;
    }


    template <typename T>
     std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>, 
                                py::array_t<T>, py::array_t<T>, py::array_t<T>, py::array_t<T>> ProjectFieldToCartesian(
                py::array_t<T> &field,
                std::string field_coord_system,
                std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> &grid,
                std::string grid_coord_system,
                Indices &UniqueIndices)
     {
        const int nx        =  std::max(std::get<0>(grid).shape(0), 
                               std::max(std::get<1>(grid).shape(0), 
                               std::get<2>(grid).shape(0)));
        const int ny        = nx;
        const int nz        = nx;

        auto gridx          = py::array_t<T> ({nx});
        auto gridy          = py::array_t<T> ({ny});
        auto gridz          = py::array_t<T> ({nz});
        auto gridxRef       = gridx.template mutable_unchecked<1>();
        auto gridyRef       = gridy.template mutable_unchecked<1>();
        auto gridzRef       = gridz.template mutable_unchecked<1>();

        auto new_field      = py::array_t<T> ({3, nx, ny, nz});
        auto new_fieldRef   = new_field.template mutable_unchecked<4>();
        new_field[py::make_tuple(py::ellipsis())] = 0;

        auto gridx1Ref      = std::get<0>(grid).template unchecked<1>();
        auto gridx2Ref      = std::get<1>(grid).template unchecked<1>();
        auto gridx3Ref      = std::get<2>(grid).template unchecked<1>();
        auto fieldRef       = field.template unchecked<4>();

        T xmin, xmax, ymin, ymax, zmin, zmax;
        if (grid_coord_system == "cartesian")
        {
            gridx = std::get<0>(grid);
            gridy = std::get<1>(grid);
            gridz = std::get<2>(grid);
        }
        else if (grid_coord_system == "spherical")
        {
            xmin = -gridx1Ref(nx-1);
            xmax = gridx1Ref(nx-1);
            ymin = xmin;
            ymax = xmax;
            zmin = xmin;
            zmax = xmax;
        }
        else if (grid_coord_system == "log_spherical")
        {
            xmin = -exp(gridx1Ref(nx-1));
            xmax = exp(gridx1Ref(nx-1));
            ymin = xmin;
            ymax = xmax;
            zmin = xmin;
            zmax = xmax;
        }
        else
        {
            throw std::invalid_argument("Grid coordinate system not supported");
        }
        // Now populate the 1D arrays
        for (int i=0; i<nx; i++)
        {
            gridxRef(i) = xmin + (xmax - xmin) * i / (nx - 1);
            gridyRef(i) = ymin + (ymax - ymin) * i / (ny - 1);
            gridzRef(i) = zmin + (zmax - zmin) * i / (nz - 1);
        }

        // Now go through all the indices and populate the field
        auto all_indices                = UniqueIndices.PopulateUnravelledIndices();
        auto all_indicesRef             = all_indices.template unchecked<2>();

        const int Npoints               = all_indices.shape(1);
        auto old_points                 = py::array_t<T> ({3, Npoints});
        auto old_pointsRef              = old_points.template mutable_unchecked<2>();
        auto new_grid                   = std::make_tuple(gridx, gridy, gridz);
        auto new_indices                = py::array_t<std::size_t> ({3, Npoints});

        auto x_traj                     = py::array_t<T> ({Npoints});
        auto y_traj                     = py::array_t<T> ({Npoints});
        auto z_traj                     = py::array_t<T> ({Npoints});
        auto x_trajRef                  = x_traj.template mutable_unchecked<1>();
        auto y_trajRef                  = y_traj.template mutable_unchecked<1>();
        auto z_trajRef                  = z_traj.template mutable_unchecked<1>();

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (int i=0; i<Npoints; i++)
        {
            if (grid_coord_system == "cartesian")
            {
                old_pointsRef(0, i) = gridx1Ref(all_indicesRef(0,i));
                old_pointsRef(1, i) = gridx2Ref(all_indicesRef(1,i));
                old_pointsRef(2, i) = gridx3Ref(all_indicesRef(2,i));
            }
            else if (grid_coord_system == "spherical")
            {
                old_pointsRef(0, i) = gridx1Ref(all_indicesRef(0,i)) * sin(gridx2Ref(all_indicesRef(1,i))) * cos(gridx3Ref(all_indicesRef(2,i)));
                old_pointsRef(1, i) = gridx1Ref(all_indicesRef(0,i)) * sin(gridx2Ref(all_indicesRef(1,i))) * sin(gridx3Ref(all_indicesRef(2,i)));
                old_pointsRef(2, i) = gridx1Ref(all_indicesRef(0,i)) * cos(gridx2Ref(all_indicesRef(1,i)));
            }
            else if (grid_coord_system == "log_spherical")
            {
                old_pointsRef(0, i) = exp(gridx1Ref(all_indicesRef(0,i))) * sin(gridx2Ref(all_indicesRef(1,i))) * cos(gridx3Ref(all_indicesRef(2,i)));
                old_pointsRef(1, i) = exp(gridx1Ref(all_indicesRef(0,i))) * sin(gridx2Ref(all_indicesRef(1,i))) * sin(gridx3Ref(all_indicesRef(2,i)));
                old_pointsRef(2, i) = exp(gridx1Ref(all_indicesRef(0,i))) * cos(gridx2Ref(all_indicesRef(1,i)));
            }
        }

        // Now we know where the new fields lies in the old grid
        ReturnClosestIndexMonotonic(  old_points, new_indices, new_grid);
        auto new_indicesRef         = new_indices.unchecked<2>();
        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (int i=0; i<Npoints; i++)
        {
            // Get the old field points and coordiantes
            auto Fx1 = fieldRef(0, all_indicesRef(0, i), all_indicesRef(1, i), all_indicesRef(2, i));
            auto Fx2 = fieldRef(1, all_indicesRef(0, i), all_indicesRef(1, i), all_indicesRef(2, i));
            auto Fx3 = fieldRef(2, all_indicesRef(0, i), all_indicesRef(1, i), all_indicesRef(2, i));
            auto x   = gridxRef(new_indicesRef(0, i));
            auto y   = gridyRef(new_indicesRef(1, i));
            auto z   = gridzRef(new_indicesRef(2, i));

            x_trajRef(i) = x;
            y_trajRef(i) = y;
            z_trajRef(i) = z;

            T Fx, Fy, Fz;
            // Convert the field to cartesian coordinates
            if (field_coord_system == "cartesian")
            {
                Fx = Fx1;
                Fy = Fx2;
                Fz = Fx3;
            }
            else if (field_coord_system == "spherical")
            {
                Fx = Fx1 * sin(x) * cos(y) + Fx2 * cos(x) * cos(y) - Fx3 * sin(y);
                Fy = Fx1 * sin(x) * sin(y) + Fx2 * cos(x) * sin(y) + Fx3 * cos(y);
                Fz = Fx1 * cos(x) - Fx2 * sin(x);
            }
            else if (field_coord_system == "log_spherical")
            {
                auto r = exp(x);
                Fx = r*r*Fx1 * sin(x) * cos(y) + r*r*Fx2 * cos(x) * cos(y) - r*Fx3 * sin(y);
                Fy = r*r*Fx1 * sin(x) * sin(y) + r*r*Fx2 * cos(x) * sin(y) + r*Fx3 * cos(y);
                Fz = r*r*Fx1 * cos(x) - r*r*Fx2 * sin(x);
            }
            new_fieldRef(0, new_indicesRef(0, i), new_indicesRef(1, i), new_indicesRef(2, i)) = Fx;
            new_fieldRef(1, new_indicesRef(0, i), new_indicesRef(1, i), new_indicesRef(2, i)) = Fy;
            new_fieldRef(2, new_indicesRef(0, i), new_indicesRef(1, i), new_indicesRef(2, i)) = Fz;
        }

        return std::make_tuple(new_field, gridx, gridy, gridz, x_traj, y_traj, z_traj);
     }


#endif