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

    //  Function to find distance between two points in 3D space
    template <typename T>
    T ComputeDistanceCartesian(T x1,
                        T y1,
                        T z1,
                        T x2,
                        T y2,
                        T z2)
    {   return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));}

    template <typename T>
    T ComputeDistanceSpherical(T r1,
                        T theta1,
                        T phi1,
                        T r2,
                        T theta2,
                        T phi2)
    {   return sqrt(r1*r1 + r2*r2 - 
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

    // Interpolate field defined at Grid at points
    template <typename T>
    py::array_t<T> InterpolateField(py::array_t<T> &points,
                                    py::array_t<T> &field,
                                    Grid<T> &grid,
                                    py::array_t<std::size_t> &indices,
                                    std::vector<bool> &should_terminate)
    {
        auto pointsRef              = points.template mutable_unchecked<2>();
        const auto Npoints          = points.shape(1);
        auto indicesRef             = indices.template mutable_unchecked<2>();
        auto fieldRef               = field.template unchecked<4>();
        auto interpolated_field     = py::array_t<T>({3, static_cast<int>(Npoints)});
        auto interpolated_fieldRef  = interpolated_field.template mutable_unchecked<2>();

        // Function to compute distance between two points
        std::function<T(T, T, T, T, T, T)> ComputeDistance;
        if (grid.grid_coord_system == "cartesian")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceCartesian); }
        else if (grid.grid_coord_system == "spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistanceSpherical); }
        else if (grid.grid_coord_system == "log_spherical")
        {   ComputeDistance = static_cast<T(*)(T, T, T, T, T, T)>(ComputeDistancesLogSpherical); }

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   continue;   }

            // Compute the distance between the point and the grid points
            T d111 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(indicesRef(0, i)), grid.gridx2Ref(indicesRef(1, i)), 
                                        grid.gridx3Ref(indicesRef(2, i)));
            T d112 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(indicesRef(0, i)), grid.gridx2Ref(indicesRef(1, i)), 
                                        grid.gridx3Ref(indicesRef(2, i)+1));
            T d121 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(indicesRef(0, i)), grid.gridx2Ref(indicesRef(1, i)+1), 
                                        grid.gridx3Ref(indicesRef(2, i)));
            T d122 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(indicesRef(0, i)), grid.gridx2Ref(indicesRef(1, i)+1), 
                                        grid.gridx3Ref(indicesRef(2, i)+1));
            T d211 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(indicesRef(0, i)+1), grid.gridx2Ref(indicesRef(1, i)), 
                                        grid.gridx3Ref(indicesRef(2, i)));
            T d212 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(indicesRef(0, i)+1), grid.gridx2Ref(indicesRef(1, i)), 
                                        grid.gridx3Ref(indicesRef(2, i)+1));
            T d221 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(indicesRef(0, i)+1), grid.gridx2Ref(indicesRef(1, i)+1), 
                                        grid.gridx3Ref(indicesRef(2, i)));
            T d222 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(indicesRef(0, i)+1), grid.gridx2Ref(indicesRef(1, i)+1), 
                                        grid.gridx3Ref(indicesRef(2, i)+1));

            //if (d111 == 0) {throw std::runtime_error("Distance is zero");}
            if (d111 == 0 || std::isnan(d111)) {   d111 = softening_length; }
            if (d112 == 0 || std::isnan(d112)) {   d112 = softening_length; }
            if (d121 == 0 || std::isnan(d121)) {   d121 = softening_length; }
            if (d122 == 0 || std::isnan(d122)) {   d122 = softening_length; }
            if (d211 == 0 || std::isnan(d211)) {   d211 = softening_length; }
            if (d212 == 0 || std::isnan(d212)) {   d212 = softening_length; }
            if (d221 == 0 || std::isnan(d221)) {   d221 = softening_length; }
            if (d222 == 0 || std::isnan(d222)) {   d222 = softening_length; }

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
                                                    F222/d222) / (1/d111 + 1/d112 + 1/d121 + 1/d122 + 1/d211 + 1/d212 + 1/d221 + 1/d222 );
                // If nan then print the field values
                if (std::isnan(interpolated_fieldRef(coord_id, i)))
                {
                    std::cout << "Nan encountered in the field" << std::endl;
                    std::cout << "F111: " << F111 << " F112: " << F112 << " F121: " << F121 << " F122: " << F122 << std::endl;
                    std::cout << "F211: " << F211 << " F212: " << F212 << " F221: " << F221 << " F222: " << F222 << std::endl;
                    std::cout << "d111: " << d111 << " d112: " << d112 << " d121: " << d121 << " d122: " << d122 << std::endl;
                    std::cout << "d211: " << d211 << " d212: " << d212 << " d221: " << d221 << " d222: " << d222 << std::endl;
                    throw std::runtime_error("Nan encountered in the field");
                }
            }
        }
        return interpolated_field;
    }



#endif