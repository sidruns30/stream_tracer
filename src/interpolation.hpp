/*
    * interpolation.hpp
    *
    * Code to interpolate a field at a given position given the 
    * field values at the grid points
    * Interpolation supports a uniform grid with cartesian Coordinates
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
                                    const py::array_t<T> &field,
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

        const auto nx1 = field.shape(0);
        const auto nx2 = field.shape(1);
        const auto nx3 = field.shape(2);

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   continue;   }
            
            // Store the indices of the grid points
            auto ix1_low = indicesRef(0, i);
            auto ix2_low = indicesRef(1, i);
            auto ix3_low = indicesRef(2, i);
            auto ix1_high = ix1_low + 1;
            auto ix2_high = ix2_low + 1;
            auto ix3_high = ix3_low + 1;

            if (grid.grid_coord_system != "cartesian" && ix3_low == grid.nx3 - 1)
            {   ix3_high = 0;}

            // Compute the distance between the point and the grid points
            T d111 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(ix1_low), grid.gridx2Ref(ix2_low), 
                                        grid.gridx3Ref(ix3_low));
            T d112 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(ix1_low), grid.gridx2Ref(ix2_low), 
                                        grid.gridx3Ref(ix3_high));
            T d121 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(ix1_low), grid.gridx2Ref(ix2_high), 
                                        grid.gridx3Ref(ix3_low));
            T d122 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(ix1_low), grid.gridx2Ref(ix2_high), 
                                        grid.gridx3Ref(ix3_high));
            T d211 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(ix1_high), grid.gridx2Ref(ix2_low), 
                                        grid.gridx3Ref(ix3_low));
            T d212 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(ix1_high), grid.gridx2Ref(ix2_low), 
                                        grid.gridx3Ref(ix3_high));
            T d221 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(ix1_high), grid.gridx2Ref(ix2_high), 
                                        grid.gridx3Ref(ix3_low));
            T d222 = ComputeDistance(pointsRef(0, i), pointsRef(1, i), pointsRef(2, i),
                                        grid.gridx1Ref(ix1_high), grid.gridx2Ref(ix2_high), 
                                        grid.gridx3Ref(ix3_high));

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
                auto F111 = fieldRef(coord_id, ix1_low, ix2_low, ix3_low);
                auto F112 = fieldRef(coord_id, ix1_low, ix2_low, ix3_high);
                auto F121 = fieldRef(coord_id, ix1_low, ix2_high, ix3_low);
                auto F122 = fieldRef(coord_id, ix1_low, ix2_high, ix3_high);
                auto F211 = fieldRef(coord_id, ix1_high, ix2_low, ix3_low);
                auto F212 = fieldRef(coord_id, ix1_high, ix2_low, ix3_high);
                auto F221 = fieldRef(coord_id, ix1_high, ix2_high, ix3_low);
                auto F222 = fieldRef(coord_id, ix1_high, ix2_high, ix3_high);
                // Interpolated field weighted by inverse distances
                interpolated_fieldRef(coord_id, i) =  (F111/d111 + F112/d112 + F121/d121 + F122/d122 + F211/d211 + F212/d212 + F221/d221 +
                                                    F222/d222) / (1/d111 + 1/d112 + 1/d121 + 1/d122 + 1/d211 + 1/d212 + 1/d221 + 1/d222 );
                // If nan then print the field values
                if (std::isnan(interpolated_fieldRef(coord_id, i)))
                {
                    std::cout << "Nan encountered in the field" << std::endl;
                    std::cout << "i: " << i << std::endl;
                    std::cout << "ix1_low: " << ix1_low << " ix2_low: " << ix2_low << " ix3_low: " << ix3_low << std::endl;
                    std::cout << "ix1_high: " << ix1_high << " ix2_high: " << ix2_high << " ix3_high: " << ix3_high << std::endl;
                    std::cout << "x1: " << pointsRef(0, i) << " x2: " << pointsRef(1, i) << " x3: " << pointsRef(2, i) << std::endl;
                    std::cout << "grid x1: " << grid.gridx1Ref(ix1_low) << " grid x2: " << grid.gridx2Ref(ix2_low) << " grid x3: " << grid.gridx3Ref(ix3_low) << std::endl;
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



    // Interpolate field at a given point
    template <typename T>
    void InterpolateFieldAtPoint(  const py::array_t<T> &field,
                                const Grid<T> &grid,
                                std::function<T(T, T, T, T, T, T)> ComputeDistance,
                                T x1, T x2, T x3,
                                const std::size_t ix1, const std::size_t ix2, 
                                const std::size_t ix3, T &interp_fieldx,
                                T &interp_fieldy, T &interp_fieldz)
    {
        auto fieldRef = field.template unchecked<4>();
        auto gridx1Ref = grid.gridx1Ref;
        auto gridx2Ref = grid.gridx2Ref;
        auto gridx3Ref = grid.gridx3Ref;

        std::size_t ix1_high = ix1 + 1;
        std::size_t ix2_high = ix2 + 1;
        std::size_t ix3_high = ix3 + 1;

        if (grid.grid_coord_system != "cartesian" && ix3 == grid.nx3 - 1)
        {   ix3_high = 0;}


        // Compute the distance between the point and the grid points
        T d111 = ComputeDistance(x1, x2, x3,
                                gridx1Ref(ix1), gridx2Ref(ix2), 
                                gridx3Ref(ix3));
        T d112 = ComputeDistance(x1, x2, x3,
                                gridx1Ref(ix1), gridx2Ref(ix2), 
                                gridx3Ref(ix3_high));
        T d121 = ComputeDistance(x1, x2, x3,
                                gridx1Ref(ix1), gridx2Ref(ix2_high), 
                                gridx3Ref(ix3));
        T d122 = ComputeDistance(x1, x2, x3,
                                gridx1Ref(ix1), gridx2Ref(ix2_high), 
                                gridx3Ref(ix3_high));
        T d211 = ComputeDistance(x1, x2, x3,
                                gridx1Ref(ix1_high), gridx2Ref(ix2), 
                                gridx3Ref(ix3));
        T d212 = ComputeDistance(x1, x2, x3,
                                gridx1Ref(ix1_high), gridx2Ref(ix2), 
                                gridx3Ref(ix3_high));
        T d221 = ComputeDistance(x1, x2, x3,
                                gridx1Ref(ix1_high), gridx2Ref(ix2_high), 
                                gridx3Ref(ix3));
        T d222 = ComputeDistance(x1, x2, x3,
                                gridx1Ref(ix1_high), gridx2Ref(ix2_high), 
                                gridx3Ref(ix3_high));
        if (d111 == 0 || std::isnan(d111)) {   d111 = softening_length; }
        if (d112 == 0 || std::isnan(d112)) {   d112 = softening_length; }
        if (d121 == 0 || std::isnan(d121)) {   d121 = softening_length; }
        if (d122 == 0 || std::isnan(d122)) {   d122 = softening_length; }
        if (d211 == 0 || std::isnan(d211)) {   d211 = softening_length; }
        if (d212 == 0 || std::isnan(d212)) {   d212 = softening_length; }
        if (d221 == 0 || std::isnan(d221)) {   d221 = softening_length; }
        if (d222 == 0 || std::isnan(d222)) {   d222 = softening_length; }
        // Interpolate the field at the point
        interp_fieldx =  (fieldRef(0, ix1, ix2, ix3)/d111 + fieldRef(0, ix1, ix2, ix3_high)/d112 + 
                                fieldRef(0, ix1, ix2_high, ix3)/d121 + fieldRef(0, ix1, ix2_high, ix3_high)/d122 +
                                fieldRef(0, ix1_high, ix2, ix3)/d211 + fieldRef(0, ix1_high, ix2, ix3_high)/d212 +
                                fieldRef(0, ix1_high, ix2_high, ix3)/d221 + fieldRef(0, ix1_high, ix2_high, ix3_high)/d222) / 
                                (1/d111 + 1/d112 + 1/d121 + 1/d122 + 1/d211 + 1/d212 + 1/d221 + 1/d222 );
        interp_fieldy =  (fieldRef(1, ix1, ix2, ix3)/d111 + fieldRef(1, ix1, ix2, ix3_high)/d112 +
                                fieldRef(1, ix1, ix2_high, ix3)/d121 + fieldRef(1, ix1, ix2_high, ix3_high)/d122 +
                                fieldRef(1, ix1_high, ix2, ix3)/d211 + fieldRef(1, ix1_high, ix2, ix3_high)/d212 +
                                fieldRef(1, ix1_high, ix2_high, ix3)/d221 + fieldRef(1, ix1_high, ix2_high, ix3_high)/d222) / 
                                (1/d111 + 1/d112 + 1/d121 + 1/d122 + 1/d211 + 1/d212 + 1/d221 + 1/d222 );
        interp_fieldz =  (fieldRef(2, ix1, ix2, ix3)/d111 + fieldRef(2, ix1, ix2, ix3_high)/d112 +
                                fieldRef(2, ix1, ix2_high, ix3)/d121 + fieldRef(2, ix1, ix2_high, ix3_high)/d122 +
                                fieldRef(2, ix1_high, ix2, ix3)/d211 + fieldRef(2, ix1_high, ix2, ix3_high)/d212 +
                                fieldRef(2, ix1_high, ix2_high, ix3)/d221 + fieldRef(2, ix1_high, ix2_high, ix3_high)/d222) / 
                                (1/d111 + 1/d112 + 1/d121 + 1/d122 + 1/d211 + 1/d212 + 1/d221 + 1/d222 );
        if (std::isnan(interp_fieldx) || std::isnan(interp_fieldy) || std::isnan(interp_fieldz))
        {
            std::cout << "Nan encountered in the field" << std::endl;
            std::cout << "Field shape: " << field.shape(0) << " " << field.shape(1) << " " << field.shape(2) << std::endl;
            std::cout << "ix1: " << ix1 << " ix2: " << ix2 << " ix3: " << ix3 << std::endl;
            std::cout << "x1: " << x1 << " x2: " << x2 << " x3: " << x3 << std::endl;
            std::cout << "ix1 calculation:" <<  static_cast<std::size_t>((x1 - grid.x1min) / grid.dx1);
            std::cout << "grid x1: " << gridx1Ref(ix1) << " grid x2: " << gridx2Ref(ix2) << " grid x3: " << gridx3Ref(ix3) << std::endl;
            std::cout << "F111: " << fieldRef(0, ix1, ix2, ix3) << " F112: " << fieldRef(0, ix1, ix2, ix3_high) 
                      << " F121: " << fieldRef(0, ix1, ix2_high, ix3) << " F122: " << fieldRef(0, ix1, ix2_high, ix3_high) << std::endl;
            std::cout << "F211: " << fieldRef(0, ix1_high, ix2, ix3) << " F212: " << fieldRef(0, ix1_high, ix2, ix3_high) 
                      << " F221: " << fieldRef(0, ix1_high, ix2_high, ix3) << " F222: " << fieldRef(0, ix1_high, ix2_high, ix3_high)  << std::endl;
            std::cout << "d111: " << d111  << " d112: "  << d112  << " d121: "  << d121  << " d122: "  << d122  << std::endl;
            std::cout << "d211: "  << d211  << " d212: "  << d212  << " d221: "  << d221  << " d222: "  << d222  << std::endl;
            throw std::runtime_error("Nan encountered in the field");
        }
        return;
    }

#endif