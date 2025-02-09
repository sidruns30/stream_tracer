/*
    * adaptive_stepsize.hpp
    *
    * Estimates the step size for the stream tracer by computing
    * Linv = (F dot grad F) / |F|^2 for each position on the grid, for
    * a given coordinate system. The step size is then computed by
    * finding the nearest value of Linv on the grid to a given point array
*/
#ifndef ADAPTIVE_STEPSIZE_HPP_
    #define ADAPTIVE_STEPSIZE_HPP_

    #include "global.hpp"

    namespace py                = pybind11;

    /*
        Metrics for cartesian, spherical and log-spherical coordinates
        for the gradients
    */
    template <typename T>
    void MetricCartesian(  T x, T y, T z, T &g11, T & g22, T &g33)
    {
        g11 = 1;
        g22 = 1;
        g33 = 1;
        return;
    }

    template <typename T>
    void MetricSpherical(  T r, T theta, T phi, T &g11, T & g22, T &g33)
    {
        g11 = 1;
        g22 = r*r;
        g33 = r*r*sin(theta)*sin(theta);
        return;
    }

    template <typename T>
    void MetricLogSpherical(  T lnr, T theta, T phi, T &g11, T & g22, T &g33)
    {
        g11 = 1;
        g22 = exp(2*lnr);
        g33 = exp(2*lnr)*sin(theta)*sin(theta);
        return;
    }

    /*
        Function to compute the step size for the stream tracer
        Inputs: points, field, grid, grid_coord_system
        Outputs: step_size
        Compute F dot grad F / |F|^2 for each point on the grid
    */
    template <typename T>
    py::array_t<T> compute_adaptive_stepsize(   py::array_t<std::size_t> indices,
                                                py::array_t<T> field,
                                                std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> grid,
                                                std::string grid_coord_system,
                                                std::vector<bool> &should_terminate)
    {
        auto indicesRef             = indices.template unchecked<2>();
        const auto Npoints          = indices.shape(1);
        auto gridx1                 = std::get<0>(grid);
        auto gridx2                 = std::get<1>(grid);
        auto gridx3                 = std::get<2>(grid);
        auto gridx1Ref              = gridx1.template unchecked<1>();
        auto gridx2Ref              = gridx2.template unchecked<1>();
        auto gridx3Ref              = gridx3.template unchecked<1>();
        auto fieldRef               = field.template unchecked<4>();
        auto step_size              = py::array_t<T>({static_cast<int>(Npoints)});
        auto step_sizeRef           = step_size.template mutable_unchecked<1>();
        // create array with the same shape as the field except the first dimension
        auto fieldmag               = py::array_t<T>({static_cast<int>(gridx1Ref.shape(0)), 
                                                    static_cast<int>(gridx2Ref.shape(0)),
                                                    static_cast<int>(gridx3Ref.shape(0))});
        auto fieldmagRef            = fieldmag.template mutable_unchecked<3>();

        auto dx1                   = gridx1Ref(1) - gridx1Ref(0);
        auto dx2                   = gridx2Ref(1) - gridx2Ref(0);
        auto dx3                   = gridx3Ref(1) - gridx3Ref(0);

        // Function to compute the metric tensor
        std::function<void(T, T, T, T&, T&, T&)> Metric;
        if (grid_coord_system == "cartesian")
        {   Metric = static_cast<void(*)(T, T, T, T&, T&, T&)>(MetricCartesian); }
        else if (grid_coord_system == "spherical")
        {   Metric = static_cast<void(*)(T, T, T, T&, T&, T&)>(MetricSpherical); }
        else if (grid_coord_system == "log_spherical")
        {   Metric = static_cast<void(*)(T, T, T, T&, T&, T&)>(MetricLogSpherical); }

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   continue;   }
            fieldmagRef(indicesRef(0, i), indicesRef(1, i), indicesRef(2, i)) = sqrt(square(fieldRef(0, indicesRef(0, i), indicesRef(1, i), indicesRef(2, i))) +
                                                                                square(fieldRef(1, indicesRef(0, i), indicesRef(1, i), indicesRef(2, i))) +
                                                                                square(fieldRef(2, indicesRef(0, i), indicesRef(1, i), indicesRef(2, i))));
        }

        // Now cmopute the gradient of the magnitude of the field
        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   continue;   }

            auto ix1 = indicesRef(0, i);
            auto ix2 = indicesRef(1, i);
            auto ix3 = indicesRef(2, i);

            // If the point is on the boundary
            if (ix1 == 0 || ix1 == gridx1.shape(0) - 1 ||
                ix2 == 0 || ix2 == gridx2.shape(0) - 1 ||
                ix3 == 0 || ix3 == gridx3.shape(0) - 1)
            {
                step_sizeRef(i) = std::min({dx1, dx2, dx3});
                continue;
            }

            // Otherwise project along the gradient of the field
            T g11, g22, g33;
            Metric(gridx1Ref(ix1), gridx2Ref(ix2), gridx3Ref(ix3), g11, g22, g33);

            auto gradmag_x1 = fieldmagRef(ix1 + 1, ix2, ix3) - fieldmagRef(ix1, ix2, ix3);
            auto gradmag_x2 = fieldmagRef(ix1, ix2 + 1, ix3) - fieldmagRef(ix1, ix2, ix3);
            auto gradmag_x3 = fieldmagRef(ix1, ix2, ix3 + 1) - fieldmagRef(ix1, ix2, ix3);
            gradmag_x1      /= (gridx1Ref(ix1 + 1) - gridx1Ref(ix1));
            gradmag_x2      /= (gridx2Ref(ix2 + 1) - gridx2Ref(ix2));
            gradmag_x3      /= (gridx3Ref(ix3 + 1) - gridx3Ref(ix3));

            auto graddotF   =   gradmag_x1 * fieldRef(0, ix1, ix2, ix3) +
                                gradmag_x2 * fieldRef(1, ix1, ix2, ix3) +
                                gradmag_x3 * fieldRef(2, ix1, ix2, ix3);

            step_sizeRef(i) = std::min( static_cast<T>(0.1 * fabs(square(fieldmagRef(ix1 + 1, ix2, ix3)) / graddotF)), 
                                        static_cast<T>(10. * std::min({dx1, dx2, dx3})));
            step_sizeRef(i) = std::max(step_sizeRef(i), static_cast<T>(std::min({dx1, dx2, dx3})));
        }
        return step_size;
    }



#endif /* ADAPTIVE_STEPSIZE_HPP_ */
