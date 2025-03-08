#ifndef GRID_HPP_
    #define GRID_HPP_

    // streamtracer headers
    #include "global.hpp"

    // Function to copy a py::array_t into another array
    template <typename T>
    void CopyOneDArray(py::array_t<T> &from, py::array_t<T> &to)
    {
        std::size_t Npoints = from.shape(0);
        #pragma omp parallel for num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        { to[i] = from[i];}
        return;
    }

    // Placeholder for the grid
    template <typename T>
    struct Grid
    {
        py::array_t<T> gridx1;
        py::array_t<T> gridx2;
        py::array_t<T> gridx3;
        decltype(std::declval<py::array_t<T>>().template unchecked<1>()) gridx1Ref;
        decltype(std::declval<py::array_t<T>>().template unchecked<1>()) gridx2Ref;
        decltype(std::declval<py::array_t<T>>().template unchecked<1>()) gridx3Ref;
        T x1min, x1max, x2min, x2max, x3min, x3max;
        std::size_t nx1;
        std::size_t nx2;
        std::size_t nx3;
        std::string grid_coord_system;
        bool isMonotonic;
        bool isUniform;

        // Constructor
        Grid (  py::array_t<T> gridx1, py::array_t<T> gridx2, 
                py::array_t<T> gridx3, std::string grid_coord_system)  : gridx1(gridx1), gridx2(gridx2), gridx3(gridx3), 
                nx1(gridx1.shape(0)), nx2(gridx2.shape(0)), nx3(gridx3.shape(0)),
                grid_coord_system(grid_coord_system),
                gridx1Ref(gridx1.template unchecked<1>()),
                gridx2Ref(gridx2.template unchecked<1>()),
                gridx3Ref(gridx3.template unchecked<1>()) 
        {
            // Check if the grid is uniform and monotonic
            if (nx1 < 2 || nx2 < 2 || nx3 < 2)
            {   throw std::invalid_argument("Each dimension must have at least 2 cells");}
            auto dx1                   = gridx1Ref(1) - gridx1Ref(0);
            auto dx2                   = gridx2Ref(1) - gridx2Ref(0);
            auto dx3                   = gridx3Ref(1) - gridx3Ref(0);
            isMonotonic = true;
            isUniform = true;
            for (std::size_t ix1=0; ix1<nx1-1; ix1++)
            {
                if ((fabs(1 - (gridx1Ref(ix1+1) - gridx1Ref(ix1))/dx1) > relative_precision))
                {   isUniform           = false;    }
                if (gridx1Ref(ix1+1) <= gridx1Ref(ix1))
                {   isMonotonic         = false;    }
            }
            for (std::size_t ix2=0; ix2<nx2-1; ix2++)
            {
                if ((fabs(1 - (gridx2Ref(ix2+1) - gridx2Ref(ix2))/dx2) > relative_precision))
                {   isUniform           = false;    }
                if (gridx2Ref(ix2+1) <= gridx2Ref(ix2))
                {   isMonotonic         = false;    }
            }
            for (std::size_t ix3=0; ix3<nx3-1; ix3++)
            {
                if ((fabs(1 - (gridx3Ref(ix3+1) - gridx3Ref(ix3))/dx3) > relative_precision))
                {   isUniform           = false;    }
                if (gridx3Ref(ix3+1) <= gridx3Ref(ix3))
                {   isMonotonic         = false;    }
            }
        }

        // Get indices of points closest to grid points and flag points on the grid boundary
        void ReturnClosestIndex(    py::array_t<T> &points, 
                                    py::array_t<std::size_t> &indices_of_points,
                                    std::vector<bool> &should_terminate)
        {
            auto indicesRef             = indices_of_points.template mutable_unchecked<2>();
            auto pointsRef              = points.template unchecked<2>();
            const auto Npoints          = points.shape(1);
            if (isUniform)
            {
                const auto dx1              = this->gridx1Ref(1) - this->gridx1Ref(0);
                const auto dx2              = this->gridx2Ref(1) - this->gridx2Ref(0);
                const auto dx3              = this->gridx3Ref(1) - this->gridx3Ref(0);
                #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
                for (std::size_t i=0; i<Npoints; i++)
                {
                    if (should_terminate[i]){   continue;   }
                    indicesRef(0, i) = static_cast<std::size_t>((pointsRef(0, i) - this->gridx1Ref(0)) / dx1);
                    indicesRef(1, i) = static_cast<std::size_t>((pointsRef(1, i) - this->gridx2Ref(0)) / dx2);
                    indicesRef(2, i) = static_cast<std::size_t>((pointsRef(2, i) - this->gridx3Ref(0)) / dx3);
                    indicesRef(0, i) = std::max(indicesRef(0, i), static_cast<std::size_t>(0));
                    indicesRef(1, i) = std::max(indicesRef(1, i), static_cast<std::size_t>(0));
                    indicesRef(2, i) = std::max(indicesRef(2, i), static_cast<std::size_t>(0));
                    indicesRef(0, i) = std::min(indicesRef(0, i), static_cast<std::size_t>(this->nx1 - 1));
                    indicesRef(1, i) = std::min(indicesRef(1, i), static_cast<std::size_t>(this->nx2 - 1));
                    indicesRef(2, i) = std::min(indicesRef(2, i), static_cast<std::size_t>(this->nx3 - 1));
                    if (indicesRef(0, i) == this->nx1 - 1 || 
                    indicesRef(1, i) == this->nx2 - 1 || 
                    indicesRef(2, i) == this->nx3 - 1 ||
                    indicesRef(0, i) == 0 ||
                    indicesRef(1, i) == 0 ||
                    indicesRef(2, i) == 0)
                {   should_terminate[i] = true; }
                }
            }
            else
            {
                #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
                for (std::size_t i=0; i<Npoints; i++)
                {
                    if (should_terminate[i]){   continue;   }
                    std::size_t left = 0, right = gridx1.shape(0) - 1;
                    while (left < right)
                    {
                        std::size_t mid = left + (right - left) / 2;
                        if (gridx1Ref(mid) <= pointsRef(0, i)) {   left = mid + 1; }
                        else {   right = mid;    }
                    }
                    indicesRef(0, i) = left - 1;
                    left = 0;
                    right = gridx2.shape(0) - 1;
                    while (left < right)
                    {
                        std::size_t mid = left + (right - left) / 2;
                        if (gridx2Ref(mid) <= pointsRef(1, i)){   left = mid + 1; }
                        else {   right = mid;    }
                    }
                    indicesRef(1, i) = left - 1;
                    left = 0;
                    right = gridx3.shape(0) - 1;
                    while (left < right)
                    {
                        std::size_t mid = left + (right - left) / 2;
                        if (gridx3Ref(mid) <= pointsRef(2, i)) {   left = mid + 1; }
                        else {   right = mid;    }
                    }
                    indicesRef(2, i) = left - 1;
                    // Check if the point is on the boundary
                    if (indicesRef(0, i) == this->nx1 - 1 || 
                    indicesRef(1, i) == this->nx2 - 1 || 
                    indicesRef(2, i) == this->nx3 - 1 ||
                    indicesRef(0, i) == 0 ||
                    indicesRef(1, i) == 0 ||
                    indicesRef(2, i) == 0)
                {   should_terminate[i] = true; }
                }
            }
            return;
        }

        // Destructor
        ~Grid() {}
    };

#endif