#ifndef GRID_HPP_
    #define GRID_HPP_

    // streamtracer headers
    #include "global.hpp"

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
        T dx1, dx2, dx3;
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
            this->dx1                   = dx1;
            this->dx2                   = dx2;
            this->dx3                   = dx3;
            this->x1min                 = gridx1Ref(0);
            this->x1max                 = gridx1Ref(nx1-1);
            this->x2min                 = gridx2Ref(0);
            this->x2max                 = gridx2Ref(nx2-1);
            this->x3min                 = gridx3Ref(0);
            this->x3max                 = gridx3Ref(nx3-1);
            if (grid_coord_system != "cartesian" && grid_coord_system != "spherical" && 
                grid_coord_system != "log_spherical")
            {   throw std::invalid_argument(
                "Invalid grid coordinate system. Must be 'cartesian', 'spherical' or 'log_spherical'");}
        }

        // Return the closest index of a single point (bool -> should terminate)
        bool ReturnClosestIndexUniformGrid(    T pointx1, T pointx2, T pointx3, 
                                    std::size_t &index_x1, std::size_t &index_x2,
                                    std::size_t &index_x3)
        {
            bool should_terminate = false;
            index_x1 = static_cast<std::size_t>((pointx1 - this->x1min) / this->dx1);
            index_x2 = static_cast<std::size_t>((pointx2 - this->x2min) / this->dx2);
            index_x3 = static_cast<std::size_t>((pointx3 - this->x3min) / this->dx3);
            index_x1 = std::max(index_x1, static_cast<std::size_t>(0));
            index_x2 = std::max(index_x2, static_cast<std::size_t>(0));
            index_x3 = std::max(index_x3, static_cast<std::size_t>(0));
            index_x1 = std::min(index_x1, static_cast<std::size_t>(this->nx1 - 2));
            index_x2 = std::min(index_x2, static_cast<std::size_t>(this->nx2 - 2));
            index_x3 = std::min(index_x3, static_cast<std::size_t>(this->nx3 - 2));
            if (grid_coord_system == "cartesian" && (index_x1 >= this->nx1 - 2 ||
            index_x2 >= this->nx2 - 2 ||
            index_x3 >= this->nx3 - 2 ||
            index_x1 == 0 ||
            index_x2 == 0 ||
            index_x3 == 0))
            {   should_terminate = true; }
            else if (index_x1 >= this->nx1 - 2 ||
            index_x2 >= this->nx2 - 2 ||
            index_x1 == 0 ||
            index_x2 == 0)
            {   should_terminate = true; }
            return should_terminate;
        }
    };

#endif