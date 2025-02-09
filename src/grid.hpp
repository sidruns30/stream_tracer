/*
    * grid.hpp                                      
    *
    * Created on: 2024. 12. 27.                     
    * File to load a vector field to a grid,        
    * find indices of coordinate tuple (x1,x2,x3)   
    * The grid must be monotonically increasing     
    * in all coordinates.                           
    *
*/
#ifndef GRID_HPP_
    #define GRID_HPP_

    #include "global.hpp"

    namespace py                = pybind11;

    /*
        Function to check if the grid is monotonic and  
        uniform
        The grid is deemed to be uniform if |1 - dx0 / dxi| < 0.01
        Returns a tuple of two booleans (isMonotonic, isUniform)
    */
    template <typename T>
    std::tuple<bool, bool> CheckIfUniform(  std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> &grid)
    {
        bool isMonotonic            = true;
        bool isUniform              = true;
        auto gridx1                 = std::get<0>(grid);
        auto gridx2                 = std::get<1>(grid);
        auto gridx3                 = std::get<2>(grid);
        auto gridx1Ref              = gridx1.template unchecked<1>();
        auto gridx2Ref              = gridx2.template unchecked<1>();
        auto gridx3Ref              = gridx3.template unchecked<1>();
        const std::size_t nx1       = gridx1.shape(0);
        const std::size_t nx2       = gridx2.shape(0);
        const std::size_t nx3       = gridx3.shape(0);
        if (nx1 < 2 || nx2 < 2 || nx3 < 2)
        {   throw std::invalid_argument("Each dimension must have at least 2 cells");}

        auto dx1                   = gridx1Ref(1) - gridx1Ref(0);
        auto dx2                   = gridx2Ref(1) - gridx2Ref(0);
        auto dx3                   = gridx3Ref(1) - gridx3Ref(0);
        // Check x array
        for (std::size_t ix1=0; ix1<nx1-1; ix1++)
        {
            if ((fabs(1 - (gridx1Ref(ix1+1) - gridx1Ref(ix1))/dx1) > relative_precision))
            {   isUniform           = false;    }
            if (gridx1Ref(ix1+1) <= gridx1Ref(ix1))
            {   isMonotonic         = false;    
            }
        }
        // Check y array
        for (std::size_t ix2=0; ix2<nx2-1; ix2++)
        {
            if ((fabs(1 - (gridx2Ref(ix2+1) - gridx2Ref(ix2))/dx2) > relative_precision))
            {   isUniform           = false;    }
            if (gridx2Ref(ix2+1) <= gridx2Ref(ix2))
            {   isMonotonic         = false;    
            }
        }
        // Check z array
        for (std::size_t ix3=0; ix3<nx3-1; ix3++)
        {
            if ((fabs(1 - (gridx3Ref(ix3+1) - gridx3Ref(ix3))/dx3) > relative_precision))
            {   isUniform           = false;    }
            if (gridx3Ref(ix3+1) <= gridx3Ref(ix3))
            {   isMonotonic         = false;    
            }
        }
        if (isUniform)
        {   std::cout << "Grid is uniform to within 1%" << std::endl;    }
        return std::make_tuple( isMonotonic, isUniform);
    }

    /*
        Out of bounds error
    */
    class OutOfBoundsError : public std::exception
    {
        public:
            OutOfBoundsError(const std::string &message) : message(message) {}
            virtual const char* what() const throw()
            {   return message.c_str(); }
        private:
            std::string message;
    };

    /*
        Ensure that all the points lie within the grid
    */
    template <typename T>
    void CheckBounds(   py::array_t<T> &points,
                        std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> &grid,
                        std::vector <bool> &should_terminate,
                        const bool raise_out_of_bounds_error)
    {
        auto pointsRef              = points.template unchecked<2>();
        const auto Npoints          = points.shape(1);
        auto gridx1                 = std::get<0>(grid);
        auto gridx2                 = std::get<1>(grid);
        auto gridx3                 = std::get<2>(grid);
        auto gridx1Ref              = gridx1.template unchecked<1>();
        auto gridx2Ref              = gridx2.template unchecked<1>();
        auto gridx3Ref              = gridx3.template unchecked<1>();
        #pragma omp parallel for schedule(static) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (pointsRef(0, i) < gridx1Ref(1) || pointsRef(0, i) > gridx1Ref(gridx1.shape(0)-2) ||
                pointsRef(1, i) < gridx2Ref(1) || pointsRef(1, i) > gridx2Ref(gridx2.shape(0)-2) ||
                pointsRef(2, i) < gridx3Ref(1) || pointsRef(2, i) > gridx3Ref(gridx3.shape(0)-2))
            {   
                should_terminate[i] = true; 
                if (raise_out_of_bounds_error)
                {   // Print point coordinates and grid limits
                    std::cout << "Point coordinates: " << pointsRef(0, i) << ", " << pointsRef(1, i) << ", " << pointsRef(2, i) << std::endl;
                    std::cout << "x1 min: " << gridx1Ref(0) << ", x1 max: " << gridx1Ref(gridx1.shape(0)-1) << std::endl;
                    std::cout << "x2 min: " << gridx2Ref(0) << ", x2 max: " << gridx2Ref(gridx2.shape(0)-1) << std::endl;
                    std::cout << "x3 min: " << gridx3Ref(0) << ", x3 max: " << gridx3Ref(gridx3.shape(0)-1) << std::endl;
                    auto message = "Point " + std::to_string(i) + " is out of bounds";
                    throw OutOfBoundsError(message);
                }
            }
        }
        return;
    }

    /*
        Count and print the number of streamlines terminated
    */
   float PercentTerminate(std::vector<bool> &should_terminate)
    {
        const auto Npoints          = should_terminate.size();
        std::size_t count            = 0;
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   count++;    }
        }
        return count / Npoints;
    }


    /*
        Find the nearest cell in the grid that is less than 
        xd[j] for a monotonically increasing 1D grid, for all j's
        <---><---><--->...<--->...<--->
    i =  0     1    2      k        n
                            ^xd[j]
                            -> return k
    */
    template <typename T>
    void ReturnClosestIndexMonotonic(   py::array_t<T> &points, 
                                        py::array_t<std::size_t> &indices_of_points,
                                        std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> &grid)
    {
        auto indicesRef             = indices_of_points.template mutable_unchecked<2>();
        auto pointsRef              = points.template unchecked<2>();
        const auto Npoints          = points.shape(1);
        auto gridx1                 = std::get<0>(grid);
        auto gridx2                 = std::get<1>(grid);
        auto gridx3                 = std::get<2>(grid);
        auto gridx1Ref              = gridx1.template unchecked<1>();
        auto gridx2Ref              = gridx2.template unchecked<1>();
        auto gridx3Ref              = gridx3.template unchecked<1>();

        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            // Start with x1
            std::size_t left = 0;
            std::size_t right = gridx1.shape(0) - 1;
            while (left < right)
            {
                std::size_t mid = left + (right - left) / 2;
                if (gridx1Ref(mid) <= pointsRef(0, i))
                {   left = mid + 1; }
                else
                {   right = mid;    }
            }
            indicesRef(0, i) = left - 1;
            // x2
            left = 0;
            right = gridx2.shape(0) - 1;
            while (left < right)
            {
                std::size_t mid = left + (right - left) / 2;
                if (gridx2Ref(mid) <= pointsRef(1, i))
                {   left = mid + 1; }
                else
                {   right = mid;    }
            }
            indicesRef(1, i) = left - 1;
            // x3
            left = 0;
            right = gridx3.shape(0) - 1;
            while (left < right)
            {
                std::size_t mid = left + (right - left) / 2;
                if (gridx3Ref(mid) <= pointsRef(2, i))
                {   left = mid + 1; }
                else
                {   right = mid;    }
            }
            indicesRef(2, i) = left - 1;
        }
        return;
    }

    /*
        Find the nearest cell in the grid that is greater than or equal
        to ^xk[j] for a uniform 1D grid, for all j's
        <---><---><--->...<--->...<--->
    i =  0     1    2      k        n
                            ^xk[j]
    */
   template <typename T>
   void ReturnClosestIndexUniform(  py::array_t<T> &points, 
                                    py::array_t<std::size_t> &indices_of_points,
                                    std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> &grid)
    {
        auto indicesRef             = indices_of_points.template mutable_unchecked<2>();
        auto pointsRef              = points.template unchecked<2>();
        const auto Npoints          = points.shape(1);
        auto gridx1                 = std::get<0>(grid);
        auto gridx2                 = std::get<1>(grid);
        auto gridx3                 = std::get<2>(grid);
        auto gridx1Ref              = gridx1.template unchecked<1>();
        auto gridx2Ref              = gridx2.template unchecked<1>();
        auto gridx3Ref              = gridx3.template unchecked<1>();
        const auto dx1              = gridx1Ref(1) - gridx1Ref(0);
        const auto dx2              = gridx2Ref(1) - gridx2Ref(0);
        const auto dx3              = gridx3Ref(1) - gridx3Ref(0);
        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            indicesRef(0, i) = std::min( static_cast<std::size_t>((pointsRef(0, i) - gridx1Ref(0)) / dx1),
                                                    static_cast<std::size_t>(gridx1.shape(0) - 2));
            indicesRef(1, i) = std::min( static_cast<std::size_t>((pointsRef(1, i) - gridx2Ref(0)) / dx2),
                                                    static_cast<std::size_t>(gridx2.shape(0) - 2));
            indicesRef(2, i) = std::min( static_cast<std::size_t>((pointsRef(2, i) - gridx3Ref(0)) / dx3),
                                                    static_cast<std::size_t>(gridx3.shape(0) - 2));
        }
        return;
    }

#endif