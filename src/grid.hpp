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

    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <tuple>
    #include <stdlib.h>
    #include <iostream>

    namespace py                = pybind11;

    /*
        Function to check if the grid is monotonic and  
        uniform
        Returns a tuple of two booleans (isMonotonic, isUniform)
    */
    template <typename T>
    std::tuple<bool, bool> CheckIfUniform(  py::array_t<T> &myGridx,
                                            py::array_t<T> &myGridy,
                                            py::array_t<T> &myGridz)
    {
        bool isMonotonic            = true;
        bool isUniform              = true;
        auto gridxRef               = myGridx.template unchecked<1>();
        auto gridyRef               = myGridy.template unchecked<1>();
        auto gridzRef               = myGridz.template unchecked<1>();
        const std::size_t ny        = gridyRef.shape(0);
        const std::size_t nx        = gridxRef.shape(0);
        const std::size_t nz        = gridzRef.shape(0);
        if (nx < 2 || ny < 2 || nz < 2)
        {   throw std::invalid_argument("Each dimension must have at least 2 cells");}

        float dx                    = gridxRef(1) - gridxRef(0);
        float dy                    = gridyRef(1) - gridyRef(0);
        float dz                    = gridzRef(1) - gridzRef(0);
        std::cout << "dx = " << dx << " dy = " << dy << " dz = " << dz << std::endl;
        // Check x array
        for (std::size_t ix=0; ix<nx-1; ix++)
        {
            if ((fabs(1 - (gridxRef(ix+1) - gridxRef(ix))/dx) > 0.01))
            {   isUniform           = false;    }
            if (gridxRef(ix+1) <= gridxRef(ix))
            {   isMonotonic         = false;    
            }
        }
        // Check y array
        for (std::size_t iy=0; iy<ny-1; iy++)
        {
            if ((fabs(1 - (gridyRef(iy+1) - gridyRef(iy))/dy) > 0.01))
            {   isUniform           = false;    }
            if (gridyRef(iy+1) <= gridyRef(iy))
            {   isMonotonic         = false;    
            }
        }
        // Check z array
        for (std::size_t iz=0; iz<nz-1; iz++)
        {
            if ((fabs(1 - (gridzRef(iz+1) - gridzRef(iz))/dz) > 0.01))
            {   isUniform           = false;    }
            if (gridzRef(iz+1) <= gridzRef(iz))
            {   isMonotonic         = false;    
            }
        }
        return std::make_tuple( isMonotonic, isUniform);
    }


    /*
        Find the nearest cell in the grid that is greater than or equal
        to xd[j] for a monotonically increasing 1D grid, for all j's
        <---><---><--->...<--->...<--->
    i =  0     1    2      k        n
                            ^xd[j]
                            -> return k
    */
    template <typename T>
    void ReturnClosestIndexMonotonic(py::array_t<T> &xk, std::vector <std::size_t> &indices,
                                     py::array_t<T> &myGrid1D)
    {
        auto gridRef                = myGrid1D.template unchecked<1>();
        auto xkRef                  = xk.template unchecked<1>();
        const std::size_t Npoints   = xkRef.shape(0);
        const std::size_t Ngrid     = gridRef.shape(0);
        #pragma omp parallel for schedule(dynamic)
        for (std::size_t i=0; i<Npoints; i++)
        {
            std::size_t index = 0;
            while(gridRef(index) <= xkRef(i) || index < Ngrid-2)
            {   index++;    }
            indices[i] = std::max(index - 1, static_cast<std::size_t>(0));
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
   void ReturnClosestIndexUniform( py::array_t<T> &xk, std::vector <std::size_t> &indices,
                                    py::array_t<T> &myGrid1D)
    {
        auto gridRef                = myGrid1D.template unchecked<1>();
        auto xkRef                  = xk.template unchecked<1>();
        const std::size_t Npoints   = xkRef.shape(0);
        const std::size_t Ngrid     = gridRef.shape(0);
        float dx                    = gridRef(1) - gridRef(0);
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<Npoints; i++)
        {   indices[i] = std::min(  static_cast<std::size_t>(xkRef(i) / dx),
                                    static_cast<std::size_t>(Ngrid - 2));
        }
        return;
    }

    /*
        Condition to check if the streamline has left the grid
    */
    template <typename T>
    void TerminationCondition(  std::vector<bool> &should_terminate, const py::array_t<T> &Posx1,
                                const py::array_t<T> &Posx2, const py::array_t<T> &Posx3,
                                const std::vector<std::size_t> &indices_x1,
                                const std::vector<std::size_t> &indices_x2,
                                const std::vector<std::size_t> &indices_x3,
                                const T x1min, const T x1max,
                                const T x2min, const T x2max,
                                const T x3min, const T x3max)
    {
        auto posx1_ref             = Posx1.template unchecked<1>();
        auto posx2_ref             = Posx2.template unchecked<1>();
        auto posx3_ref             = Posx3.template unchecked<1>();
        const auto Npoints = posx1_ref.shape(0);
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   continue;   }

            if (posx1_ref(i) < x1min || posx1_ref(i) > x1max ||
                posx2_ref(i) < x2min || posx2_ref(i) > x2max ||
                posx3_ref(i) < x3min || posx3_ref(i) > x3max ||
                indices_x1[i] == 0 || indices_x2[i] == 0 || indices_x3[i] == 0 ||
                indices_x1[i] == indices_x1.size()-1 ||
                indices_x2[i] == indices_x2.size()-1 ||
                indices_x3[i] == indices_x3.size()-1)
            {   should_terminate[i] = true;    }
        }
        return;
    }

    // Declare all template arguments
    //template std::tuple<bool, bool> CheckIfUniform<float>(  py::array_t<float> &myGridx,
    //                                                        py::array_t<float> &myGridy,
    //                                                        py::array_t<float> &myGridz);
    //template void ReturnClosestIndexMonotonic<float>(py::array_t<float> &xk, std::vector <std::size_t> &indices,
    //                                                    py::array_t<float> &myGrid1D);
    //template void ReturnClosestIndexUniform<float>(  py::array_t<float> &xk, std::vector <std::size_t> &indices,
    //                                                py::array_t<float> &myGrid1D);
    //template void TerminationCondition<float>(  std::vector<bool> &should_terminate, const py::array_t<float> &Posx1,
    //                                            const py::array_t<float> &Posx2, const py::array_t<float> &Posx3,
    //                                            const std::vector<std::size_t> &indices_x1,
    //                                            const std::vector<std::size_t> &indices_x2,
    //                                            const std::vector<std::size_t> &indices_x3,
    //                                            const float x1min, const float x1max,
    //                                            const float x2min, const float x2max,
    //                                            const float x3min, const float x3max);

#endif