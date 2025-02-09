/*
    * indices.hpp
    *
    * Implements raveling and unravelling of indices for a 3D grid
    * Provides function to get unique indices for a 3D grid
*/
#ifndef INDICES_HPP_
    #define INDICES_HPP_
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include "global.hpp"
    namespace py                = pybind11;

    /*
        Container struct to store all the sorted ravelled indices
        for a 3D grid
    */
    class Indices
    {
        private:
            std::vector<std::size_t> all_indices_ravel;
            std::size_t nx1, nx2, nx3;

        public:
            Indices(py::array_t<std::size_t> &indices, std::size_t nx1, std::size_t nx2, 
                    std::size_t nx3)
            {
                auto indicesRef             = indices.template unchecked<2>();
                const auto Npoints          = indices.shape(1);
                all_indices_ravel.reserve(Npoints);
                this->nx1 = nx1;
                this->nx2 = nx2;
                this->nx3 = nx3;
                for (std::size_t i=0; i<Npoints; i++)
                {
                    auto index = indicesRef(0, i) + 
                                 indicesRef(1, i) * nx1 + 
                                 indicesRef(2, i) * nx1 * nx2;
                    all_indices_ravel.push_back(index);
                }
            }

            // Check if an index is already in the list of indices
            bool IsIndexInList(std::size_t index)
            {
                return std::binary_search(all_indices_ravel.begin(), all_indices_ravel.end(), index);
            }

            // Add new indices after sorting ravelled indices
            void AddNewIndices(py::array_t<std::size_t> &new_indices)
            {
                auto new_indicesRef             = new_indices.template unchecked<2>();
                const auto Npoints              = new_indices.shape(1);
                for (std::size_t i=0; i<Npoints; i++)
                {
                    auto index = new_indicesRef(0, i) + 
                                 new_indicesRef(1, i) * this->nx1 + 
                                 new_indicesRef(2, i) * this->nx1 * this->nx2;
                    if (!IsIndexInList(index))
                    {   all_indices_ravel.push_back(index);    }
                }
                return;
            }

            // Populate unravelled indices and return them as a tuple
            py::array_t<std::size_t> PopulateUnravelledIndices()
            {
                std::sort(this->all_indices_ravel.begin(), this->all_indices_ravel.end());
                const int Npoints                   = this->all_indices_ravel.size();
                auto all_indices_py_array           = py::array_t<std::size_t> ({3, Npoints});
                auto all_indices_py_arrayRef        = all_indices_py_array.template mutable_unchecked<2>();
                #pragma omp parallel for schedule(static) num_threads(number_of_threads)
                for (std::size_t i=0; i<Npoints; i++)
                {
                    auto index                      = this->all_indices_ravel[i];
                    auto x1                         = index % this->nx1;
                    auto x2                         = (index / this->nx1) % this->nx2;
                    auto x3                         = index / (this->nx1 * this->nx2);
                    all_indices_py_arrayRef(0, i)   = x1;
                    all_indices_py_arrayRef(1, i)   = x2;
                    all_indices_py_arrayRef(2, i)   = x3;
                }
                return all_indices_py_array;
            }
    };


#endif /* INDICES_HPP_ */