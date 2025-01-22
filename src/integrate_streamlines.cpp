#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>
#include <tuple>
#include <iostream>
#include "grid.hpp"
#include "interpolation.hpp"
#include "display.hpp"


namespace py                = pybind11;

/*
    Function to integrate along a streamline for a single step
*/
template <typename T>
void TakeStep(py::array_t<T> &myFieldx,
              py::array_t<T> &myFieldy,
              py::array_t<T> &myFieldz,
              py::array_t<T> &myGridx,
              py::array_t<T> &myGridy,
              py::array_t<T> &myGridz,
              std::vector <std::size_t> &indicesx,
              std::vector <std::size_t> &indicesy,
              std::vector <std::size_t> &indicesz,
              py::array_t<T> &Posx,
              py::array_t<T> &Posy,
              py::array_t<T> &Posz,
              const std::string &coordinateSystem, const bool isUniform,
              const std::vector<bool> &should_terminate)
{
    const std::size_t Npoints  = indicesx.size();
    auto myFieldxRef           = myFieldx.template unchecked<3>();
    auto myFieldyRef           = myFieldy.template unchecked<3>();
    auto myFieldzRef           = myFieldz.template unchecked<3>();
    auto myGridxRef            = myGridx.template unchecked<1>();
    auto myGridyRef            = myGridy.template unchecked<1>();
    auto myGridzRef            = myGridz.template unchecked<1>();
    auto PosxRef               = Posx.template mutable_unchecked<1>();
    auto PosyRef               = Posy.template mutable_unchecked<1>();
    auto PoszRef               = Posz.template mutable_unchecked<1>();

    // Interpolate field values at the current position
    auto interpFieldx          = py::array_t<T> (Npoints);
    auto interpFieldy          = py::array_t<T> (Npoints);
    auto interpFieldz          = py::array_t<T> (Npoints);

    auto interpFieldxRef       = interpFieldx.template mutable_unchecked<1>();
    auto interpFieldyRef       = interpFieldy.template mutable_unchecked<1>();
    auto interpFieldzRef       = interpFieldz.template mutable_unchecked<1>();

    interpolate::InterpolateField(Posx, Posy, Posz, indicesx, indicesy, indicesz,
                     myGridx, myGridy, myGridz, myFieldx, interpFieldx, coordinateSystem);
    interpolate::InterpolateField(Posx, Posy, Posz, indicesx, indicesy, indicesz,
                     myGridx, myGridy, myGridz, myFieldy, interpFieldy, coordinateSystem);
    interpolate::InterpolateField(Posx, Posy, Posz, indicesx, indicesy, indicesz,
                     myGridx, myGridy, myGridz, myFieldz, interpFieldz, coordinateSystem);

    // Estimate the first step size along the streamline
    auto dx                    = myGridxRef(1) - myGridxRef(0);
    auto dy                    = myGridyRef(1) - myGridyRef(0);
    auto dz                    = myGridzRef(1) - myGridzRef(0);
    auto step_size             = std::min({dx, dy, dz});
    // Take a step along the streamline
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i=0; i<Npoints; i++)
    {
        if (should_terminate[i])
        {   continue;   }
            auto norm        = sqrt(interpFieldxRef(i)*interpFieldxRef(i)
                                + interpFieldyRef(i)*interpFieldyRef(i)
                                + interpFieldzRef(i)*interpFieldzRef(i));

            PosxRef(i) += interpFieldxRef(i) * step_size / norm;
            PosyRef(i) += interpFieldyRef(i) * step_size / norm;
            PoszRef(i) += interpFieldzRef(i) * step_size / norm;
    }
    if (isUniform)
    {
        ReturnClosestIndexUniform(Posx, indicesx, myGridx);
        ReturnClosestIndexUniform(Posy, indicesy, myGridy);
        ReturnClosestIndexUniform(Posz, indicesz, myGridz);
    }
    else
    {
        ReturnClosestIndexMonotonic(Posx, indicesx, myGridx);
        ReturnClosestIndexMonotonic(Posy, indicesy, myGridy);
        ReturnClosestIndexMonotonic(Posz, indicesz, myGridz);
    }
    return;
}

/*
    Main function to integrate the streamlines
    Inputs: python arrays: fields, field derivatives, grid1D, initial positions
    Outputs: python arrays: final positions of the streamlines every N steps
*/
template <typename T>
py::array_t<T> IntegrateAllStreamlines(
                                py::array_t<T> myFieldx, py::array_t<T> myFieldy, py::array_t<T> myFieldz,
                                py::array_t<T> myGridx,  py::array_t<T> myGridy,  py::array_t<T> myGridz,
                                py::array_t<T> Posx,     py::array_t<T> Posy,     py::array_t<T> Posz,
                                const std::size_t Nsteps, const std::size_t Nout,
                                const T xmin, const T xmax,
                                const T ymin, const T ymax,
                                const T zmin, const T zmax,
                                const T theta_min, 
                                const std::string &coordinateSystem)
{
    auto Npoints                = Posx.shape(0);
    bool isMonotonic            = true;
    bool isUniform              = true;
    auto Nsteps_display         = Nsteps / 20;
    auto DisplayObject          = DisplayTerminal<T>(Nsteps, "Integrating Streamlines", 
                                                     Nsteps_display);
    
    // Initial Checks to make sure that the input dimensions are consistent
    {
        auto Nx                    = myFieldx.shape(0);
        auto Ny                    = myFieldx.shape(1);
        auto Nz                    = myFieldx.shape(2);
        bool shapeMatch             = true;
        for (auto i=0; i<3; i++)
        {
            if (myFieldx.shape(i) != myFieldy.shape(i) || myFieldx.shape(i) != myFieldz.shape(i))
            {   printf("Field shapes in direction %d do not match \n", i);
                shapeMatch          = false;
            }
        }
        if (myGridx.shape(0) != Nx || myGridy.shape(0) != Ny || myGridz.shape(0) != Nz)
        {   printf("Grid shapes do not match the field shapes \n");
            shapeMatch              = false;
        }
        if (Posx.shape(0) != Posy.shape(0) || Posx.shape(0) != Posz.shape(0))
        {   printf("Initial position shapes do not match \n");
            shapeMatch              = false;
        }

        if (!shapeMatch)
        {   throw std::invalid_argument("Shapes do not match");}

        // Check if the grid is uniform and monotonic
        std::tie(isMonotonic, isUniform) = CheckIfUniform(myGridx, myGridy, myGridz);
        if (!isMonotonic)
        {   throw std::invalid_argument("Grid is not monotonic");}

        if (isUniform)
        {   std::cout << "Grid is uniform" << std::endl;}
        else
        {   std::cout << "Grid is non-uniform but monotonic" << std::endl;}

        if (coordinateSystem == "spherical")
        {   
            throw std::invalid_argument("Spherical coordinates not supported");
        }
    }

    // Convert the initial positions into cpp vectors of indices
    std::vector <std::size_t> indicesx(Npoints), indicesy(Npoints), indicesz(Npoints);

    if (isUniform)
    {
        ReturnClosestIndexUniform(Posx, indicesx, myGridx);
        ReturnClosestIndexUniform(Posy, indicesy, myGridy);
        ReturnClosestIndexUniform(Posz, indicesz, myGridz);
    }
    else
    {
        ReturnClosestIndexMonotonic(Posx, indicesx, myGridx);
        ReturnClosestIndexMonotonic(Posy, indicesy, myGridy);
        ReturnClosestIndexMonotonic(Posz, indicesz, myGridz);
    }

    // Pointers to grid
    auto Posxref                = Posx.template mutable_unchecked<1>();
    auto Posyref                = Posy.template mutable_unchecked<1>();
    auto Poszref                = Posz.template mutable_unchecked<1>();
    
    // Create output array
    auto streamline_output = py::array_t<T> ({  3, static_cast<int>(Nout), 
                                                   static_cast<int>(Npoints)});

    // Populate the streamlines
    std::vector<bool> should_terminate(Npoints, false);
    auto Ncheckpoint            = Nsteps / Nout;
    for (std::size_t i=0; i<Nsteps; i++)
    {
        bool writeOutput = (i % Ncheckpoint == 0) ? true : false;

        // Write to output array
        if (writeOutput)
        {
            auto outindex       = i / Ncheckpoint;
            auto out_ref   = streamline_output.mutable_unchecked();

            for (std::size_t j=0; j<Npoints; j++)
            {
                out_ref(0, outindex, j) = Posxref(j);
                out_ref(1, outindex, j) = Posyref(j);
                out_ref(2, outindex, j) = Poszref(j);
            }
        }

        TakeStep(   myFieldx, myFieldy, myFieldz, myGridx, myGridy, myGridz,
                    indicesx, indicesy, indicesz, Posx, Posy, Posz, coordinateSystem, isUniform, 
                    should_terminate);
        TerminationCondition(   should_terminate, Posx, Posy, Posz, 
                                indicesx, indicesy, indicesz,
                                xmin, xmax, ymin, ymax, zmin, zmax);
        DisplayObject.UpdateProgress(1);
        DisplayObject.DisplayProgress();
    }
    return streamline_output;
}


PYBIND11_MODULE(IntegrateStreamlines, m)
{
    m.doc() = "Streamline tracer module";
    m.def("IntegrateStreamlines", &IntegrateAllStreamlines<float>);
    m.def("IntegrateStreamlines", &IntegrateAllStreamlines<double>);
    py::arg("myFieldx"), py::arg("myFieldy"), py::arg("myFieldz"),
    py::arg("myGridx"), py::arg("myGridy"), py::arg("myGridz"),
    py::arg("Posx"), py::arg("Posy"), py::arg("Posz"),
    py::arg("Nsteps"), py::arg("Nout");
    py::arg("xmin"), py::arg("xmax"), py::arg("ymin"), py::arg("ymax"), py::arg("zmin"), py::arg("zmax"),
    py::arg("theta_min"), py::arg("coordinateSystem");
}
