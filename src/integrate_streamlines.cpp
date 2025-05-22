#include "global.hpp"
#include "grid.hpp"
#include "interpolation.hpp"
#include "display.hpp"
#include "coordinates.hpp"
#include "payloads.hpp"
#include "integration.hpp"

namespace py                = pybind11;

template <typename T>
py::tuple InitializeData(   const py::array_t<T> &field_input,
                            const py::array_t<T> &gridx1_input, 
                            const py::array_t<T> &gridx2_input,
                            const py::array_t<T> &gridx3_input,
                            const py::array_t<T> &points_input,
                            const std::string &grid_coord_system)
{
    auto field_copy      = py::array_t<T>(field_input.request());
    auto gridx1_copy     = py::array_t<T>(gridx1_input.request());
    auto gridx2_copy     = py::array_t<T>(gridx2_input.request());
    auto gridx3_copy     = py::array_t<T>(gridx3_input.request());
    auto points_copy     = py::array_t<T>(points_input.request());
    return py::make_tuple(field_copy, gridx1_copy, gridx2_copy, gridx3_copy, points_copy);
}


/*
    Main function to integrate the streamlines
    Inputs: python arrays: fields, field derivatives, grid1D, initial positions
    Outputs: python arrays: final positions of the streamlines every N steps
    payloads = (npayloads, {'name 1', ..., 'name npayload'}, {payload1, ..., payload npayload})
*/
template <typename T>
py::tuple IntegrateAllStreamlines(  const py::array_t<T> &field_input,
                                    const py::array_t<T> &gridx1_input, 
                                    const py::array_t<T> &gridx2_input,
                                    const py::array_t<T> &gridx3_input,
                                    const std::string &grid_coord_system,
                                    py::array_t<T>  &points_input, 
                                    const std::size_t Nsteps, 
                                    const std::size_t Nout,
                                    const py::tuple &payloads)
{
    // Copy input into local arrays
    py::tuple data                  = InitializeData(   field_input, gridx1_input, gridx2_input,
                                                        gridx3_input, points_input, grid_coord_system);
    auto field                      = data[0].cast<py::array_t<T>>();
    auto gridx1                     = data[1].cast<py::array_t<T>>();
    auto gridx2                     = data[2].cast<py::array_t<T>>();
    auto gridx3                     = data[3].cast<py::array_t<T>>();
    auto points                     = data[4].cast<py::array_t<T>>();

    auto display_every              = Nsteps / Ndisplay;
    std::size_t Ncheckpoint         = Nsteps / Nout;
    Timers timers(Nsteps, display_every);
    std::vector<std::string> timer_names = {"Initial Checks", "Output", "Integration"};
    timers.AddTimer(timer_names);
    Grid<T> Grid(gridx1, gridx2, gridx3, grid_coord_system);
    bool end_integration            = false;
    const auto Npoints              = points.shape(1);
    std::vector<bool> should_terminate(Npoints, false);
    omp_set_dynamic(0);
    std::size_t number_of_payloads;
    std::vector<std::string> payload_names;
    std::vector<py::array_t<T>> payload_arrays;
    std::cout << "Performing initial checks" << std::endl;
    auto pointsRef                  = points.template unchecked<2>();
    // Initial Checks to make sure that the input dimensions are consistent
    timers.BeginTimer("Initial Checks");
    {
        if (grid_coord_system != "cartesian" && (grid_coord_system != "spherical" && 
            grid_coord_system != "log_spherical"))
        {
            throw std::invalid_argument(
                "Invalid grid coordinate system. Must be 'cartesian', 'spherical' or 'log_spherical'");
        }
        std::cout << "Grid coordinate system: " << grid_coord_system << std::endl;

        if (field.shape(1) != gridx1.shape(0) || field.shape(2) != gridx2.shape(0) || 
            field.shape(3) != gridx3.shape(0))
        {   throw std::invalid_argument("Field and grid dimensions do not match");}
        std::cout << "Field and grid dimensions match" << std::endl;

        if (field.ndim() != 4)
        {   throw std::invalid_argument("Field must shape: (3, Nx, Ny, Nz)");}
        std::cout << "Field shape: (3, Nx, Ny, Nz)" << std::endl;

        if (gridx1.ndim() != 1)
        {   throw std::invalid_argument("Grid x1 must be 1D");}
        std::cout << "Grid x1 is 1D" << std::endl;
        if (gridx2.ndim() != 1)
        {   throw std::invalid_argument("Grid x2 must be 1D");}
        std::cout << "Grid x2 is 1D" << std::endl;
        if (gridx3.ndim() != 1)
        {   throw std::invalid_argument("Grid x3 must be 1D");}
        std::cout << "Grid x3 is 1D" << std::endl;
        if (points.ndim() != 2)
        {   throw std::invalid_argument("Point array must have shape: (3, Npoints)");}
        std::cout << "Point array shape: (3, Npoints)" << std::endl;
        std::cout << "Converted point coordinates to grid coordinate system" << std::endl;
        if (!Grid.isMonotonic)
        {   throw std::invalid_argument("Grid is not monotonic");}

        if (Grid.isUniform)
        {   std::cout << "Grid is uniform" << std::endl;}
        else
        {   throw std::invalid_argument("Non uniform grids not implemented");}
        
        {
            number_of_payloads  = payloads[0].cast<std::size_t>();
            payload_names       = payloads[1].cast<std::vector<std::string>>();
            payload_arrays      = payloads[2].cast<std::vector<py::array_t<T>>>();
            if (number_of_payloads != payload_names.size() || number_of_payloads != payload_arrays.size())
            {   throw std::invalid_argument("Payloads are not consistent");}
            std::cout << "Payload names and shapes are: ";
            for (std::size_t i=0; i<number_of_payloads; i++)
            {   
                std::cout << payload_names[i] << " ";
                std::cout << "(";
                for (std::size_t j=0; j<payload_arrays[i].ndim(); j++)
                {   std::cout << payload_arrays[i].shape(j) << " ";    }
                std::cout << ") ";
            }
            std::cout << std::endl;
        }
        {
            if constexpr (integration_method ==  IntegrationMethod::Euler)
            {   std::cout << "Integration method: Euler" << std::endl;    }
            else if constexpr (integration_method ==  IntegrationMethod::RK4)
            {   std::cout << "Integration method: RK4" << std::endl;    }
            else
            {   throw std::invalid_argument("Integration method not implemented");    }
        }
    }
    timers.EndTimer("Initial Checks");
    std::cout << "Initial checks complete" << std::endl;

    // Indices of closest grid points to the initial positions
    auto indices        = py::array_t<std::size_t> ({3, static_cast<int>(Npoints)});

    // Auxilliary user-constructed quantities computed at each timestep
    auto current_quantity_values = py::array_t<T> ({static_cast<int>(Npoints)});

    // Create output and termination arrays
    auto streamline_output = py::array_t<T> ({  3, static_cast<int>(Nout), 
                                                static_cast<int>(Npoints)});
    // Auxilliary user-constructed quantities
    auto auxilliary_output = py::array_t<T> ({ static_cast<int>(Nout), 
                                                static_cast<int>(Npoints)});
    std::size_t iteration;
    float percent_terminate = 0.0;
    // Main streamline loop
    for (iteration=0; iteration < Nsteps; iteration++)
    {
        if (end_integration)    {   break;  }
        bool writeOutput        = (iteration % Ncheckpoint == 0) ? true : false;
        if (writeOutput)
        {
            timers.BeginTimer("Output");
            std::size_t  outindex  = static_cast<std::size_t>((static_cast<double>(iteration) / Nsteps) * Nout);
            auto out_ref        = streamline_output.mutable_unchecked();
            auto aux_ref        = auxilliary_output.mutable_unchecked();
            auto current_ref    = current_quantity_values.unchecked();

            auto pointsRef          = points.template unchecked<2>();
            for (std::size_t j=0; j<Npoints; j++)
            {    
                aux_ref(outindex, j) = current_ref(j);
                for (auto coord_id=0; coord_id < 3; coord_id++)
                {   out_ref(coord_id, outindex, j) = pointsRef(coord_id, j);    }
                
            }
            timers.EndTimer("Output");
        }


        timers.BeginTimer("Integration");
        if constexpr (integration_method ==  IntegrationMethod::Euler)
        {   percent_terminate = Integrators::TakeStepEuler( points, field, Grid, should_terminate, 
                                            current_quantity_values, payload_names,
                                            payload_arrays, timers);
        }
        else if constexpr (integration_method ==  IntegrationMethod::RK4)
        {   percent_terminate = Integrators::TakeStepRK4(  points, field, Grid, should_terminate, 
                                            current_quantity_values, payload_names,
                                            payload_arrays, timers);
        }
        else
        {   std::cout << "Integration method not implemented" << std::endl;    }
        end_integration = percent_terminate > terminate_fraction ? true : false;
        timers.EndTimer("Integration");
        
        timers.PrintString("Iteration: " + std::to_string(iteration) + 
                            " Percent terminated: " + std::to_string(percent_terminate * 100) + "%",
                            iteration);
        timers.PrintTimers(iteration);
    }

    // Additionally return the vector field transformed to cartesian coordinates, on a uniform cartesian grid
    timers.BeginTimer("Output");
    {
        std::size_t  outindex  = static_cast<std::size_t>((static_cast<double>(iteration) / Nsteps) * Nout);
        auto out_ref        = streamline_output.mutable_unchecked();
        auto aux_ref        = auxilliary_output.mutable_unchecked();
        auto current_ref    = current_quantity_values.unchecked();
        auto pointsRef      = points.template unchecked<2>();
        for (auto curindex = outindex; curindex < Nout; curindex++)
        {
            for (std::size_t j=0; j<Npoints; j++)
            {
                aux_ref(curindex, j) = current_ref(j);
                for (auto coord_id=0; coord_id < 3; coord_id++)
                {   out_ref(coord_id, curindex, j) = pointsRef(coord_id, j);    }
            }
        }
    }
    timers.EndTimer("Output");
    timers.PrintTimers(Nsteps);
    std::cout << "Integration complete" << std::endl;
    return py::make_tuple(streamline_output, auxilliary_output);
}

PYBIND11_MODULE(IntegrateStreamlines, m)
{
    m.doc() = "Streamline tracer module";
    m.def("IntegrateStreamlines", &IntegrateAllStreamlines<float>);
    m.def("IntegrateStreamlines", &IntegrateAllStreamlines<double>);
    py::arg("field"),
    py::arg("gridx1"), py::arg("gridx2"), py::arg("gridx3"),
    py::arg("grid_coord_system"),
    py::arg("points"),
    py::arg("Nsteps"), py::arg("Nout"),
    py::arg("payloads") = py::none();
}
