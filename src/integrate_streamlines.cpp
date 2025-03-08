#include "global.hpp"
#include "grid.hpp"
#include "interpolation.hpp"
#include "display.hpp"
#include "coordinates.hpp"
#include "payloads.hpp"


namespace py                = pybind11;



/*
    Function to integrate along a streamline for a single step
*/
template <typename T>
float TakeStep( py::array_t<T> &points,
                py::array_t<T> &field,
                Grid<T> &Grid,
                py::array_t<std::size_t> &indices,
                std::vector<bool> &should_terminate,
                py::array_t<T> &current_quantity_values,
                std::vector<std::string> &payload_names,
                std::vector<py::array_t<T>> &payload_arrays,
                Timers &timers)
{
    timers.BeginTimer("Interpolation");
    auto interpolated_field     = InterpolateField( points, 
                                                    field,
                                                    Grid,
                                                    indices,
                                                    should_terminate);
    auto interpolated_fieldRef  = interpolated_field.template unchecked<2>();
    timers.EndTimer("Interpolation");

    timers.BeginTimer("Coordinate Transformation");
    auto pointsRef              = points.template mutable_unchecked<2>();
    auto indicesRef             = indices.template unchecked<2>();
    auto current_quantity_valuesRef = current_quantity_values.mutable_unchecked();
    const auto Npoints          = points.shape(1);
    std::size_t count           = 0;
    auto cart_points            = py::array_t<T>({3, static_cast<int>(Npoints)});
    auto cart_pointsRef         = cart_points.template mutable_unchecked<2>();
    ConvertCoordiantes(points, Grid.grid_coord_system, "cartesian", cart_points);
    timers.EndTimer("Coordinate Transformation");

    timers.BeginTimer("Integration");
    #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads) reduction(+:count)
    for (std::size_t i=0; i<Npoints; i++)
    {
        if (should_terminate[i])
        {   count++;    continue;   }
        auto norm        = sqrt(square(interpolated_fieldRef(0,i)) +
                                square(interpolated_fieldRef(1,i)) +
                                square(interpolated_fieldRef(2,i)));
        auto ix1         = indicesRef(0,i);
        auto ix2         = indicesRef(1,i);
        auto ix3         = indicesRef(2,i);

        auto dx1         = Grid.gridx1Ref(ix1+1) - Grid.gridx1Ref(ix1);
        auto dx2         = Grid.gridx2Ref(ix2+1) - Grid.gridx2Ref(ix2);
        auto dx3         = Grid.gridx3Ref(ix3+1) - Grid.gridx3Ref(ix3);

        T stepsize;
        if (Grid.grid_coord_system == "cartesian")
        {   stepsize = sqrt(square(dx1) + square(dx2) + square(dx3));}
        else if (Grid.grid_coord_system == "spherical")
        {   
            stepsize = sqrt(square(dx1) + square(Grid.gridx1Ref(ix1)*dx2) + 
            square(Grid.gridx1Ref(ix1)*sin(Grid.gridx2Ref(ix2)*dx3)));
        }
        else if (Grid.grid_coord_system == "log_spherical")
        {
            stepsize =  exp(Grid.gridx1Ref(ix1)) * sqrt(square(dx1) + square(dx2) +
                        square(sin(Grid.gridx2Ref(ix2))*dx3));
        }
        cart_pointsRef(0,i) += interpolated_fieldRef(0,i) * stepsize * CFL / norm;
        cart_pointsRef(1,i) += interpolated_fieldRef(1,i) * stepsize * CFL / norm;
        cart_pointsRef(2,i) += interpolated_fieldRef(2,i) * stepsize * CFL / norm;

        // Compute the user-defined quantity
        current_quantity_valuesRef(i) = CustomUserOperation( ix1, ix2, ix3,
                                                            pointsRef(0,i), pointsRef(1,i), pointsRef(2,i),
                                                            dx1, dx2, dx3,
                                                            interpolated_fieldRef(0,i), interpolated_fieldRef(1,i), interpolated_fieldRef(2,i),
                                                            payload_names, payload_arrays);
    }
    timers.EndTimer("Integration");

    timers.BeginTimer("Coordinate Transformation");
    ConvertCoordiantes(cart_points, "cartesian", Grid.grid_coord_system, points);
    timers.EndTimer("Coordinate Transformation");

    return count / Npoints;
}

/*
    Main function to integrate the streamlines
    Inputs: python arrays: fields, field derivatives, grid1D, initial positions
    Outputs: python arrays: final positions of the streamlines every N steps


    payloads = (npayloads, {'name 1', ..., 'name npayload'}, {payload1, ..., payload npayload})
*/
template <typename T>
py::tuple IntegrateAllStreamlines(  py::array_t<T> field,
                                    py::array_t<T> gridx1, 
                                    py::array_t<T> gridx2,
                                    py::array_t<T> gridx3,
                                    std::string grid_coord_system,
                                    py::array_t<T> points, 
                                    std::string point_coord_system,
                                    const std::size_t Nsteps, const std::size_t Nout,
                                    py::tuple payloads)
{
    auto display_every              = Nsteps / Ndisplay;
    auto Ncheckpoint                = Nsteps / Nout;
    Timers timers(Nsteps, display_every);
    timers.AddTimer({"Indexing", "Initial Checks", "Output", "Interpolation", "Coordinate Transformation",
                     "Integration"});
    Grid Grid(gridx1, gridx2, gridx3, grid_coord_system);
    bool end_integration            = false;
    const auto Npoints              = points.shape(1);
    std::vector<bool> should_terminate(Npoints, false);
    omp_set_dynamic(0);

    std::size_t number_of_payloads;
    std::vector<std::string> payload_names;
    std::vector<py::array_t<T>> payload_arrays;
    std::cout << "Performing initial checks" << std::endl;
    // Initial Checks to make sure that the input dimensions are consistent
    timers.BeginTimer("Initial Checks");
    {
        if (grid_coord_system != "cartesian" && grid_coord_system != "spherical" && 
            grid_coord_system != "log_spherical")
        {
            throw std::invalid_argument(
                "Invalid grid coordinate system. Must be 'cartesian', 'spherical' or 'log_spherical'");
        }
        std::cout << "Grid coordinate system: " << grid_coord_system << std::endl;

        if (point_coord_system != "cartesian" && point_coord_system != "spherical" && 
            point_coord_system != "log_spherical")
        {
            throw std::invalid_argument(
                "Invalid point coordinate system. Must be 'cartesian', 'spherical' or 'log_spherical'");
        }
        std::cout << "Point coordinate system: " << point_coord_system << std::endl;

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
        // Change the coordinate system of the points to that of the grid
        if (point_coord_system != grid_coord_system)
        {
            ConvertCoordiantes(points, point_coord_system, grid_coord_system, points);
        }
        std::cout << "Converted point coordinates to grid coordinate system" << std::endl;
        if (!Grid.isMonotonic)
        {   throw std::invalid_argument("Grid is not monotonic");}

        if (Grid.isUniform)
        {   std::cout << "Grid is uniform" << std::endl;}
        else
        {   std::cout << "Grid is non-uniform but monotonic; interpolation will be slower" << std::endl;}

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
    }
    timers.EndTimer("Initial Checks");
    std::cout << "Initial checks complete" << std::endl;


    // Indices of closest grid points to the initial positions
    auto indices        = py::array_t<std::size_t> ({3, static_cast<int>(Npoints)});

    timers.BeginTimer("Indexing");
    Grid.ReturnClosestIndex(points, indices, should_terminate);
    timers.EndTimer("Indexing");

    // Auxilliary user-constructed quantities computed at each timestep
    auto current_quantity_values = py::array_t<T> ({static_cast<int>(Npoints)});

    // Create output and termination arrays
    auto streamline_output = py::array_t<T> ({  3, static_cast<int>(Nout), 
                                                static_cast<int>(Npoints)});
    // Auxilliary user-constructed quantities
    auto auxilliary_output = py::array_t<T> ({ static_cast<int>(Nout), 
                                                static_cast<int>(Npoints)});
    std::size_t iteration;
    // Populate the streamlines
    for (iteration=0; iteration < Nsteps; iteration++)
    {
        if (end_integration)    {   break;  }
        bool writeOutput        = (iteration % Ncheckpoint == 0) ? true : false;
        if (writeOutput)
        {
            timers.BeginTimer("Output");
            auto outindex       = iteration / Ncheckpoint;
            auto out_ref        = streamline_output.mutable_unchecked();
            auto aux_ref        = auxilliary_output.mutable_unchecked();
            auto current_ref    = current_quantity_values.unchecked();

            // Convert back to original coordinates
            if (point_coord_system != grid_coord_system)
            {   ConvertCoordiantes(points, grid_coord_system, point_coord_system, points);}
            
            auto pointsRef          = points.template unchecked<2>();
            for (std::size_t j=0; j<Npoints; j++)
            {
                aux_ref(outindex, j) = current_ref(j);
                for (auto coord_id=0; coord_id < 3; coord_id++)
                {   out_ref(coord_id, outindex, j) = pointsRef(coord_id, j);    }
            }

            // Convert to grid coordinates again
            if (point_coord_system != grid_coord_system)
            {
                ConvertCoordiantes(points, point_coord_system, grid_coord_system, points);
            }
            timers.EndTimer("Output");
        }

        auto percent_termiante = TakeStep(  points, field, Grid, indices, should_terminate, 
                                            current_quantity_values, payload_names,
                                            payload_arrays, timers);
        end_integration = percent_termiante > terminate_fraction ? true : false;

        // Find new indices after taking the step
        timers.BeginTimer("Indexing");
        Grid.ReturnClosestIndex(points, indices, should_terminate);
        timers.EndTimer("Indexing");

        timers.PrintTimers(iteration);
    }

    // Additionally return the vector field transformed to cartesian coordinates, on a uniform cartesian grid
    timers.BeginTimer("Output");
    {
        auto outindex       = iteration / Ncheckpoint;
        auto out_ref        = streamline_output.mutable_unchecked();
        auto aux_ref        = auxilliary_output.mutable_unchecked();
        auto current_ref    = current_quantity_values.unchecked();
        // Convert back to original coordinates
        if (point_coord_system != grid_coord_system)
        {   ConvertCoordiantes(points, grid_coord_system, point_coord_system, points); }

        auto pointsRef          = points.template unchecked<2>();
        for (auto curindex = outindex; curindex < Nout; curindex++)
        {
            for (std::size_t j=0; j<Npoints; j++)
            {
                aux_ref(curindex, j) = current_ref(j);
                for (auto coord_id=0; coord_id < 3; coord_id++)
                {   out_ref(coord_id, curindex, j) = pointsRef(coord_id, j);    }
            }
        }
        // Convert to grid coordinates again
        if (point_coord_system != grid_coord_system)
        {   ConvertCoordiantes(points, point_coord_system, grid_coord_system, points);}
    }
    timers.EndTimer("Output");
    std::cout << "Integration complete" << std::endl;
    timers.PrintTimers(Nsteps);

    // Clear all memory
    {
        payload_names.clear();
        payload_arrays.clear();
        should_terminate.clear();
    }

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
    py::arg("points"), py::arg("point_coord_system"),
    py::arg("Nsteps"), py::arg("Nout"),
    py::arg("payloads") = py::none();
}
