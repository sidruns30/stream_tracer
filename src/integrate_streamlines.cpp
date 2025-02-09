#include "global.hpp"
#include "grid.hpp"
#include "interpolation.hpp"
#include "display.hpp"
#include "coordinates.hpp"
#include "adaptive_stepsize.hpp"
#include "indices.hpp"


namespace py                = pybind11;

/*
    Function to integrate along a streamline for a single step
*/
template <typename T>
void TakeStep(  py::array_t<T> &points,
                py::array_t<T> &field,
                std::tuple<py::array_t<T>, py::array_t<T>, py::array_t<T>> &grid,
                py::array_t<std::size_t> &indices,
                std::vector<bool> &should_terminate,
                std::string grid_coord_system,
                const bool use_adaptive_stepsize,
                Timers &timers)
{

    timers.BeginTimer("Interpolation");
    auto interpolated_field     = InterpolateField(points, field, grid, indices, should_terminate, grid_coord_system);
    auto interpolated_fieldRef  = interpolated_field.template unchecked<2>();
    timers.EndTimer("Interpolation");

    // Estimate the first step size along the streamline
    auto gridx1                 = std::get<0>(grid);
    auto gridx2                 = std::get<1>(grid);
    auto gridx3                 = std::get<2>(grid);
    auto gridx1Ref              = std::get<0>(grid).template unchecked<1>();
    auto gridx2Ref              = std::get<1>(grid).template unchecked<1>();
    auto gridx3Ref              = std::get<2>(grid).template unchecked<1>();
    const auto dx1              = gridx1Ref(1) - gridx1Ref(0);
    const auto dx2              = gridx2Ref(1) - gridx2Ref(0);
    const auto dx3              = gridx3Ref(1) - gridx3Ref(0);
    T uniform_step_size;

    auto pointsRef             = points.template mutable_unchecked<2>();
    auto indicesRef            = indices.template unchecked<2>();
    const auto Npoints         = points.shape(1);
    
    if (use_adaptive_stepsize)
    {
        timers.BeginTimer("Adaptive Stepsize");
        auto step_size             = compute_adaptive_stepsize(indices, field, grid, 
                                    grid_coord_system, should_terminate);
        auto step_sizeRef          = step_size.template unchecked<1>();
        timers.EndTimer("Adaptive Stepsize");
    
        timers.BeginTimer("Integration");
        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   continue;   }
                auto norm        = sqrt(square(interpolated_fieldRef(0,i)) +
                                    square(interpolated_fieldRef(1,i)) +
                                    square(interpolated_fieldRef(2,i)));
                pointsRef(0,i) += interpolated_fieldRef(0,i) * step_sizeRef(i) / norm;
                pointsRef(1,i) += interpolated_fieldRef(1,i) * step_sizeRef(i) / norm;
                pointsRef(2,i) += interpolated_fieldRef(2,i) * step_sizeRef(i) / norm;

        }
        timers.EndTimer("Integration");
    }
    else
    {
        timers.BeginTimer("Integration");
        #pragma omp parallel for schedule(dynamic) num_threads(number_of_threads)
        for (std::size_t i=0; i<Npoints; i++)
        {
            if (should_terminate[i])
            {   continue;   }
            auto norm        = sqrt(square(interpolated_fieldRef(0,i)) +
                                square(interpolated_fieldRef(1,i)) +
                                square(interpolated_fieldRef(2,i)));
            auto ix1         = indicesRef(0,i);
            auto ix2         = indicesRef(1,i);
            auto ix3         = indicesRef(2,i);
            if (ix1 == gridx1.shape(0) - 1 || ix2 == gridx2.shape(0) - 1 || ix3 == gridx3.shape(0) - 1)
            {
                uniform_step_size = dx1 * CFL;
            }
            else
            {
                uniform_step_size = (gridx1Ref(ix1+1) - gridx1Ref(ix1)) * CFL;
            }
            pointsRef(0,i) += interpolated_fieldRef(0,i) * uniform_step_size / norm;
            pointsRef(1,i) += interpolated_fieldRef(1,i) * uniform_step_size / norm;
            pointsRef(2,i) += interpolated_fieldRef(2,i) * uniform_step_size / norm;
            if (grid_coord_system == "spherical")
            {
                if (pointsRef(0,i) < inner_termination_radius)
                {   should_terminate[i] = true;    }
            }
            else if (grid_coord_system == "log_spherical")
            {
                if (exp(pointsRef(0,i)) < inner_termination_radius)
                {   should_terminate[i] = true;    }
            }
            else if (grid_coord_system == "cartesian")
            {
                if (sqrt(square(pointsRef(0,i)) + square(pointsRef(1,i)
                    + square(pointsRef(2,i)))) < inner_termination_radius)
                {   should_terminate[i] = true;    }
            }
        }
        timers.EndTimer("Integration");
    }
    return;
}

/*
    Main function to integrate the streamlines
    Inputs: python arrays: fields, field derivatives, grid1D, initial positions
    Outputs: python arrays: final positions of the streamlines every N steps
*/
template <typename T>
py::tuple IntegrateAllStreamlines( py::array_t<T> field,
                                        std::string field_coord_system,
                                        py::array_t<T> gridx1, 
                                        py::array_t<T> gridx2,
                                        py::array_t<T> gridx3,
                                        std::string grid_coord_system,
                                        py::array_t<T> points, 
                                        std::string point_coord_system,
                                        const bool use_adaptive_stepsize,
                                        const std::size_t Nsteps, const std::size_t Nout)
{
    auto display_every              = Nsteps / Ndisplay;
    auto Ncheckpoint                = Nsteps / Nout;

    Timers timers(Nsteps, display_every);
    timers.AddTimer({"Indexing", "Initial Checks", "Output", "Interpolation",
                     "Adaptive Stepsize", "Integration"});
    bool isMonotonic                = true;
    bool isUniform                  = true;
    bool raise_out_of_bounds_error  = true;
    bool end_integration            = false;
    const auto Npoints              = points.shape(1);
    auto grid                       = std::make_tuple(gridx1, gridx2, gridx3);
    std::vector<bool> should_terminate(Npoints, false);
    omp_set_dynamic(0);

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
            auto new_points = ConvertCoordiantes(points, point_coord_system, grid_coord_system);
            points = new_points;
        }
        std::cout << "Converted point coordinates to grid coordinate system" << std::endl;
        // Check if the coordinates are within the bounds
        {   CheckBounds(points, grid, should_terminate, raise_out_of_bounds_error);    }
        std::cout << "Checked bounds" << std::endl;
        // Check if the grid is uniform and monotonic
        std::tie(isMonotonic, isUniform) = CheckIfUniform(grid);
        if (!isMonotonic)
        {   throw std::invalid_argument("Grid is not monotonic");}

        if (isUniform)
        {   std::cout << "Grid is uniform" << std::endl;}
        else
        {   std::cout << "Grid is non-uniform but monotonic; interpolation will be slower" << std::endl;}
    }
    timers.EndTimer("Initial Checks");
    std::cout << "Initial checks complete" << std::endl;


    // Indices of closest grid points to the initial positions
    auto indices        = py::array_t<std::size_t> ({3, static_cast<int>(Npoints)});

    timers.BeginTimer("Indexing");
    if (isUniform)
    {   ReturnClosestIndexUniform(points, indices, grid);   }
    else
    {   ReturnClosestIndexMonotonic(points, indices, grid); }
    timers.EndTimer("Indexing");

    // Container to store the indices of the grid points which the streamlines cross
    Indices UniqueIndices = Indices(indices, gridx1.shape(0), gridx2.shape(0), gridx3.shape(0));


    // Create output and termination arrays
    auto streamline_output = py::array_t<T> ({  3, static_cast<int>(Nout), 
                                                static_cast<int>(Npoints)});
    raise_out_of_bounds_error = false;
    std::size_t iteration;
    // Populate the streamlines
    for (iteration=0; iteration<Nsteps; iteration++)
    {
        if (end_integration)
        {   break;  }
        bool writeOutput        = (iteration % Ncheckpoint == 0) ? true : false;
        if (writeOutput)
        {
            timers.BeginTimer("Output");
            auto outindex       = iteration / Ncheckpoint;
            auto out_ref        = streamline_output.mutable_unchecked();

            // Convert back to original coordinates
            if (point_coord_system != grid_coord_system)
            {
                auto new_points = ConvertCoordiantes(points, grid_coord_system, point_coord_system);
                points = new_points;
            }
            
            auto pointsRef          = points.template unchecked<2>();
            for (std::size_t j=0; j<Npoints; j++)
            {
                for (auto coord_id=0; coord_id < 3; coord_id++)
                {   out_ref(coord_id, outindex, j) = pointsRef(coord_id, j);    }
            }

            // Convert to grid coordinates again
            if (point_coord_system != grid_coord_system)
            {
                auto new_points = ConvertCoordiantes(points, point_coord_system, grid_coord_system);
                points = new_points;
            }
            timers.EndTimer("Output");
        }

        TakeStep(   points, field, grid, indices, should_terminate, grid_coord_system, 
                    use_adaptive_stepsize, timers);
        CheckBounds(points, grid, should_terminate, raise_out_of_bounds_error);

        // Find new indices after taking the step
        timers.BeginTimer("Indexing");
        if (isUniform)
        {   ReturnClosestIndexUniform(points, indices, grid);   }
        else
        {   ReturnClosestIndexMonotonic(points, indices, grid); }
        timers.EndTimer("Indexing");

        UniqueIndices.AddNewIndices(indices);

        timers.PrintTimers(iteration);
        end_integration = PercentTerminate(should_terminate) > terminate_fraction ? true : false;
    }

    // Additionally return the vector field transformed to cartesian coordinates, on a uniform cartesian grid
    timers.BeginTimer("Output");
    {
        auto outindex       = iteration / Ncheckpoint;
        auto out_ref        = streamline_output.mutable_unchecked();
        // Convert back to original coordinates
        if (point_coord_system != grid_coord_system)
        {
            auto new_points = ConvertCoordiantes(points, grid_coord_system, point_coord_system);
            points = new_points;
        }

        auto pointsRef          = points.template unchecked<2>();
        for (auto curindex = outindex; curindex < Nout; curindex++)
        {
            for (std::size_t j=0; j<Npoints; j++)
            {
                for (auto coord_id=0; coord_id < 3; coord_id++)
                {   out_ref(coord_id, curindex, j) = pointsRef(coord_id, j);    }
            }
        }
        // Convert to grid coordinates again
        if (point_coord_system != grid_coord_system)
        {
            auto new_points = ConvertCoordiantes(points, point_coord_system, grid_coord_system);
            points = new_points;
        }
    }

    /*
        Populate the last output with the last positions of the streamlines
    */



    auto CartesianizedOutput = ProjectFieldToCartesian(field, field_coord_system, grid, grid_coord_system, 
                                                        UniqueIndices);
                                                        
    auto Fcartesian = std::get<0>(CartesianizedOutput);
    auto x1d = std::get<1>(CartesianizedOutput);
    auto y1d = std::get<2>(CartesianizedOutput);
    auto z1d = std::get<3>(CartesianizedOutput);
    auto xtraj = std::get<4>(CartesianizedOutput);
    auto ytraj = std::get<5>(CartesianizedOutput);
    auto ztraj = std::get<6>(CartesianizedOutput);
    timers.EndTimer("Output");
    timers.PrintTimers(Nsteps);
    return py::make_tuple(streamline_output, Fcartesian, x1d, y1d, z1d, xtraj, ytraj, ztraj);
}

PYBIND11_MODULE(IntegrateStreamlines, m)
{
    m.doc() = "Streamline tracer module";
    m.def("IntegrateStreamlines", &IntegrateAllStreamlines<float>);
    m.def("IntegrateStreamlines", &IntegrateAllStreamlines<double>);
    py::arg("field"), py::arg("field_coord_system"),
    py::arg("gridx1"), py::arg("gridx2"), py::arg("gridx3"),
    py::arg("grid_coord_system"),
    py::arg("points"), py::arg("point_coord_system"),
    py::arg("use_adaptive_stepsize"),
    py::arg("Nsteps"), py::arg("Nout");
}
