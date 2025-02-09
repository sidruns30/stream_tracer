set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add the header files into a variable
set(HEADERS grid.hpp interpolation.hpp display.hpp coordinates.hpp adaptive_stepsize.hpp indices.hpp)

# Bind them using pybind
pybind11_add_module(IntegrateStreamlines integrate_streamlines.cpp ${HEADERS})

target_link_libraries(IntegrateStreamlines PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS IntegrateStreamlines DESTINATION .)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -fopenmp -fPIC")

#add_library(grid_lib grid.hpp)
#add_library(field_lib fields.hpp)
#add_library(integrate_streamlines_lib integrate_streamlines.cpp)
#add_executable(main main.cpp)
#target_link_libraries(main integrate_streamlines_lib)

#add_executable(main main.cpp force_free.cpp)
#target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

# Command to compile with openmp
# cmake . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DUSE_WERROR=ON