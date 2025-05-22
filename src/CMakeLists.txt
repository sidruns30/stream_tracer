set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add the header files into a variable
set(HEADERS grid.hpp interpolation.hpp display.hpp coordinates.hpp payloads.hpp integration.hpp)

# Bind them using pybind
pybind11_add_module(IntegrateStreamlines integrate_streamlines.cpp ${HEADERS})

target_link_libraries(IntegrateStreamlines PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS IntegrateStreamlines DESTINATION .)

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O0 -g -fsanitize=address -march=native -fopenmp -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp -fPIC")