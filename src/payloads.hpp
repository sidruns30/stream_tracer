#ifndef PAYLOADS_HPP_
    #define PAYLOADS_HPP_

    #include "global.hpp"

    template <typename T>
    T CustomUserOperation(std::size_t ix1, std::size_t ix2, std::size_t ix3,
                            T x1, T x2, T x3, T dx1, T dx2, T dx3,
                            T Fx, T Fy, T Fz,
                            const std::string &grid_coord_system,
                            const std::vector<std::string> &payload_names,
                            const std::vector<py::array_t<T>> &payload_arrays)
        {
            // Find indices of the payload arrays Ex, Ey, Ez
            std::size_t Ex_index, Ey_index, Ez_index;
            for (std::size_t i=0; i<payload_names.size(); i++)
            {
                if (payload_names[i] == "Ex")       {   Ex_index = i; }
                else if (payload_names[i] == "Ey")  {   Ey_index = i; }
                else if (payload_names[i] == "Ez")  {   Ez_index = i; }
            }

            // Compute the displacement vector in cartesian coordinates
            T dx, dy, dz;
            if (grid_coord_system == "cartesian")
            { 
                dx = dx1;
                dy = dx2;
                dz = dx3;
            }
            else if (grid_coord_system == "spherical")
            {
                dx = dx1 * sin(x2) * cos(x3);
                dy = dx1 * sin(x2) * sin(x3);
                dz = dx1 * cos(x2);
            }
            else if (grid_coord_system == "log_spherical")
            {
                dx = dx1 * exp(x2) * sin(x3) * cos(x4);
                dy = dx1 * exp(x2) * sin(x3) * sin(x4);
                dz = dx1 * exp(x2) * cos(x3);
            }




            // Compute E dot B
            auto Ex = payload_arrays[Ex_index].template unchecked<3>()(ix1, ix2, ix3);
            auto Ey = payload_arrays[Ey_index].template unchecked<3>()(ix1, ix2, ix3);
            auto Ez = payload_arrays[Ez_index].template unchecked<3>()(ix1, ix2, ix3);

            // Raise error if nan exists
            if (std::isnan(Ex) || std::isnan(Ey) || std::isnan(Ez) || 
                std::isnan(Fx) || std::isnan(Fy) || std::isnan(Fz))
            {
                std::cout << "Nan encountered in the field" << std::endl;
                std::cout << "Ex: " << Ex << " Ey: " << Ey << " Ez: " << Ez << std::endl;
                std::cout << "ix1: " << ix1 << " ix2: " << ix2 << " ix3: " << ix3 << std::endl;
                std::cout << "x1: " << x1 << " x2: " << x2 << " x3: " << x3 << std::endl;
                std::cout << "dx1: " << dx1 << " dx2: " << dx2 << " dx3: " << dx3 << std::endl;
                std::cout << "Fx: " << Fx << " Fy: " << Fy << " Fz: " << Fz << std::endl;

                throw std::runtime_error("Nan encountered in the field");
            }

            return Ex*Fx + Ey*Fy + Ez*Fz;
        }

#endif /* PAYLOADS_HPP_ */