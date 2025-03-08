#ifndef PAYLOADS_HPP_
    #define PAYLOADS_HPP_

    #include "global.hpp"

    template <typename T>
    T CustomUserOperation(std::size_t ix1, std::size_t ix2, std::size_t ix3,
                            T x1, T x2, T x3, T dx1, T dx2, T dx3,
                            T Fx, T Fy, T Fz,
                            std::vector<std::string> &payload_names,
                            std::vector<py::array_t<T>> &payload_arrays)
        {
            // Compute E dot B
            auto Ex = payload_arrays[0].template unchecked<3>()(ix1, ix2, ix3);
            auto Ey = payload_arrays[1].template unchecked<3>()(ix1, ix2, ix3);
            auto Ez = payload_arrays[2].template unchecked<3>()(ix1, ix2, ix3);

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