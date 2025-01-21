/*
    * coordinates.hpp
    *
    * Code to convert between different coordinate systems for vectors and positions
*/
#ifndef COORDINATES_HPP_
    #define COORDINATES_HPP_

    #include <omp.h>
    #include <vector>

    template <typename T>
    void SphericalToCartesianCoords(std::vector<T> &r, 
                                    std::vector<T> &theta,
                                    std::vector<T> &phi,
                                    std::vector<T> &x,
                                    std::vector<T> &y,
                                    std::vector<T> &z)
    {
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<r.size(); i++)
        {
            x[i] = r[i] * sin(theta[i]) * cos(phi[i]);
            y[i] = r[i] * sin(theta[i]) * sin(phi[i]);
            z[i] = r[i] * cos(theta[i]);
        }
        return;
    }

    template <typename T>
    void CartesianToSphericalCoords(std::vector<T> &x,
                                    std::vector<T> &y,
                                    std::vector<T> &z,
                                    std::vector<T> &r,
                                    std::vector<T> &theta,
                                    std::vector<T> &phi)
    {
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<r.size(); i++)
        {
            r[i] = sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            theta[i] = acos(z[i] / r[i]);
            phi[i] = atan2(y[i], x[i]);
        }
        return;
    }

    template <typename T>
    void SphericalToCartesianField(std::vector<T> &r, 
                                    std::vector<T> &theta,
                                    std::vector<T> &phi,
                                    std::vector<T> &Br,
                                    std::vector<T> &Btheta,
                                    std::vector<T> &Bphi,
                                    std::vector<T> &Bx,
                                    std::vector<T> &By,
                                    std::vector<T> &Bz)
    {
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<r.size(); i++)
        {
            Bx[i] = Br[i] * sin(theta[i]) * cos(phi[i]) + 
                    Btheta[i] * cos(theta[i]) * cos(phi[i]) - 
                    Bphi[i] * sin(phi[i]);
            By[i] = Br[i] * sin(theta[i]) * sin(phi[i]) + 
                    Btheta[i] * cos(theta[i]) * sin(phi[i]) + 
                    Bphi[i] * cos(phi[i]);
            Bz[i] = Br[i] * cos(theta[i]) - Btheta[i] * sin(theta[i]);
        }
        return;
    }

    template <typename T>
    void CartesianToSphericalField(std::vector<T> &x,
                                    std::vector<T> &y,
                                    std::vector<T> &z,
                                    std::vector<T> &Bx,
                                    std::vector<T> &By,
                                    std::vector<T> &Bz,
                                    std::vector<T> &Br,
                                    std::vector<T> &Btheta,
                                    std::vector<T> &Bphi)
    {
        #pragma omp parallel for schedule(static)
        for (std::size_t i=0; i<x.size(); i++)
        {
            Br[i] = x[i] * Bx[i] + y[i] * By[i] + z[i] * Bz[i];
            Btheta[i] = (x[i] * Bx[i] + y[i] * By[i] + z[i] * Bz[i]) / sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
            Bphi[i] = (x[i] * By[i] - y[i] * Bx[i]) / (x[i]*x[i] + y[i]*y[i]);
        }
        return;
    }


#endif /* COORDINATES_HPP_ */