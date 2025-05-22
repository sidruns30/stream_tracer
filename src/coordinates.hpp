/*
    * coordinates.hpp
    *
    * Code to convert between different coordinate systems for vectors and positions
*/
#ifndef COORDINATES_HPP_
    #define COORDINATES_HPP_

    #include "global.hpp"
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    namespace py = pybind11;
    
    namespace Coordinates
    {
        // Same functions as above but for a single point
        template <typename T>
        void SphericalToCartesianPoint( const T &r,
                                        const T &theta,
                                        const T &phi,
                                        T &x,
                                        T &y,
                                        T &z)
        {
            x = r * sin(theta) * cos(phi);
            y = r * sin(theta) * sin(phi);
            z = r * cos(theta);
            return;
        }

        template <typename T>
        void CartesianToSphericalPoint( const T &x,
                                        const T &y,
                                        const T &z,
                                        T &r,
                                        T &theta,
                                        T &phi)
        {
            r = sqrt(square(x) + square(y) + square(z));
            theta = acos(z / r);
            phi = atan2(y, x);
            if (phi > phi_max)
            {   phi = phi_min + (phi - phi_max);}
            else if (phi < phi_min)
            {   phi = phi_max + (phi - phi_min);}
            return;
        }

        template <typename T>
        void SphericalToLogSphericalPoint(  const T &r,
                                            const T &theta,
                                            const T &phi,
                                            T &lnr,
                                            T &theta_out,
                                            T &phi_out)
        {
            lnr = log(r);
            theta_out = theta;
            phi_out = phi;
            return;
        }

        template <typename T>
        void LogSphericalToSphericalPoint(  const T &lnr,
                                            const T &theta,
                                            const T &phi,
                                            T &r,
                                            T &theta_out,
                                            T &phi_out)
        {
            r = exp(lnr);
            theta_out = theta;
            phi_out = phi;
            return;
        }

        template <typename T>
        void LogSphericalToCartesianPoint(  const T &lnr,
                                            const T &theta,
                                            const T &phi,
                                            T &x,
                                            T &y,
                                            T &z)
        {
            x = exp(lnr) * sin(theta) * cos(phi);
            y = exp(lnr) * sin(theta) * sin(phi);
            z = exp(lnr) * cos(theta);
            return;
        }

        template <typename T>
        void CartesianToLogSphericalPoint(  const T &x,
                                            const T &y,
                                            const T &z,
                                            T &lnr,
                                            T &theta,
                                            T &phi)
        {
            lnr = log(sqrt(square(x) + square(y) + square(z)));
            theta = acos(z / exp(lnr));
            phi = atan2(y, x);
            if (phi > phi_max)
            {   phi = phi_min + (phi - phi_max);}
            else if (phi < phi_min)
            {   phi = phi_max + (phi - phi_min);}
            return;
        }

        template <typename T>
        void CopyPoint( const T &fromx1,
                        const T &fromx2,
                        const T &fromx3,
                        T &tox1,
                        T &tox2,
                        T &tox3)
        {
            tox1 = fromx1;
            tox2 = fromx2;
            tox3 = fromx3;
            return;
        }
    }

#endif /* COORDINATES_HPP_ */