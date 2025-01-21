import numpy as np
import matplotlib.pyplot as plt
import IntegrateStreamlines as IS

# Create a dipolar field in 3D
def dipolar_field(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    Bx = 3*x*z/r**5
    By = 3*y*z/r**5
    Bz = (3*z**2 - r**2)/r**5
    return Bx, By, Bz

# Create arrays
x = np.linspace(-1, 1, 100, dtype=np.float32)
y = np.linspace(-1, 1, 100, dtype=np.float32)
z = np.linspace(-1, 1, 100, dtype=np.float32)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
Bx, By, Bz = dipolar_field(X, Y, Z)
theta_min = 0.1
Nout = 100
Nsteps = 10000
# Pick a streamline to trace


output = IS.IntegrateStreamlines(Bx, By, Bz,
                        x, y, z,
                        np.array([0.1]), np.array([0.0]), np.array([0.]),
                        Nsteps, Nout,
                        x[1], x[-2],
                        y[1], y[-2],
                        z[1], z[-2],
                        theta_min, "cartesian")

output_x, output_y, output_z = output[...,0]

#print(output_x)
#print(output_y)
#print(output_z)
# Plot the field in xz plane
plt.figure()
plt.streamplot(X[:,50,:].T, Z[:,50,:].T, Bx[:,50,:].T, Bz[:,50,:].T)
plt.plot(output_x, output_z, 'ro')
print(output_x)
print(output_z)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Dipolar field in xz plane')
plt.show()
plt.savefig('streamline_test.png')
