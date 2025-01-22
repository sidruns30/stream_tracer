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
y = np.linspace(-0.01, 0.01, 100, dtype=np.float32)
z = np.linspace(-1, 1, 100, dtype=np.float32)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
Bx, By, Bz = dipolar_field(X, Y, Z)
theta_min = 0.1
Nout = 100
Nsteps = 10000

# Generate random points and sample some coordinates
Npoints = 10000
x_points = np.random.uniform(x[1], x[-2], Npoints)
y_points = np.random.uniform(-0.01, 0.01, Npoints)
z_points = np.random.uniform(z[1], z[-2], Npoints)

# Pick a streamline to trace


output = IS.IntegrateStreamlines(Bx, By, Bz,
                        x, y, z,
                        x_points, y_points, z_points,
                        Nsteps, Nout,
                        x[1], x[-2],
                        y[1], y[-2],
                        z[1], z[-2],
                        theta_min, "cartesian")

output_x, output_y, output_z = output[...,:]

#print(output_x)
#print(output_y)
#print(output_z)
# Plot the field in xz plane
plt.figure()
plt.streamplot(X[:,50,:].T, Z[:,50,:].T, Bx[:,50,:].T, Bz[:,50,:].T)
for i in range(Npoints):
    plt.plot(output_x[:,i], output_z[:,i], 'r.')
    plt.plot(output_x[0,i], output_z[0,i], 'bo')

plt.xlabel('x')
plt.ylabel('z')
plt.title('Dipolar field in xz plane')
#plt.show()
plt.savefig('streamline_test.png')
