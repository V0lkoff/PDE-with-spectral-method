# PDE with spectral method

In my science work i have derived the fifth-order partial differential equation for the description of the Fermi-Pasta-Ulam model. I prefer to use **Matlab** for the numerical simulation. Now i decided to use **numpy**. 

Simulation of the high-order PDE is easier with the spectral method. The idea is to use the Fourier transform to reduce it to ODE. That is what my program do. The algorithm is:

1. Create a spatial-time grid
2. Prepare linear and non-linear Fourier operators (to speedup computation)
3. Prepare filter to provide stability
4. In Fourier space solve the derived ODE
5. Use the inverse Fourier transform to get the result

The output is plots of solution for different time moments.

![Alt-the example of the output file](https://github.com/V0lkoff/PDE-with-spectral-method/blob/master/example.png)

There are a lot of commented commands. The reason is that the script was used for the investigation of wave processes.
