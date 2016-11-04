# PDE with spectral method

In my science work i have derived the fifth-order partial differential equation for the description of the Fermi-Pasta-Ulam model. I prefer to use **Matlab** for the numerical simulation. Now i decided to use **numpy**. 

Simulation of the high-order PDE is easier with the spectral method. The idea is to use the Fourier transform to reduce it to ODE. That is what my program do. The algorithm is:

1. Create a spatial-time greed 
