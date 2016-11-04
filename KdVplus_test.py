import numpy as np;
import matplotlib.pyplot as plt;
from numpy.fft import *
from numpy import *

#for nice output 
plt.rcParams['lines.linewidth']=2.5
plt.rcParams['axes.titlesize']=15

#grid parameters - create grid for computation
N=256; #number of steps
tau=1/N**2;  #time step
T = 1.13; #time of computation
a, b = 0, 2; #area 
nmax = round(T/tau);
x = np.arange(a, b, (b-a)/N);
inc, dec = np.arange(0, N/2), np.arange(-N/2, 0)
kx =  2*np.pi/(b-a) *np.concatenate([inc, dec])
#k = (2*np.pi/(b-a) ) * np.concatenate((np.arange(0,N/2+1),\
#	np.array([0]), np.arange(-N/2+1,0));

j=1; count=0;  #j - number of picture
Z = 200; # Numbers of time layers in which we will plot the pictures
nplt = floor( nmax/Z);

#parameters of equation
beta, t, delta = 1, 0, 0.022;
L = -delta**2*(1j*kx)**3 - 2/5*delta**4*(1j*kx)**5; #linear operator for 1st eq (in Fourier space)
E1, E2 = np.exp(tau*L/2), np.exp(tau*L); #to speed up calculation
L2 = -delta**2*(1j*kx)**3;	# linear operator for 2nd eq
E21, E22 = np.exp(tau*L2/2), np.exp(tau*L2); #to speed up calculation

#initial conditions
print("start")
plt.ion() # enables interactive mode
u = np.cos(np.pi*x)
plt.subplot(2, 2, j)
plt.plot(x, u, 'k')
plt.title('t=0')
plt.xlabel('x')
plt.ylabel('u', fontsize=14, rotation=0)
plt.draw()


v = v2 = np.fft.fft(u);

#filter (to reach stable computation)
#xs = []
#xs.append(1)
rho = np.zeros( (1, N) );
rho2 = (2/3)*(2*np.pi)/(b-a)
for l in range(0,N-1):
        rho1 = 2*kx[l]/N
        if rho1 < rho2 and rho1 > - rho2:
                rho[0,l]=1
        else:
                rho[0,l]=0
p1 = -tau*1/2*1j*kx*rho;
p21 = -tau*delta**2*(1j*kx)*rho;
p22 = 1j*kx*rho;
p31 = -tau*delta**2*rho;
p32 = (1j*kx)**3*rho;
p41 = -tau*(2/5)*delta**4*rho;
p42 = (1j*kx)**5*rho;
p51 = -tau*delta**4*(1j*kx)*rho;
p52 = (1j*kx)**2*rho;
p61 = -tau*6/5*delta**4*rho;
p62 = (1j*kx)*rho;
p63 = (1j*kx)**4*rho;


#main cycle
for n in range (1, nmax):
	t = n*tau

	#Runge-Kutta 4th order
	#5th order equation
	a1 =  p1 *np.fft.fft (real(ifft( v ))**2) + p21*np.fft.fft( real(ifft(p22*v))**2)\
            + p31* np.fft.fft( real(ifft(v)) * real(ifft(p32*v)) );
	a2 =  p1 *np.fft.fft (real(ifft( (v+a1/2)*E1 ))**2) + p21*np.fft.fft( real(ifft(p22*(v+a1/2)*E1))**2)\
            + p31* np.fft.fft( real(ifft((v+a1/2)*E1)) * real(ifft(p32*(v+a1/2)*E1)) );
	a3 =  p1 *np.fft.fft (real(ifft( ( E1*v + a2/2 ) ))**2) + p21*np.fft.fft( real(ifft(p22*( E1*v + a2/2 )))**2)\
            + p31* np.fft.fft( real(ifft(( E1*v + a2/2 ))) * real(ifft(p32*( E1*v + a2/2 ))) );
	a4 =  p1 *np.fft.fft (real(ifft( (E2*v+a3*E1) ))**2) + p21*np.fft.fft( real(ifft(p22*(E2*v+a3*E1)))**2)\
            + p31* np.fft.fft( real(ifft((E2*v+a3*E1))) * real(ifft(p32*(E2*v+a3*E1))) );
	v= E2*v + (E2*a1 + 2*E1*(a2 +a3) + a4)/6;

        #KdV
	#a12 = p1 *np.fft.fft( real(ifft( v2 )**2) );
	#a22 = p1 *np.fft.fft( real(ifft( (v2+a12/2)*E21))**2);
	#a32 = p1 *np.fft.fft( real(ifft ( E21*v2 + a22/2 ))**2);
	#a42 = p1 *np.fft.fft( real(ifft ( (v2*E22+ a32*E21) ))**2);
	#v2 = E22*v2 + (E22*a12 + 2*E21*(a22 +a32) + a42)/6;
	
	# inverse
	if mod(n, nplt) == 0 :
		u, u2 = real(ifft(v)), real(ifft(v2))
		count = count+1

		if count == 50 or count == 100 or count ==200:
			j = j+1
			print(j)
			#print("draw {}".format(n))
			#line.set_ydata(u[0])
			#plt.draw()
			#plt.pause(0.001)
			#plt.figure()
			plt.subplot(2, 2, j)
			plt.plot(x, u[0], 'k')
			plt.title('t='+str(t)[0:4])
			plt.xlabel('x')
			plt.ylabel('u', fontsize=14, rotation=0)
			plt.show(block=False)						

plt.show()
