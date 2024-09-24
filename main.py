import numpy as np
import scipy.special as sp
import scipy.constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math  

hbar = const.hbar  
m_e = const.m_e    
e = const.e        
epsilon_0 = const.epsilon_0  

a0 = 4 * np.pi * epsilon_0 * hbar**2 / (m_e * e**2)

def radial_wavefunction(n, l, r):
    
    
    rho = 2 * r / (n * a0)
    normalization = np.sqrt((2 / (n * a0))**3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))
    
    laguerre_poly = sp.genlaguerre(n - l - 1, 2 * l + 1)(rho)
    Rnl = normalization * np.exp(-rho / 2) * rho**l * laguerre_poly
    return Rnl

def radial_probability_density(n, l, r):
    
    Rnl = radial_wavefunction(n, l, r)
    P_r = r**2 * np.abs(Rnl)**2
    return P_r

def spherical_harmonic(l, m, theta, phi):
    
    return sp.sph_harm(m, l, phi, theta)

def plot_radial_wavefunction(n, l):
    
    r = np.linspace(0, 20 * a0, 1000) 
    Rnl = radial_wavefunction(n, l, r)
    
    plt.plot(r / a0, Rnl, label=f'n={n}, l={l}')
    plt.title(f'Radial Wavefunction for Hydrogen Atom (n={n}, l={l})')
    plt.xlabel('Radius (Bohr radius units)')
    plt.ylabel('R(r)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_radial_probability_density(n, l):
    
    r = np.linspace(0, 20 * a0, 1000)  
    P_r = radial_probability_density(n, l, r)
    
    plt.plot(r / a0, P_r, label=f'n={n}, l={l}')
    plt.title(f'Radial Probability Density for Hydrogen Atom (n={n}, l={l})')
    plt.xlabel('Radius (Bohr radius units)')
    plt.ylabel('P(r)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_spherical_harmonic(l, m):
    
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    
    Y_lm = np.abs(spherical_harmonic(l, m, theta, phi))**2

    x = Y_lm * np.sin(theta) * np.cos(phi)
    y = Y_lm * np.sin(theta) * np.sin(phi)
    z = Y_lm * np.cos(theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    plt.title(f'Spherical Harmonic (l={l}, m={m})')
    plt.show()

def main():
    n = int(input("Enter the principal quantum number (n): "))
    l = int(input("Enter the azimuthal quantum number (l): "))

    if l >= n or l < 0:
        print("Invalid input: 'l' must be in the range 0 ≤ l < n.")
        return

    plot_radial_wavefunction(n, l)

    plot_radial_probability_density(n, l)

    for m in range(-l, l + 1):
        plot_spherical_harmonic(l, m)

main()
