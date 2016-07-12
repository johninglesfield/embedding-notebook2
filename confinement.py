# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 12:08:37 2016

@author: johninglesfield
"""
import numpy as np
from numpy import sqrt, sin, cos, pi, exp, trace
from scipy.integrate import quad, dblquad
from scipy.linalg import eig, solve
from scipy.special import jv, yn

def hamiltonian(param):
    """
    created Sunday 12 June 2016
    Hamiltonian, and eigenvalues and eigenvectors of 1D square well for
    embedding treatment of confinement. Note that this only gives even
    solutions.
    Input parameters are v, depth of well; d, width of well; D, parameter
    defining basis functions; n_max, number of basis functions.
    Returns eigen, list of eigenvalues and basis fn. coefficients.
    """
    global v,d,D,n_max
    v,d,D,n_max=param
    sigma=sqrt(0.5*v)
    ovr=np.zeros((n_max,n_max))
    kin=np.zeros((n_max,n_max))
    emb_vector=np.zeros(n_max)
    for m in range(0,n_max):
        emb_vector[m]=cos(m*pi*d/D)
        for n in range(0,n_max):
            (ovr[m,n],kin[m,n])=overlap(m,n)
    emb_matrix=np.outer(emb_vector,emb_vector)
    ham=kin+2.0*sigma*emb_matrix
    eig_val,eig_vector=eig(ham,ovr,left=False,right=True)
    eigen=[]
    for n in range(n_max):
        coeff=eig_vector[:,n].real
        norm=np.dot(ovr,coeff)
        norm=np.dot(coeff,norm)
        coeff=coeff/sqrt(norm)
        eigen.append((eig_val[n].real,coeff))
    eigen=sorted(eigen)
    return eigen
        
def overlap(m,n):
    """
    created Sunday 12 June 2016
    Calculates overlap and kinetic energy matrix el. for 1D square well.
    """
    if m==0 and n==0:
        overlap_int=d
        kinetic_en=0.0
    elif m==n:
        overlap_int=(pi*d/D+sin(2.0*m*pi*d/D)/(2.0*m))*D/(2.0*pi)
        kinetic_en=(pi*d/D-sin(2.0*m*pi*d/D)/(2.0*m))*m*m*pi/D
    else:
        overlap_int=(sin((m-n)*pi*d/D)/(m-n)+sin((m+n)*pi*d/D)/\
        (m+n))*D/(2.0*pi)
        kinetic_en=(sin((m-n)*pi*d/D)/(m-n)-sin((m+n)*pi*d/D)/\
        (m+n))*m*n*pi/D
    return overlap_int,kinetic_en
    
def wave_function(coeff,z):
    """
    created Sunday 12 June 2016
    Calculates wave-function at point z using basis fn. coefficients for
    1D square well.
    """
    psi=0.0
    for n in range(n_max):
        psi=psi+coeff[n]*cos(2.0*n*pi*z/D)
    return psi

def exact(n_psi,z):
    """
    created Sunday 12 June 2016
    Calculatex exact wave-function for infinite barrier 1D square-well, 
    or for v=0 exact wave-function with zero derivative boundary 
    conditions.
    """
    if v==0.0:
        if n_psi==0:
            normalisation=sqrt(1.0/d)
        else:
            normalisation=sqrt(2.0/d)
        psi_exact=cos(2.0*n_psi*pi*z/d)*normalisation
    else:
        psi_exact=cos((2.0*n_psi+1)*pi*z/d)*sqrt(2.0/d)
    return psi_exact
    
def vol_integrand(r,csth,alpha,beta,gamma,delta):
    """
    created Sunday 12 June 2016
    Confined H-atom: integrand for kinetic and potential energy
    matrix elements. 
    alpha and beta are the exponents for basis function m, and
    gamma and delta are the exponents for basis function n.
    We use r for \hat{r}, and csth for \cos(\hat{\theta}).
    r_kin=2\pi r^2 x 1/2\nabla_r u_m.\nabla_r u_n,
    th_kin=2\pi r^2 x 1/2 \nabla_th u_m.\nabla_th u_n,
    pot=2\pi r u_m u_n.
    The function returns fn, the integrand for the energy matrix el.
    """
    r_kin=pi*csth**(beta+delta)*(r**(alpha+gamma+2)-(alpha+gamma)*\
    r**(alpha+gamma+1)+alpha*gamma*r**(alpha+gamma))
    if beta==0 or delta==0:
        th_kin=0.0
    else:
        th_kin=pi*beta*delta*(1.0-csth*csth)*csth**(beta+delta-2)*\
        r**(alpha+gamma)
    pot=-2.0*pi*r**(alpha+gamma+1.0)*csth**(beta+delta)
    fn=exp(-2.0*r)*(r_kin+th_kin+pot)
    return fn
    
def ovlp_integrand(r,csth,alpha,beta,gamma,delta):
    """
    created Sunday 12 June 2016
    Confined H-atom: integrand for overlap.
    alpha and beta are the exponents for basis function m, and
    gamma and delta are the exponents for basis function n.
    We use r for \hat{r}, and csth for \cos(\hat{\theta}).
    ovlp (returned by the function)=2\pi r^2 u_m u_n.
    """
    ovlp=2.0*pi*csth**(beta+delta)*r**(alpha+gamma+2)*exp(-2.0*r)   
    return ovlp
    
def surf_integrand(csph,v,r0,a,alpha,beta,gamma,delta):
    """
    created Sunday 12 June 2016
    Confined H-atom: integrand for embedding contribution.
    alpha and beta are the exponents for basis function m, and
    gamma and delta are the exponents for basis function n.
    We use r for \hat{r}, and csth for \cos(\hat{\theta}), the arguments
    of the basis functions. These are calculated in terms of a 
    (displacement), r0 (sphere radius) and csph (equivalent to 
    \cos{\theta} in notes). sigma is the embedding potential, given in
    terms of v, the confining potential.
    The integrand is returned as fn, with fn=2\pi r_0^2 \Sigma u_m u_n.
    """
    sigma=sqrt(0.5*v)
    r=sqrt(a*a+r0*r0-2.0*a*r0*csph)
    csth=(r0*csph-a)/r
    fn=2.0*pi*r0*r0*sigma*exp(-2.0*r)*r**(alpha+gamma)*csth**(beta+delta)
    return fn

def ham_matrix(v,r0,a,m,n):
    """
    created Sunday 12 June 2016
    Confined H-atom: constructs embedded Hamiltonian matrix element.
    alpha and beta are the exponents for basis function m, and
    gamma and delta are the exponents for basis function n.
    a is the displacement of the H-atom in the sphere of radius r0.
    The first integral is the volume integral of the Hamiltonian, and
    the second integral is the surface integral of the embedding pot.
    The function returns the embedded Hamiltonian matrix element 
    between basis functions u_m and u_n. 
    """
    alpha=m[0]; beta=m[1]; gamma=n[0]; delta=n[1]
# Double integral, the inner integral over x=\cos(\hat{\theta}), 
# and the outer integral over \hat{r}. The lambda function gives the
# upper limit of \hat{r}, i.e. surface of the sphere.
    vol_int=dblquad(vol_integrand,-1.0,1.0,lambda x: 0,lambda x:\
    -a*x+sqrt(r0*r0-a*a*(1-x*x)), args=(alpha,beta,gamma,delta))
# Integral over surface
    surf_int=quad(surf_integrand,-1.0,1.0,args=(v,r0,a,alpha,beta,gamma,\
    delta))
# returns matrix element of embedded Hamiltonian
    return vol_int[0]+surf_int[0]
    
def overlap_matrix(r0,a,m,n):
    """
    created Sunday 12 June 2016
    Confined H-atom: constructs overlap matrix element.
    alpha and beta are the exponents for basis function m, and
    gamma and delta are the exponents for basis function n.
    a is the displacement of the H-atom in the sphere of radius r0.
    Returns the volume integral of the overlap between basis functions
    u_m and u_n.
    """
    alpha=m[0]; beta=m[1]; gamma=n[0]; delta=n[1]
# Double integrand, the inner integral over x=\cos(\hat{\theta}), 
# and the outer integral over \hat{r}. The lambda function gives the
# upper limit of \hat{r}, i.e. surface of the sphere.
    overlap=dblquad(ovlp_integrand,-1.0,1.0,lambda x: 0,lambda x:\
    -a*x+sqrt(r0*r0-a*a*(1-x*x)), args=(alpha,beta,gamma,delta))
# returns matrix element of overlap
    return overlap[0]    
    
def displaced_H_eigenvalues(param):
    """
    created Sunday 12 June 2016
    Confined H-atom: evaluates eigenvalues for embedding problem. 
    Sets up arrays for basis functions, i.e. alpha and beta exponents
    associated with a given m, then sets up matrix elements and 
    evaluates eigenvalues.
    Input parameters, v = confining potential, r0 = radius of sphere,
    a = displacement, nn determines number of basis functions (N in notes).
    alpha and beta both run from 0 to nn-1, so number of basis functions
    is nn^2.
    Function returns sorted eigenvalues.
    """
    v,r0,a,nn=param
    ab_array=np.zeros((nn,nn),dtype=int)
    for i in range(nn):
        ab_array[i,]=range(nn)
    al=ab_array.flatten(order='C')
    be=ab_array.flatten(order='F')
    ham=np.zeros((nn*nn,nn*nn))
    ovlp=np.zeros((nn*nn,nn*nn))
    for i in range(nn*nn):
        for j in range(nn*nn):
            m=[al[i],be[i]]
            n=[al[j],be[j]]
            ham[i,j]=ham_matrix(v,r0,a,m,n)
            ovlp[i,j]=overlap_matrix(r0,a,m,n)
    eigenvalues=eig(ham,ovlp)[0].real
    eigenvalues=sorted(eigenvalues)
    return eigenvalues
    
def quadrant_ovlp_int(theta,r,m,n):
    """    
    created Monday 11 July 2016
    Electron confined by quadrant: evaluates integrand for overlap
    integral for basis functions defined by m and n. Arguments theta, 
    r are circular polar coordinates.      
    """
    fac=pi/R
    x=r*cos(theta)
    y=r*sin(theta)
    chi_i=sin(fac*m[0]*x)*sin(fac*m[1]*y)
    chi_j=sin(fac*n[0]*x)*sin(fac*n[1]*y)
    return r*chi_i*chi_j
 
def quadrant_kinen_int(theta,r,m,n):
    """
    created Monday 11 July 2016
    Electron confined by quadrant: evaluates integrand for kinetic
    energy matrix element, between basis functions defined by m and n
    Arguments theta, r are circular polar coordinates.  
    """
    fac=pi/R
    x=r*cos(theta)
    y=r*sin(theta)
    grad_chi_i=np.array([m[0]*cos(fac*m[0]*x)*sin(fac*m[1]*y),\
    m[1]*sin(fac*m[0]*x)*cos(fac*m[1]*y)])
    grad_chi_j=np.array([n[0]*cos(fac*n[0]*x)*sin(fac*n[1]*y),\
    n[1]*sin(fac*n[0]*x)*cos(fac*n[1]*y)])
    return 0.5*r*fac*fac*np.dot(grad_chi_i,grad_chi_j)

def quadrant_matrices(param):
    """
    created Monday 11 July 2016
    Electron confined by quadrant: evaluates Hamiltonian and overlap
    integrals, and evaluates eigevalues. 
    Input parameters: V, confining potential; R, radius of quadrant;
    N, number of basis functions in each direction.
    Function returns sorted eigenvalues.
    """
    global V,R,N
    V,R,N=param
    sigma=sqrt(0.5*V)
    ab_array=np.zeros((N,N),dtype=int)
    for i in range(N):
        ab_array[i,]=range(1,N+1)
    al=ab_array.flatten(order='C')
    be=ab_array.flatten(order='F')
    ham=np.zeros((N*N,N*N))
    ovlp=np.zeros((N*N,N*N))
    for i in range(N*N):
        m=[al[i],be[i]]
        for j in range(N*N):
            n=[al[j],be[j]]
            ovlp[i,j]=dblquad(quadrant_ovlp_int,0.0,R,lambda x:0.0,\
            lambda x:0.5*pi,args=(m,n))[0]
            kinen=dblquad(quadrant_kinen_int,0.0,R,lambda x:0.0,\
            lambda x:0.5*pi,args=(m,n))[0]
            confine=sigma*quad(quadrant_ovlp_int,0.0,0.5*pi,args=(R,m,n))[0]
            ham[i,j]=kinen+confine
        if i%5==0:
            print '%s %3d %s %3d' % ('Number of integrals done =',i+1,\
            'x',N*N)
    print '%s %3d %s %3d' % ('Number of integrals done =',i+1,'x',N*N)   
    eigen=eig(ham,ovlp)[0].real
    eigen=sorted(eigen)
    return eigen
    
def corner_matrices(param):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates Hamiltonian and 
    overlap integrals.
    Input parameters: V, confining potential; r1, inner radius; r2, outer
    radius; D, defines basis functions; N, number of basis functions in
    each direction.
    Function returns Hamiltonian and overlap matrices, and al, be, which
    define basis functions.
    """
    global V,r1,r2,D,N
    V,r1,r2,D,N=param
    sigma=sqrt(0.5*V)
    ab_array=np.zeros((N,N),dtype=int)
    for i in range(N):
        ab_array[i,]=range(N)
    al=ab_array.flatten(order='C')
    be=ab_array.flatten(order='F')
    ham=np.zeros((N*N,N*N))
    ovlp=np.zeros((N*N,N*N))
    for i in range(N*N):
        m=[al[i],be[i]]
        for j in range(N*N):
            n=[al[j],be[j]]
            ovlp[i,j]=dblquad(corner_ovlp_int,r1,r2,lambda x:0.0,\
            lambda x:0.5*pi,args=(m,n))[0]
            kinen=dblquad(corner_kinen_int,r1,r2,lambda x:0.0,\
            lambda x:0.5*pi,args=(m,n))[0]
            confine=sigma*(quad(corner_ovlp_int,0.0,0.5*pi,\
            args=(r1,m,n))[0]+quad(corner_ovlp_int,0.0,0.5*pi,\
            args=(r2,m,n))[0])
            ham[i,j]=kinen+confine
        if i%5==0:
            print '%s %3d %s %3d' % ('Number of integrals done =',i+1,\
            'x',N*N)
    print '%s %3d %s %3d' % ('Number of integrals done =',i+1,'x',N*N)
    return ham,ovlp,al,be
    
def corner_ovlp_int(theta,r,m,n):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates integrand for overlap
    integral for basis functions defined by m and n. Arguments theta, 
    r are circular polar coordinates. 
    """
    fac=pi/D
    x=r*cos(theta)+0.5*(D-r2)
    y=r*sin(theta)+0.5*(D-r2)
    chi_i=cos(fac*m[0]*x)*cos(fac*m[1]*y)
    chi_j=cos(fac*n[0]*x)*cos(fac*n[1]*y)
    return r*chi_i*chi_j
    
def corner_kinen_int(theta,r,m,n):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates integrand for kinetic
    energy matrix element, between basis functions defined by m and n
    Arguments theta, r are circular polar coordinates. 
    """
    fac=pi/D
    x=r*cos(theta)+0.5*(D-r2)
    y=r*sin(theta)+0.5*(D-r2)
    grad_chi_i=np.array([-m[0]*sin(fac*m[0]*x)*cos(fac*m[1]*y),\
    -m[1]*cos(fac*m[0]*x)*sin(fac*m[1]*y)])
    grad_chi_j=np.array([-n[0]*sin(fac*n[0]*x)*cos(fac*n[1]*y),\
    -n[1]*cos(fac*n[0]*x)*sin(fac*n[1]*y)])
    return 0.5*r*fac*fac*np.dot(grad_chi_i,grad_chi_j)
      
def corner_eigenvalues(ham,ovlp):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates sorted eigenvalues 
    and eigenvectors. 
    """
    eig_value,eig_vector=eig(ham,ovlp,left=False,right=True)
    eigen=[]
    for i in range(N*N):
        coeff=eig_vector[:,i].real
        norm=np.dot(ovlp,coeff)
        norm=np.dot(coeff,norm)
        coeff=coeff/sqrt(norm)
        eigen.append((eig_value[i].real,coeff))
    eigen=sorted(eigen)
    return eigen
    
def corner_exact(k,m):
    """
    created Monday 11 July 2016
    Function which vanishes at eigenenergies of circular corner.
    """
    psi=jv(m,r1*k)-jv(m,r2*k)*yn(m,r1*k)/yn(m,r2*k)
    return psi
    
def corner_plot(xp,yp,coeff,xfac,yfac):
    """
    created Monday 11 July 2016
    Electron confined to circular corner: constructs wave-function, given
    coefficients of basis functions.
    """
    x=xp+0.5*(D-r2)
    y=yp+0.5*(D-r2)
    psi=0.0
    for i in range(N*N):
        psi=psi+coeff[i]*cos(xfac[i]*x)*cos(yfac[i]*y)
    return psi
    
def corner_embed_matrices(n_emb):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates integral of waveguide
    function x part of basis function.
    """
    emb_int=np.zeros((n_emb,N))
    for p in range(n_emb):
        for m in range(N):    
            emb_int[p,m]=quad(corner_embed_int,r1,r2,args=(p+1,m))[0]
    return emb_int
            
def corner_embed_int(r,p,m):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: waveguide function x part of
    basis function.
    """
    w=r2-r1
    x=r+0.5*(D-r2)
    xp=r-r1
    return sin(p*pi*xp/w)*cos(m*pi*x/D)    
       

def corner_green(energy,eta,emb_int,ham,ovlp,n_emb,al,be): 
    """
    created Monday 11 July 2016
    Electron confined by circular corner: constructs Green function of
    circular corner embedded onto straight waveguides. Returns corner
    density of states, evaluated at (energy + j eta).
    """
    x=0.5*(D-r2)
    w=r2-r1
    en=complex(energy,eta)
    embed=np.zeros((N*N,N*N),complex)
    cs=[cos(pi*i*x/D) for i in range(N)]
    sigma=np.zeros((n_emb),complex)
    for p in range(n_emb):
        sigma[p]=complex(0.0,-1.0)*sqrt(2.0*en-((p+1)*pi/w)**2)/w
    for i in range(N*N):
        for j in range(N*N):
            fac_al=cs[al[i]]*cs[al[j]]
            fac_be=cs[be[i]]*cs[be[j]]
            for p in range(n_emb):
                embed[i,j]=embed[i,j]+(fac_al*emb_int[p,be[i]]*\
                emb_int[p,be[j]]+fac_be*emb_int[p,al[i]]*\
                emb_int[p,al[j]])*sigma[p]
    green=ham+embed-en*ovlp
    lds=trace(solve(green,ovlp)).imag/pi
    return lds