"""
Implement an efficient numerical evaluation of the impurity-bath ground-state correlations 
along spatial directions (1, 0) and (1, 1), as well as the bath-bath ground-state correlations, 
using a fast Fourier transform.
"""
module GroundStateCorrelations2D
export impurity_correlations_10, impurity_correlations_11, bath_correlations

include("./resolvent/self-energies.jl")
using QuadGK, FFTW

"""
    dist, corr = impurity_correlations_10(Delta, g, Ef, M)

Compute impurity-bath ground-state correlations along spatial direction (1, 0) at 
various distances from the impurity.
    
#### Arguments
- `Delta`: impurity on-site energy.
- `g`: impurity-bath coupling strength.
- `Ef`: Fermi level (function is valid only for Ef < 4).
- `M`: number of points (2M) in momentum space for the FFT.

#### Outputs
- `dist`: distances from the impurity at which correlations are evaluated.
- `corr`: impurity-bath correlations.
"""
function impurity_correlations_10(Delta, g, Ef, M)

    integration_limit(k) = acos(2 - cos(k) - Ef/2)

    function integrand1(k, q)
        w = 2*(2 - cos(k) - cos(q))
        return real(1/(w - Delta - Sig2DI(w - 1e-12im, g)))
    end

    function ref1(k, kf)    # real part of f₁
        if abs(k) > kf
            return 0.0
        else
            l = integration_limit(k)
            val, _ = quadgk(q -> integrand1(k, q), -l, l)
            return val/(2*pi)
        end
    end

    function integrand2(k, w)
        sigma = Sig2DI(w + 1e-12im, g)
        dos = -imag(sigma)
        b = w + 1e-12im + 2*cos(k) - 4
        return -real(dos/(abs2(w - Delta - sigma)*sqrt(b^2 - 4)))
    end
    
    function ref2(k, kf)    # real part of f₂
        if abs(k) > kf
            val, _ = quadgk(w -> integrand2(k, w), 0, Ef)
        else
            val, _ = quadgk(w -> integrand2(k, w), 0, 2 - 2*cos(k))
        end
        return val/pi
    end

    k = pi/M * (0:M)                    # discrete momenta
    kf = integration_limit(0)
    y = ref1.(k, kf) + ref2.(k, kf)     # momentum-space integrand of 1D (inverse) FFT
    y = ifft([y; y[(end - 1):-1:2]])    # perform inverse FFT
    # since the function to be transformed is symmetric, we repeat (don´t have to compute again) 
    # half of the values. Actually, we could have used a discrete cosine transform exploiting
    # this symmetry, but we're lazy.
    return 0:(2*M - 1), y
end

"""
    dist, corr = impurity_correlations_11(Delta, g, Ef, M)

Compute impurity-bath ground-state correlations along spatial direction (1, 1) at 
various distances from the impurity.
    
#### Arguments
- `Delta`: impurity on-site energy.
- `g`: impurity-bath coupling strength.
- `Ef`: Fermi level (function is valid only for Ef < 4).
- `M`: number of points (2M) in momentum space for the FFT.

#### Outputs
- `dist`: distances from the impurity at which correlations are evaluated.
- `corr`: impurity-bath correlations.
"""
function impurity_correlations_11(Delta, g, Ef, M)

    integration_limit(k) = acos(sec(k)*(1 - Ef/4))

    function integrand1(k, q)
        w = 4*(1 - cos(k)*cos(q))
        return real(1/(w - Delta - Sig2DI(w - 1e-12im, g)))
    end

    function ref1(k, kf)
        if k > kf
            return 0.0 
        else
            l = integration_limit(k)
            val, _ = quadgk(q -> integrand1(k, q), -l, l)
            return val/(2*pi)
        end
    end

    function integrand2(k, w)
        sigma = Sig2DI(w + 1e-12im, g)
        dos = -imag(sigma)
        ck2 = 2*cos(k)
        b = (w + 1e-12im - 4)/ck2
        return -real(dos/(abs2(w - Delta - sigma)*ck2*sqrt(b^2 - 4)))
    end

    function ref2(k, kf)
        if k > kf
            val, _ = quadgk(w -> integrand2(k, w), 0, Ef)
        else
            val, _ = quadgk(w -> integrand2(k, w), 0, 4 - 4*cos(k))
        end
        return val/pi
    end

    k = pi/M * (0:M)                       # discrete momenta for 1D FFT
    kf = integration_limit(0)                 
    y = ref1.(k/2, kf) + ref2.(k/2, kf)    # momentum-space integrand of 1D (inverse) FFT
    y = ifft([y; y[(end - 1):-1:2]])       # perform inverse FFT
    # since the function to be transformed is symmetric, we repeat (don´t have to compute again) 
    # half of the values. Actually, we could have used a discrete cosine transform exploiting
    # this symmetry, but we're lazy.
    return sqrt(2)*(0:(2*M - 1)), y
end

"""
    symmetric_indices(i, j, M)

List all indices labelling matrix elements of a `2M` by `2M` matrix related to the indices 
`(i, j)` by horizontal and vertical reflections and transposition.
"""
function symmetric_indices(i, j, M)
    a, b = map(x -> mod(2*M - x + 1, 2*M) + 1, [i, j])
    inds = [(i, j), (i, b), (a, j), (a, b)]
    return CartesianIndex.([inds; reverse.(inds)])
end

"""
    sigma(z, k, r)

Partial integration of the self-energy function.
"""
function sigma(z, k, r)
    b = z + 2*cos(k) - 4
    s = sign(real(b))*sqrt(b^2 - 4)
    root = (-b + s)/2
    return root^abs(r)/s
end

"""
    corr = bath_correlations(Delta, g, Ef, M)

Compute the difference between the bath-bath correlations in the ground state with and 
without the impurity.

#### Arguments
- `Delta`: impurity on-site energy
- `g`: impurity-bath coupling strength.
- `Ef`: Fermi level.
- `M`: number of points for the FFT (2M x 2M).

#### Outputs
- `corr`: bath-bath correlations (deviation from impurity-free case).
"""
function bath_correlations(Delta, g, Ef, M)
    
    f1(w) = Sig2DI(w - 1e-12im, g) * G2DI(w - 1e-12im, Delta, g) * (w < Ef)
    k = pi/M * (0:M)
    term1 = Matrix{ComplexF64}(undef, 2*M, 2*M)
    for j in 1:(M + 1)
        for i in j:(M + 1)
            w = 2*(2 - cos(k[i]) - cos(k[j]))
            term1[symmetric_indices(i, j, M)] .= f1(w)
        end
    end
    ifft!(term1, 1:2)

    rho(z) = -imag(Sig2DI(z + 1e-12im, g))/(pi*g^2)  # density of states
    function f2(k, r)
        integrand(w) = g^2 * rho(w) * abs2(G2DI(w + 1e-12im, Delta, g)) * (w - Delta) * sigma(w + 1e-12im, k, r)
        if Ef > 6 - 2*cos(k)
            val, _ = quadgk(integrand, 0, 2 - 2*cos(k), 6 - 2*cos(k), Ef)
        elseif Ef > 2 - 2*cos(k)
            val, _ = quadgk(integrand, 0, 2 - 2*cos(k), Ef)
        else
            val, _ = quadgk(integrand, 0, Ef)
        end
        return val
    end
    term2 = Matrix{ComplexF64}(undef, M + 1, M + 1)
    for j in 1:(M + 1)
        for i in 1:(M + 1)
            term2[i, j] = f2(k[i], j - 1)
        end
    end
    term2 = ifft([term2; term2[(end - 1):-1:2, :]], 1)
    term2 = [term2 term2[:, (end - 1):-1:2]]

    return term1 + term2
end

end