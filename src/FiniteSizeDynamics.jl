"""
Implement exact time-evolution for impurities coupled to a finite 1D bath (using spectral 
methods) and to D-dimensional finite baths with arbitrary dispersion (using the method of 
frequency discretisation).

For more details on the methods used, see e.g. González-Tudela et al., PRA 96 (2017).
"""
module FiniteSizeDynamics
using FFTW, LinearAlgebra
export Impurity, spectral_evolution!, discretised_hamiltonian, hamiltonian_evolution!

"""
    Impurity{D}(Delta, g, x)

Structure representing an impurity coupled to a D-dimensional bath.
The constructor admits keyword arguments too.

#### Arguments
- `Delta::Float64`: impurity on-site energy.
- `g::Float64`: impurity-bath coupling strength.
- `x::NTuple{D,Int64}`: lattice site to which impurity is coupled.
"""
struct Impurity{D}
    Delta::Float64
    g::Float64
    x::NTuple{D,Int64}
end

Impurity(; Delta, g, x) = Impurity(Delta, g, x)

"""
    times, record = spectral_evolution!(output, Cc, Cb, Ccb, impurity_list, dt, Nsteps)

Calculate the time-evolution of arbitrary quantities for impurities coupled to a 1D bath 
using the spectral method.

#### Arguments
- `output::Function`: quantities of interest in position representation [of the form 
  `output(Cc, Cb, Ccb)`].
- Initial correlation matrix in momentum representation (overwritten with the final 
  correlation matrix):
    - `Cc::Matrix{ComplexF64}`: impurity block.
    - `Cb::Matrix{ComplexF64}`: bath block.
    - `Ccb::Matrix{ComplexF64}`: impurity-bath block.
- `impurity_list::Vector{Impurity{1}}`: list of impurities coupled to the bath.
- `dt::Float64`: time step of the trotterised time-evolution.
- `Nsteps::Int64`: number of time steps computed.

#### Outputs
- `times::StepRangeLen`: range of times at which the quantities of interest are evaluated.
- `record::Matrix{eltype(output(Cc, Cb, Ccb))}`: record of outputs.
"""
function spectral_evolution!(
    output::Function,
    Cc::Matrix{ComplexF64},
    Cb::Matrix{ComplexF64},
    Ccb::Matrix{ComplexF64},
    impurity_list::Vector{Impurity{1}},
    dt::Float64,
    Nsteps::Int64
    )
    
    N = length(impurity_list)   # number of impurities
    L = size(Cb, 1)             # number of bath sites
    
    # bare unitary evolution (momentum space)
    Ec = [exp(im*e.Delta*dt) for e in impurity_list]
    Eb = [exp(im*2*(1 - cos(2*pi*k/L))*dt) for k in 0:(L - 1)]
    U0c = Ec .* Ec'
    U0b = Eb .* Eb'
    U0cb = Ec .* Eb'

    # plan Fourier transforms
    Fleft_b = plan_fft!(Cb, 1)
    Fright_b = plan_fft!(Cb, 2)
    Fright_cb = plan_fft!(Ccb, 2)

    # interaction unitary evolution (real space)
    UVs = Array{Complex{Float64}, 3}(undef, 2, 2, N)
    for (i, c) in enumerate(impurity_list)
        UVs[:, :, i] = [cos(c.g*dt) im*sin(c.g*dt); im*sin(c.g*dt) cos(c.g*dt)]
    end
    temp1 = zeros(ComplexF64, 2, N + L)
    temp2 = zeros(ComplexF64, N + L, 2)
    temp3 = zeros(ComplexF64, 2, 2)

    out = output(Cc, Cb, Ccb)
    record = zeros(eltype(out), Nsteps + 1, length(out))   # record of outputs
    
    for i in 1:Nsteps
        
        # transform to real space
        Fright_cb * Ccb
        Fleft_b \ (Fright_b * Cb)
        
        # update records
        record[i, :] = output(Cc, Cb, Ccb)
        
        # compute real-space time step
        for (i, c) in enumerate(impurity_list)
            x = c.x[1]
            temp1 .= UVs[:, :, i] * [Cc[i:i, :] Ccb[i:i, :]; Ccb[:, x]' Cb[x:x, :]]
            temp2 .= [[Cc[:, i]; conj(Ccb[i, :])] [Ccb[:, x]; Cb[:, x]]] * UVs[:, :, i]'
            temp3 .= UVs[:, :, i] * [Cc[i, i] Ccb[i, x]; conj(Ccb[i, x]) Cb[x, x]] * UVs[:, :, i]'
            Cc[i, :] .= temp1[1, 1:N]
            Cb[x, :] .= temp1[2, (N + 1):end]
            Ccb[i, :] .= temp1[1, (N + 1):end]
            Cc[:, i] .= temp2[1:N, 1]
            Cb[:, x] .= temp2[(N + 1):end, 2]
            Ccb[:, x] .= temp2[1:N, 2]
            Cc[i, i], Ccb[i, x], Cb[x, x] = temp3[1, 1], temp3[1, 2], temp3[2, 2]
        end

        # transform to momentum space
        Fright_cb \ Ccb
        Fleft_b * (Fright_b \ Cb)
        
        # compute momentum-space time step
        Cc .= U0c .* Cc
        Cb .= U0b .* Cb
        Ccb .= U0cb .* Ccb
    end

    # transform to real space
    Fright_cb * Ccb
    Fleft_b \ (Fright_b * Cb)
    
    # update records
    record[Nsteps + 1, :] = output(Cc, Cb, Ccb)

    return range(0, step=dt, length=(Nsteps + 1)), record
end

"""
    cumulhist(data, bins)

Return both the cumulative histogram of the given `data` in the specified `bins`
and the permutation that sorts the `data` in increasing order.

See also: [`hist`](@ref)
"""
function cumulhist(data::Vector{T}, bins::Vector{T}) where {T<:Real}
    # permutation to sort data according to increasing values
    p = sortperm(data)                                    
    # add additional bin s.t. each data value is definitely sorted
    newbins = [bins; max(bins[end], data[end]) + one(T)]  
    counts = zeros(Int, length(newbins))
    n = 1
    count = 0
    for elem in data[p]
        while elem >= newbins[n]
            counts[n] = count      
            n += 1
        end
        count += 1
    end
    counts[n:end] .= count
    # p[(counts[n - 1] + 1):counts[n]] contains the indices of the elements
    # that fall inside the nth bin
    return counts, p
end

"""
    discretised_hamiltonian(impurity_list, L, Nbin, dispersion)

Construct the Hamiltonian of a set of impurities coupled to a D-dimensional bath with a 
discretised energy band.

#### Arguments
- `dispersion::Function`: bath dispersion relation [of the form `dispersion(k)`].
- `impurity_list::Vector{Impurity{D}}`: list of impurities coupled to the bath.
- `L::Int64`: linear bath size.
- `Nbin::Int64`: number of frequency bins.
"""
function discretised_hamiltonian(
    dispersion::Function, 
    impurity_list::Vector{Impurity{D}}, 
    L::Int64, 
    Nbin::Int64
    ) where{D}

    Nimp = length(impurity_list)    # number of impurities
    Nbath = L^D                     # number of bath modes

    freqs = Vector{Float64}(undef, Nbath)         # energies of the bath eigenmodes
    G = Matrix{ComplexF64}(undef, Nbath, Nimp)    # coupling constants

    # coupling matrix
    dk = 2*pi/L
    dims = Tuple(fill(L, D))
    for (ind, I) in enumerate(CartesianIndices(dims))
        k = dk .* Tuple(I)
        freqs[ind] = dispersion(k)
        for (n, c) in enumerate(impurity_list)
            G[ind, n] = c.g*exp(-im*dot(k, c.x))
        end
    end

    # emitter energies
    H = zeros(ComplexF64, (Nbin + 1)*Nimp, (Nbin + 1)*Nimp)
    for (n, c) in enumerate(impurity_list)
        H[n, n] = c.Delta
    end

    norm = sqrt(Nbath)
    ind = Nimp + 1
    
    # discretize the band range
    bins = collect(range(minimum(freqs) - 1e-10, maximum(freqs) + 1e-10, length=(Nbin + 1)))
    counts, p = cumulhist(freqs, bins)
    
    # compute the coupling coefficients
    for j in 1:Nbin
        if counts[j + 1] > counts[j]
            for i in 0:(Nimp - 1)
                H[ind + i, ind + i] = 0.5*(bins[j] + bins[j + 1])
            end
            _, R = qr(G[p[(counts[j] + 1):counts[j + 1]], :])
            H[ind:(ind + Nimp - 1), 1:Nimp] = R/norm
            H[1:Nimp, ind:(ind + Nimp - 1)] = R'/norm
            ind += Nimp
        end
    end

    return H[1:(ind - 1), 1:(ind - 1)]
end

"""
    times, record = hamiltonian_evolution!(output, C, H, dt, Nsteps)

Calculate the time-evolution of arbitrary quantities for an arbitrary quadratic Hamiltonian.

#### Arguments
- `output::Function`: quantities of interest [of the form `output(C)`].
- `C::Matrix{ComplexF64}`: initial correlation matrix (overwritten with the final 
  correlation matrix).
- `H::Matrix{ComplexF64}`: single-particle Hamiltonian matrix.
- `dt::Float64`: time step of the trotterised time-evolution.
- `Nsteps::Int64`: number of time steps computed.

#### Outputs
- `times::StepRangeLen`: range of times at which the quantities of interest are evaluated.
- `record::Matrix{eltype(output(C))}`: record of outputs.
"""
function hamiltonian_evolution!(
    output::Function, 
    C::Matrix{ComplexF64}, 
    H::Matrix{ComplexF64}, 
    dt::Float64, 
    Nsteps::Int64
    )

    U = exp(-im*H*dt)    # evolution operator
    Udag = U'            # conjugate transpose of evolution opeator

    out = output(C)      # initial output
    record = zeros(eltype(out), Nsteps + 1, length(out))    # record of outputs
    record[1, :] = out

    for i in 1:Nsteps
        C = Udag * C * U
        record[i + 1, :] = output(C)
    end

    return range(0, step=dt, length=(Nsteps + 1)), record
end

end