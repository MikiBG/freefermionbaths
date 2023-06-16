include("./self-energies.jl")
using Roots

# 1D real poles (I. Riemann sheet)
function real_poles_1D(Delta, g, eps=1e-12)
    
    # find zero using a bisection algorithm
    # the zero has to be in the bracketing interval, otherwise, we
    # assume it is very close to the corresponding band edge
    f(z) = z - Delta - real(Sig1D(z, g))
    intervals = [(-10.0, -eps), (4 + eps, 14.0)]
    poles = Vector{Float64}(undef, 2)
    for (i, interval) in enumerate(intervals)
        if f(interval[1]) * f(interval[2]) > 0
            poles[i] = interval[3 - i]
        else 
            poles[i] = find_zero(f, interval)
        end
    end

    return poles..., (@. 1/(1 - dSig1D(poles, g)))...
end

# 2D real poles (I. Riemann sheet)
function real_poles_2D(Delta, g, eps=1e-12)
    
    # find zero using a bisection algorithm
    # the zero has to be in the bracketing interval, otherwise, we
    # assume it is very close to the corresponding band edge
    f(z) = z - Delta - real(Sig2DI(z, g))
    intervals = [(-10.0, -eps), (8 + eps, 18.0)]
    poles = Vector{Float64}(undef, 2)
    for (i, interval) in enumerate(intervals)
        if f(interval[1]) * f(interval[2]) > 0
            poles[i] = interval[3 - i]
        else 
            poles[i] = find_zero(f, interval)
        end
    end

    return poles..., (@. 1/(1 - dSig2DI(poles, g)))...
end