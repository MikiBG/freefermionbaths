include("./self-energies.jl")
using NLsolve

# 1D complex pole (II. Riemann sheet)
function complex_pole_1D(Delta, g)

    # A solution of the pole equation, f(z) = 0, is a solution to the non-linear 
    # system of equations F(x) = [0, 0]

    f(z) = z - Delta + Sig1D(z, g)    
    F(x) = (y = f(Complex(x[1], x[2])); [real(y), imag(y)])    
    x0 = nlsolve(F, [Delta, -2.0]).zero    # complex pole in cartesian coordinates
                                           # a good initial guess is required
    zII = Complex(x0[1], x0[2])
    RII = 1/(1 + dSig1D(zII, g))
    if abs(x0[1] - 2.0) > 2 || x0[2] > 0    
        RII = zero(RII)    # if zero is outside the integration contour it doesn't contribute
    end

    return zII, RII
end

# 2D complex poles (II. & III. Riemann sheets)
function complex_poles_2D(Delta, g)

    # A solution of the pole equation, f(z) = 0, is a solution to the non-linear 
    # system of equations F(x) = [0, 0]
    
    fII(z) = z - Delta - Sig2DII(z, g)    
    FII(x) = (y = fII(Complex(x[1], x[2])); [real(y), imag(y)])    
    x0 = nlsolve(FII, [Delta, -2.0]).zero    # complex pole in cartesian coordinates
                                             # a good initial guess is required
    zII = Complex(x0[1], x0[2])
    RII = 1/(1 - dSig2DII(zII, g))
    if abs(x0[1] - 2.0) > 2 || x0[2] > 0    
        RII = zero(RII)    # if zero is outside the integration contour it doesn't contribute
    end

    fIII(z) = z - Delta - Sig2DIII(z, g)    
    FIII(x) = (y = fIII(Complex(x[1], x[2])); [real(y), imag(y)])    
    x0 = nlsolve(FIII, [Delta, -2.0]).zero    # complex pole in cartesian coordinates
                                              # a good initial guess is required
    zIII = Complex(x0[1], x0[2])
    RIII = 1/(1 - dSig2DIII(zIII, g))
    if abs(x0[1] - 6.0) > 2 || x0[2] > 0    
        RIII = zero(RIII)    # if zero is outside the integration contour it doesn't contribute
    end

    return zII, zIII, RII, RIII
end