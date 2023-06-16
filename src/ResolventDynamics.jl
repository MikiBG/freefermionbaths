"""
Implement exact time-evolution of the impurity occupation, as well as its long-term average, 
for a single impurity coupled to 1D and 2D baths, for an initial state with the impurity 
occupied and the bath in its zero-temperature Fermi sea state, computed using the correlation 
matrix resolvent formalism.
"""
module ResolventDynamics
export occupation_1D, occupation_2D, LTA_occupation_1D, LTA_occupation_2D

include("./resolvent/self-energies.jl")
include("./resolvent/real_poles.jl")
include("./resolvent/complex_poles.jl")
include("./resolvent/branch_cuts.jl")

"""
    occupation_1D(Delta, g, Ef, times)

Evaluate the impurity occupation of a single impurity coupled to a 1D bath
at a given time using the resolvent formalism for the correlation matrix.

#### Arguments
- `Delta`: impurity on-site energy.
- `g`: impurity-bath coupling strength.
- `Ef`: Fermi level.
- `times`: time values at which the impurity occupation is computed.
"""
function occupation_1D(Delta, g, Ef, times)

    zL, zU, RL, RU = real_poles_1D(Delta, g)
    zII, RII = complex_pole_1D(Delta, g)

    # impurity amplitude
    total = [abs2(RL*exp(-im*zL*t) + RU*exp(-im*zU*t) +
                  RII*exp(-im*zII*t) +
                  sum(branch_cuts_1D(Delta, g, t))) for t in times]
    if Ef > 0
        # bath amplitude
        Aw1D(z, t) = g*(RL/(zL - z)*exp(-im*zL*t) + RU/(zU - z)*exp(-im*zU*t) + 
                        G1DII(z, Delta, g)*exp(-im*z*t) +
                        RII/(zII - z)*exp(-im*zII*t) +
                        sum(branch_cuts_w_1D(Delta, g, t, z))/g)
        f(z, t) = -imag(Sig1D(z + 1e-12im, 1))/pi*abs2(Aw1D(z, t))
        total .+= [quadgk(z -> f(z, t), 0, Ef)[1] for t in times]
    end

    return total
end

"""
    occupation_2D(Delta, g, Ef, times)

Evaluate the impurity occupation of a single impurity coupled to a 2D bath
at a given time using the resolvent formalism for the correlation matrix.

#### Arguments
- `Delta`: impurity on-site energy.
- `g`: impurity-bath coupling strength.
- `Ef`: Fermi level.
- `times`: time values at which the impurity occupation is computed.
"""
function occupation_2D(Delta, g, Ef, times)

    zL, zU, RL, RU = real_poles_2D(Delta, g)
    zII, zIII, RII, RIII = complex_poles_2D(Delta, g)

    # impurity amplitude
    total = [abs2(RL*exp(-im*zL*t) + RU*exp(-im*zU*t) +
                  RII*exp(-im*zII*t) + RIII*exp(-im*zIII*t) +
                  sum(branch_cuts_2D(Delta, g, t))) for t in times]
    if Ef > 0
        # bath amplitude
        Aw2D(z, t) = g*(RL/(zL - z)*exp(-im*zL*t) + RU/(zU - z)*exp(-im*zU*t) + 
                        G2DI(z, Delta, g)*exp(-im*z*t) +
                        RII/(zII - z)*exp(-im*zII*t) + RIII/(zIII - z)*exp(-im*zIII*t) +
                        sum(branch_cuts_w_2D(Delta, g, t, z))/g)
        f(z, t) = -imag(Sig2DI(z + 1e-12im, 1))/pi*abs2(Aw2D(z, t))
        total .+= [quadgk(z -> f(z, t), 0, Ef)[1] for t in times]
    end

    return total
end

"""
    LTA_occupation_1D(Delta, g, Ef)

Compute the long-term averaged impurity occupation (using the analytical formula) for a 
single impurity coupled to a 1D bath.

#### Arguments
- `Delta`: impurity on-site energy.
- `g`: impurity-bath coupling strength.
- `Ef`: Fermi level.
"""
function LTA_occupation_1D(Delta, g, Ef)
    
    # lower and upper bound state contributions
    zL, zU, RL, RU = real_poles_1D(Delta, g)
    LBS = abs2(RL)*(1 + g^2*quadgk(z -> 1/(zL - z)^2, 0, Ef)[1])
    UBS = abs2(RU)*(1 + g^2*quadgk(z -> 1/(zU - z)^2, 0, Ef)[1])

    # scattering eigenstates' contribution
    f(z) = -imag(Sig1D(z + 1e-12im, g))/pi*abs2(G1DI(z + 1e-12im, Delta, g))                                      # density of states
    scat = quadgk(f, 0, Ef)[1]
    
    return LBS + UBS + scat
end

"""
    LTA_occupation_2D(Delta, g, Ef)

Compute the long-term averaged impurity occupation (using the analytical formula) for a 
single impurity coupled to a 2D bath.

#### Arguments
- `Delta`: impurity on-site energy.
- `g`: impurity-bath coupling strength.
- `Ef`: Fermi level.
"""
function LTA_occupation_2D(Delta, g, Ef)

    # lower and upper bound state contributions
    zL, zU, RL, RU = real_poles_2D(Delta, g)
    LBS = abs2(RL)*(1 + g^2*quadgk(z -> 1/(zL - z)^2, 0, Ef)[1])
    UBS = abs2(RU)*(1 + g^2*quadgk(z -> 1/(zU - z)^2, 0, Ef)[1])              

    # scattering eigenstates' contribution
    f(z) = -imag(Sig2DI(z + 1e-12im, g))/pi*abs2(G2DI(z + 1e-12im, Delta, g))
    scat = quadgk(f, 0, Ef)[1]

    return LBS + UBS + scat
end

end