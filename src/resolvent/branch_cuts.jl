include("./self-energies.jl")
using QuadGK

# 1D branch cuts (impurity amplitude)
function branch_cuts_1D(Delta, g, t, eps=1e-12)

    # lower branch cut
    fLBC(x) = exp(-x*t)*(G1DII(eps - im*x, Delta, g) - G1DI(-eps - im*x, Delta, g))
    LBC = quadgk(fLBC, 0, Inf)[1]/2pi

    # upper branch cut
    fUBC(x) = exp(-(x + 4im)*t)*(G1DI(4 + eps - im*x, Delta, g) - G1DII(4 - eps - im*x, Delta, g))
    UBC = quadgk(fUBC, 0, Inf)[1]/2pi

    return LBC, UBC
end

# 1D branch cuts (bath amplitude)
function branch_cuts_w_1D(Delta, g, t, w, eps=1e-12)

    G1DIw(z) = g*G1DI(z, Delta, g)/(z - w)
    G1DIIw(z) = g*G1DII(z, Delta, g)/(z - w)

    # lower branch cut
    fLBC(x) = exp(-x*t)*(G1DIIw(eps - im*x) - G1DIw(-eps - im*x))
    LBC = quadgk(fLBC, 0, Inf)[1]/2pi

    # upper branch cut
    fUBC(x) = exp(-(x + 4im)*t)*(G1DIw(4 + eps - im*x) - G1DIIw(4 - eps - im*x))
    UBC = quadgk(fUBC, 0, Inf)[1]/2pi

    return LBC, UBC
end

# 2D branch cuts (impurity amplitude)
function branch_cuts_2D(Delta, g, t, eps=1e-12)

    # lower branch cut
    fLBC(x) = exp(-x*t)*(G2DII(eps - im*x, Delta, g) - G2DI(-eps - im*x, Delta, g))
    LBC = quadgk(fLBC, 0, Inf)[1]/2pi

    # middle branch cut
    fMBC(x) = exp(-(x + 4im)*t)*(G2DIII(4 + eps - im*x, Delta, g) - G2DII(4 - eps - im*x, Delta, g))
    MBC = quadgk(fMBC, 0, Inf)[1]/2pi

    # upper branch cut
    fUBC(x) = exp(-(x + 8im)*t)*(G2DI(8 + eps - im*x, Delta, g) - G2DIII(8 - eps - im*x, Delta, g))
    UBC = quadgk(fUBC, 0, Inf)[1]/2pi

    return LBC, MBC, UBC
end

# 2D branch cuts (bath amplitude)
function branch_cuts_w_2D(Delta, g, t, w, eps=1e-12)

    G2DIw(z) = g*G2DI(z, Delta, g)/(z - w)
    G2DIIw(z) = g*G2DII(z, Delta, g)/(z - w)
    G2DIIIw(z) = g*G2DIII(z, Delta, g)/(z - w)

    # lower branch cut
    fLBC(x) = exp(-x*t)*(G2DIIw(eps - im*x) - G2DIw(-eps - im*x))
    LBC = quadgk(fLBC, 0, Inf)[1]/2pi

    # middle branch cut
    fMBC(x) = exp(-(x + 4im)*t)*(G2DIIIw(4 + eps - im*x) - G2DIIw(4 - eps - im*x))
    MBC = quadgk(fMBC, 0, Inf)[1]/2pi

    # upper branch cut
    fUBC(x) = exp(-(x + 8im)*t)*(G2DIw(8 + eps - im*x) - G2DIIIw(8 - eps - im*x))
    UBC = quadgk(fUBC, 0, Inf)[1]/2pi

    return LBC, MBC, UBC
end