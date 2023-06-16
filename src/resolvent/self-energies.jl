using HypergeometricFunctions

# 1D self-energy and its derivative
Sig1D(z, g) = (a = z - 2; g^2*sign(real(a))/sqrt(Complex(a^2 - 4)))
dSig1D(z, g) = (a = z - 2; -g^2*sign(real(a))*a/sqrt(Complex(a^2 - 4))^3)

# 1D impurity Green's functions
G1DI(z, Delta, g) = 1/(z - Delta - Sig1D(z, g))
G1DII(z, Delta, g) = 1/(z - Delta + Sig1D(z, g))

# elliptic integrals using hypergeometric functions
ellipK(z) = pi/2*_₂F₁(1/2, 1/2, 1, Complex(z))
ellipE(z) = pi/2*_₂F₁(-1/2, 1/2, 1, Complex(z))

# 2D self-energy and its derivative in I. Riemann sheet
Sig2DI(z, g) = (a = 4/(z - 4); g^2*a/(2pi)*ellipK(a^2))
dSig2DI(z, g) = -2*g^2/(pi*z*(z - 8))*ellipE((4/(z - 4))^2)

# 2D self-energy and its derivative in II. Riemann sheet
Sig2DII(z, g) = (a = 4/(z - 4); g^2*a/(2pi)*(ellipK(a^2) + 2im*ellipK(1 - a^2)))
dSig2DII(z, g) = (a = 16/(z - 4)^2; b = z*(z - 8)*a/16; 
                  -g^2*a/(8pi*b)*(ellipE(a) + 2im*(ellipK(b) - ellipE(b))))

# 2D self-energy and its derivative in III. Riemann sheet
Sig2DIII(z, g) = (a = 4/(z - 4); g^2*a/(2pi)*(ellipK(a^2) - 2im*ellipK(1 - a^2)))
dSig2DIII(z, g) = (a = 16/(z - 4)^2; b = z*(z - 8)*a/16; 
                  -g^2*a/(8pi*b)*(ellipE(a) - 2im*(ellipK(b) - ellipE(b))))

# 2D impurity Green's functions
G2DI(z, Delta, g) = 1/(z - Delta - Sig2DI(z, g))
G2DII(z, Delta, g) = 1/(z - Delta - Sig2DII(z, g))
G2DIII(z, Delta, g) = 1/(z - Delta - Sig2DIII(z, g))