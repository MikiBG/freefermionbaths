using ArgParse, LinearAlgebra, HDF5

# parse input arguments from command line
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "filename"
            help = "data file name"
            arg_type = String
            required = true
        "Delta"
            help = "on-site impurity energy"
            arg_type = Float64
            required = true
        "g"
            help = "final coupling strength"
            arg_type = Float64
            required = true
        "Ef"
            help = "Fermi level"
            arg_type = Float64
            required = true
        "Tg"
            help = "adiabatic preparation time"
            arg_type = Float64
            required = true
        "L"
            help = "bath size"
            arg_type = Int
            required = true
        "dt"
            help = "trotterisation time step (optional)"
            arg_type = Float64
            default  = 0.1
        "T"
            help = "final time (optional)"
            arg_type = Float64
            default  = 300.0
    end

    return parse_args(s)
end

# conversion correlation matrix to covariance matrix
function gamma(C)
    A = [C zero(C); zero(C) (I - C)]
    B = kron([1 1; im -im], Matrix(I, size(C)))
    return im*(B*A*B' - I)
end

# Gaussian state overlap [see e.g. Schuch et al., PRB 100 (2019)]
overlap(C1, C2) = sqrt(real(det((I - gamma(C1)*gamma(C2))/2)))

# bath Hamiltonian with open BCs
bare_hamiltonian(L) = diagm(0 => fill(2, L), 1 => fill(-1, L - 1), -1 => fill(-1, L - 1))

# impurity model Hamiltonian with open BCs
function hamiltonian(Delta, g, L)
    H = zeros(ComplexF64, L + 1, L + 1)
    H[1, 1] = Delta
    H[1, 2] = g
    H[2, 1] = g
    H[2:end, 2:end] = bare_hamiltonian(L)
    return H
end

# N-excitation ground state of system with single-particle Hamiltonian H
function ground_state(N, H)
    U = eigen(Hermitian(H)).vectors
    return U*diagm([ones(N); zeros(size(H, 1) - N)])*U'
end

# compute evolution of fidelity under adiabatic protocol
function adiabatic_preparation(; Delta, g, Ef, Tg, dt, T, L)
    
    Nsteps = ceil(Int, T/dt)              # number of time steps
    dg = g*dt/Tg                          # coupling increments
    N = floor(Int, acos(1 - Ef/2)*L/pi)   # number of excitations
    
    # initial correlation matrix (Fermi sea)
    H = bare_hamiltonian(L)
    C = [0 zeros(ComplexF64, 1, L); zeros(ComplexF64, L, 1) ground_state(N, H)]
    
    # GS correlation matrix for the same particle number
    H = hamiltonian(Delta, g, L)
    CGS = ground_state(N, H)
    
    fidelities = zeros(Float64, Nsteps)
    correlations = zeros(ComplexF64, Nsteps, L)  # record of impurity-bath correlations
    times = zeros(Float64, Nsteps)

    for i in 1:Nsteps
        
        # update time & fidelity
        fidelities[i] = overlap(CGS, C)
        times[i] = dt*(i - 1)
        
        # update correlations
        correlations[i, :] = C[1, 2:end]
        
        # evolve state
        U = exp(-im*dt*hamiltonian(Delta, min(i*dg, g), L))
        C = U' * C * U
    end

    return times, fidelities
end

# run adiabatic protocol with parsed input & save data
function main()
    parsed_args = parse_commandline()

    # print parsed arguments
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("  $arg  =>  $val")
    end

    # run numerics with arguments from input
    times, fidelities = adiabatic_preparation(
        Delta    = parsed_args["Delta"],
        g        = parsed_args["g"],
        Ef       = parsed_args["Ef"],
        Tg       = parsed_args["Tg"],
        dt       = parsed_args["dt"], 
        T        = parsed_args["T"], 
        L        = parsed_args["L"]
    )

    # save data to HDF5 file
    h5open(parsed_args["filename"]*".h5", "w") do fid
        attributes(fid)["Delta"] = parsed_args["Delta"]
        attributes(fid)["g"]     = parsed_args["g"]
        attributes(fid)["Ef"]    = parsed_args["Ef"]
        attributes(fid)["Tg"]    = parsed_args["Ef"]
        attributes(fid)["L"]     = parsed_args["L"]
        write(fid, "fidelity", fidelities[end])
    end
end

main()