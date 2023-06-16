include("../src/FiniteSizeDynamics.jl")
using .FiniteSizeDynamics
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
        "Ef"
            help = "Fermi level"
            arg_type = Float64
            required = true
        "dt"
            help = "trotterisation time step (optional)"
            arg_type = Float64
            default  = 0.1
        "T"
            help = "final time (optional)"
            arg_type = Float64
            default  = 200.0
        "g"
            help = "coupling strength (optional)"
            arg_type = Float64
            default  = 0.2
        "log2L"
            help = "linear bath size (exponentiated with basis 2, optional)"
            arg_type = Int
            default  = 9
    end

    return parse_args(s)
end

# compute dynamics of impurity occupation
function occupation_dynamics(; Delta, g, Ef, dt, T, L)

    # impurity coupled to middle of the bath
    imp = Impurity(Delta, g, (Int(L/2), Int(L/2)))

    # discretised Hamiltonian (L bins)
    H = discretised_hamiltonian(k -> 2*(2 - cos(k[1]) - cos(k[2])), [imp], L, L)  

    # discretised initial correlation matrix (occupied impurity + Fermi sea)
    Cc = ones(ComplexF64, 1, 1)
    Cb = diagm(ComplexF64.(real(diag(H)[2:end]) .< Ef))
    Ccb = zeros(ComplexF64, 1, L)
    C = [[Cc Ccb]; [Ccb' Cb]]

    # dynamics of impurity occupation
    times, record = hamiltonian_evolution!(C -> [real(C[1, 1])], C, H, dt, ceil(Int, T/dt))

    return times, record
end

# run time-evolution with parsed input & save data
function main()
    parsed_args = parse_commandline()

    # print parsed arguments
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("  $arg  =>  $val")
    end

    # run numerics with arguments from input
    times, record = occupation_dynamics(
        Delta    = parsed_args["Delta"],
        g        = parsed_args["g"],
        Ef       = parsed_args["Ef"],
        dt       = parsed_args["dt"], 
        T        = parsed_args["T"], 
        L        = 2^(parsed_args["log2L"])
    )

    # save data to HDF5 file
    h5open(parsed_args["filename"]*".h5", "w") do fid
        attributes(fid)["Delta"] = parsed_args["Delta"]
        attributes(fid)["g"]     = parsed_args["g"]
        attributes(fid)["Ef"]    = parsed_args["Ef"]
        attributes(fid)["L"]     = 2^(parsed_args["log2L"])
        write(fid, "times", collect(times))
        write(fid, "impurity occupation", record[:, 1])
    end
end

main()