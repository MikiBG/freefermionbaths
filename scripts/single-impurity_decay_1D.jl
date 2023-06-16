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
            help = "bath size (exponentiated with basis 2, optional)"
            arg_type = Int
            default  = 10
    end

    return parse_args(s)
end

# compute dynamics of impurity & bath occupations
function occupation_dynamics(; Delta, g, Ef, dt, T, L)

    # impurity coupled to middle of the bath
    imp = Impurity(Delta, g, (Int(L/2),))

    # initial correlation matrix (occupied impurity + Fermi sea)
    Cc = ones(ComplexF64, 1, 1)
    Cb = diagm(ComplexF64[2*(1 - cos(2*pi*k/L)) < Ef for k in 0:(L - 1)])
    Ccb = zeros(ComplexF64, 1, L)

    # dynamics of impurity & bath occupations
    output(Cc, Cb, Ccb) = [real(Cc[1, 1]); real(diag(Cb))]
    times, record = spectral_evolution!(output, Cc, Cb, Ccb, [imp], dt, ceil(Int, T/dt));

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
        write(fid, "bath occupations", record[:, 2:end])
    end
end

main()