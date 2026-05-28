if !isdefined(@__MODULE__, :FlowFolding)
    include(joinpath(@__DIR__, "..", "FlowFolding.jl"))
end

using .FlowFolding

const ROSSLER_MALYKH_B = 0.3

rossler_origin_equilibrium() = [0.0, 0.0, 0.0]

function rossler_secondary_equilibrium(a, c; b=ROSSLER_MALYKH_B)
    z = (c - a * b) / a
    return [c - a * b, -z, z]
end

function rossler_malykh(u, p, t)
    a = p.a
    b = p.b
    c = p.c
    x = u[1]
    y = u[2]
    z = u[3]
    return [
        -y - z,
        x + a * y,
        b * x + z * (x - c),
    ]
end

function rossler_malykh_jacobian(u, p, t)
    a = p.a
    b = p.b
    c = p.c
    x = u[1]
    z = u[3]
    return [
        0.0 -1.0 -1.0
        1.0 a 0.0
        b + z 0.0 x - c
    ]
end

function rossler_y_minima_problem(a, c; b=ROSSLER_MALYKH_B)
    p = (a=Float64(a), b=Float64(b), c=Float64(c))
    return FlowFoldingProblem(
        rossler_malykh,
        p;
        variable_index=2,
        extremum=StateMinimum,
        dimension=3,
        jacobian=rossler_malykh_jacobian,
    )
end

function tangent_word(events)
    chars = map(events) do event
        event.sign > 0 ? '1' : (event.sign < 0 ? '0' : '.')
    end
    return join(chars)
end

function tangent_bits(events)
    bits = Int[]
    for event in events
        event.sign == 0 && return nothing
        push!(bits, event.sign > 0 ? 1 : 0)
    end
    return bits
end

function tangent_word_code(events)
    bits = tangent_bits(events)
    isnothing(bits) && return -1
    code = 0
    for bit in bits
        code = 2 * code + bit
    end
    return code
end
