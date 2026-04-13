module SiN

export default_params, default_state, state_order,
       Vs, ah, bh, hinf, am, bm, minf, an, bn, ninf, xinf,
       Ih, dy, melibe_h!, melibe_h

using StaticArrays

# Adapted from https://github.com/hinsley/PlantChaos (model/Plant.jl)

const default_params = @SVector Float64[
    1.0e0,    # 1: C_m
    4.0e0,    # 2: gI
    0.3e0,    # 3: gK
    0.0e0,    # 4: g_h
    0.003e0,  # 5: gL
    0.01e0,   # 6: gT
    0.03e0,   # 7: gKCa
    30.0e0,   # 8: EI
    -75.0e0,  # 9: EK
    -70.0e0,  # 10: E_h
    -40.0e0,  # 11: EL
    140.0e0,  # 12: ECa
    0.0085e0, # 13: Kc
    100.0e0,  # 14: tau_x
    20000.0,  # 15: tau_y
    0.0003e0, # 16: rho
    0.0e0,    # 17: Delta x
    0.0e0     # 18: Delta Ca
]

state_order() = (:x, :y, :n, :h, :Ca, :V)

Vs(V) = (oftype(V, 127) * V + oftype(V, 8265)) / oftype(V, 105)

function am(V)
    VsV = Vs(V)
    num = oftype(V, 0.1) * (oftype(V, 50) - VsV)
    den = exp((oftype(V, 50) - VsV) / oftype(V, 10)) - one(V)
    return num / den
end

bm(V) = oftype(V, 4) * exp((oftype(V, 25) - Vs(V)) / oftype(V, 18))

# Fast inward sodium and calcium current
minf(V) = am(V) / (am(V) + bm(V))
ah(V) = oftype(V, 0.07) * exp((oftype(V, 25) - Vs(V)) / oftype(V, 20))
bh(V) = one(V) / (one(V) + exp((oftype(V, 55) - Vs(V)) / oftype(V, 10)))
hinf(V) = ah(V) / (ah(V) + bh(V))
th(V) = oftype(V, 12.5) / (ah(V) + bh(V))
dh(h, V) = (hinf(V) - h) / th(V)
II(p, h, V) = oftype(V, p[2]) * h * minf(V)^3 * (V - oftype(V, p[8]))

function an(V)
    VsV = Vs(V)
    num = oftype(V, 0.01) * (oftype(V, 55) - VsV)
    den = exp((oftype(V, 55) - VsV) / oftype(V, 10)) - one(V)
    return num / den
end

bn(V) = oftype(V, 0.125) * exp((oftype(V, 45) - Vs(V)) / oftype(V, 80))
ninf(V) = an(V) / (an(V) + bn(V))
tn(V) = oftype(V, 12.5) / (an(V) + bn(V))
IK(p, n, V) = oftype(V, p[3]) * n^4 * (V - oftype(V, p[9]))
dn(n, V) = (ninf(V) - n) / tn(V)

xinf(p, V) = one(V) / (one(V) + exp(oftype(V, 0.15) * (oftype(V, p[17]) - V - oftype(V, 50))))
IT(p, x, V) = oftype(V, p[6]) * x * (V - oftype(V, p[8]))
dx(p, x, V) = (xinf(p, V) - x) / oftype(V, p[14])

Ih(p, y, V) = oftype(V, p[4]) * y * (V - oftype(V, p[10]))
yinf(V) = one(V) / (one(V) + exp(V + oftype(V, 54)))
dy(p, y, V) = (yinf(V) - y) / oftype(V, p[15])

Ileak(p, V) = oftype(V, p[5]) * (V - oftype(V, p[11]))

IKCa(p, Ca, V) = oftype(V, p[7]) * Ca * (V - oftype(V, p[9])) / (oftype(V, 0.5) + Ca)
dCa(p, Ca, x, V) = oftype(V, p[16]) * (oftype(V, p[13]) * x * (oftype(V, p[12]) - V + oftype(V, p[18])) - Ca)

function dV(p, x, y, n, h, Ca, V)
    return -(II(p, h, V) + IK(p, n, V) + IT(p, x, V) + IKCa(p, Ca, V) +
             Ih(p, y, V) + Ileak(p, V)) / oftype(V, p[1])
end

const default_state = @SVector Float64[
    0.8e0,      # x
    yinf(-62.0),# y
    0.137e0,    # n
    0.389e0,    # h
    0.8e0,      # Ca
    -62.0e0     # V
]

function melibe_h(u::AbstractArray{T}, p, t) where T
    return @SVector T[
        dx(p, u[1], u[6]),
        dy(p, u[2], u[6]),
        dn(u[3], u[6]),
        dh(u[4], u[6]),
        dCa(p, u[5], u[1], u[6]),
        dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
    ]
end

function melibe_h!(du, u, p, t)
    du[1] = dx(p, u[1], u[6])
    du[2] = dy(p, u[2], u[6])
    du[3] = dn(u[3], u[6])
    du[4] = dh(u[4], u[6])
    du[5] = dCa(p, u[5], u[1], u[6])
    du[6] = dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
    return nothing
end

end
