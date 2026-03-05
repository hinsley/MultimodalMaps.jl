module Plant

export melibeNew, melibeNew!, melibeNewReverse!, default_params, default_state,
       Vs, ah, bh, hinf, am, bm, minf, an, bn, ninf, xinf, IKCa, Vddot,
       numerical_derivative

using StaticArrays

default_params = @SVector Float32[
    1.0e0,    # 1: Cₘ
    4.0e0,    # 2: gI
    0.3e0,    # 3: gK
    0.0e0,    # 4: gₕ
    0.003e0,  # 5: gL
    0.01e0,   # 6: gT
    0.03e0,   # 7: gKCa
    30.0e0,   # 8: EI
    -75.0e0,  # 9: EK
    60.0e0,   # 10: Eₕ
    -42.0e0,  # 11: EL
    140.0e0,  # 12: ECa
    0.0085e0, # 13: Kc
    100.0e0,  # 14: τₓ
    0.0003e0, # 15: ρ
    0.0e0,    # 16: Δx
    0.0e0     # 17: ΔCa
]

Vs(V) = (127.0 * V + 8265.0) / 105.0

am(V) = 0.1 * (50.0 - Vs(V)) / (exp((50.0 - Vs(V)) / 10.0) - 1.0)
bm(V) = 4.0 * exp((25.0 - Vs(V))/18.0)

# Fast inward sodium and calcium current
minf(V) = am(V) / (am(V) + bm(V))
ah(V) = 0.07 * exp((25.0 - Vs(V)) / 20.0)
bh(V) = 1.0 / (1.0 + exp((55.0 - Vs(V)) / 10.0))
hinf(V) = ah(V) / (ah(V) + bh(V))
th(V) = 12.5 / (ah(V) + bh(V))
dh(h, V) = (hinf(V) - h) / th(V)
II(p, h, V) = p[2] * h * minf(V)^3.0 * (V - p[8])

an(V) = 0.01 * (55.0 - Vs(V)) / (exp((55.0 - Vs(V)) / 10.0) - 1.0)
bn(V) = 0.125 * exp((45.0 - Vs(V)) / 80.0)
ninf(V) = an(V) / (an(V) + bn(V))
tn(V) = 12.5 / (an(V) + bn(V))
IK(p, n, V) = p[3] * n^4.0 * (V - p[9])
dn(n, V) = (ninf(V) - n) / tn(V)

xinf(p, V) = 1.0 / (1.0 + exp(0.15 * (p[16] - V - 50.0)))
IT(p, x, V) = p[6] * x * (V - p[8])
dx(p, x, V) = (xinf(p, V) - x) / p[14]
xinfinv(p, xinf) = p[16] - 50.0f0 - log(1.0f0/xinf - 1.0f0)/0.15f0

Ih(p, y, V) = p[4] * y * (V - p[10])
dy(y, V) = ((1/(1+exp((V+54))))-y)/20000

Ileak(p, V) = p[5] * (V - p[11])

IKCa(p, Ca, V) = p[7] * Ca * (V - p[9]) / (0.5 + Ca)
dCa(p, Ca, x, V) = p[15] * (p[13] * x * (p[12] - V + p[17]) - Ca)

function dV(p, x, y, n, h, Ca, V)
    return -(II(p, h, V) + IK(p, n, V) + IT(p, x, V) + IKCa(p, Ca, V) + Ih(p, y, V) + Ileak(p, V)) / p[1]
end

function dV(p, x, y, n, h, Ca, V, Isyn)
    return -(II(p, h, V) + IK(p, n, V) + IT(p, x, V) + IKCa(p, Ca, V) + Ih(p, y, V) + Ileak(p, V)+ Isyn) / p[1]
end

function melibeNew(u::AbstractArray{T}, p, t) where T
    return @SVector T[
        dx(p, u[1], u[6]),
        0.0 * dy(u[2], u[6]),
        dn(u[3], u[6]),
        dh(u[4], u[6]),
        dCa(p, u[5], u[1], u[6]),
        dV(p, u[1], u[2], u[3], u[4], u[5], u[6]),
    ]
end

function melibeNew!(du, u, p, t)
    du[1] = dx(p, u[1], u[6])
    du[2] = 0.0 * dy(u[2], u[6])
    du[3] = dn(u[3], u[6])
    du[4] = dh(u[4], u[6])
    du[5] = dCa(p, u[5], u[1], u[6])
    du[6] = dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
end

function melibeNewReverse!(du, u, p, t)
    du[1] = -dx(p, u[1], u[6])
    du[2] = -0.0 * dy(u[2], u[6])
    du[3] = -dn(u[3], u[6])
    du[4] = -dh(u[4], u[6])
    du[5] = -dCa(p, u[5], u[1], u[6])
    du[6] = -dV(p, u[1], u[2], u[3], u[4], u[5], u[6])
end

default_state = @SVector Float32[
    0.8e0;
    0.137e0;
    0.389e0;
    0.8e0;
    -62.0e0;
]

const Vsprime = 127.0/105.0

function amprime(V)
    return 0.1 * (-Vsprime * (exp((50.0 - Vs(V)) / 10.0) - 1.0) + (50.0 - Vs(V)) / 10.0 * exp((50.0 - Vs(V)) / 10.0) ) / (exp((50.0 - Vs(V)) / 10.0) - 1.0)^2
end

function bmprime(V)
    return -2.0/9.0*Vsprime*exp((25.0-Vs(V))/18.0)
end

function minfdot(V, Vdot)
    return Vdot*(amprime(V)*bm(V)-am(V)*bmprime(V))/(am(V)+bm(V))^2.0
end

function IIdot(p, h, hdot, V, Vdot)
    return p[2]*((hdot*minf(V)^3.0+3.0*h*minfdot(V, Vdot)*minf(V)^2.0)*(V-p[8])+h*minf(V)^3.0*Vdot)
end

function IKdot(p, n, ndot, V, Vdot)
    return p[3]*n^3.0*(4*ndot*(V-p[9])+n*Vdot)
end

function ITdot(p, x, xdot, V, Vdot)
    return p[6]*(xdot*(V-p[8])+x*Vdot)
end

function IKCadot(p, Ca, Cadot, V, Vdot)
    return p[7]*(0.5*Cadot*(V - p[9]) + Ca*Vdot*(Ca + 0.5))/(Ca + 0.5)^2.0
end

function Ileakdot(p, Vdot)
    return p[5]*Vdot
end

function Vddot(p, h, hdot, n, ndot, x, xdot, Ca, Cadot, V, Vdot)
    return -(IIdot(p, h, hdot, V, Vdot) + IKdot(p, n, ndot, V, Vdot) + ITdot(p, x, xdot, V, Vdot) + IKCadot(p, Ca, Cadot, V, Vdot) + Ileakdot(p, Vdot))/p[1]
end

function numerical_derivative(f, u, p, dt=1e-8)
    du = melibeNew(u, p, 0.0)

    x = u[1]; xdot = du[1]
    y = u[2]; ydot = du[2]
    n = u[3]; ndot = du[3]
    h = u[4]; hdot = du[4]
    Ca = u[5]; Cadot = du[5]
    V = u[6]; Vdot = du[6]
    f_u = f(p, h, hdot, n, ndot, x, xdot, Ca, Cadot, V, Vdot)

    u_new = u .+ du .* dt
    du_new = melibeNew(u_new, p, dt)

    x_new = u_new[1]; xdot_new = du_new[1]
    y_new = u_new[2]; ydot_new = du_new[2]
    n_new = u_new[3]; ndot_new = du_new[3]
    h_new = u_new[4]; hdot_new = du_new[4]
    Ca_new = u_new[5]; Cadot_new = du_new[5]
    V_new = u_new[6]; Vdot_new = du_new[6]
    f_u_new = f(p, h_new, hdot_new, n_new, ndot_new, x_new, xdot_new, Ca_new, Cadot_new, V_new, Vdot_new)

    df_dt = (f_u_new - f_u) / dt
    return df_dt
end

end
