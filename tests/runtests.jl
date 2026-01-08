using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Test
using LinearAlgebra
using OrdinaryDiffEq
using DynamicalSystems

include(joinpath(@__DIR__, "..", "lyapunovvectors.jl"))

function lorenz!(du, u, p, t)
    x, y, z = u
    σ, ρ, β = p
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end

function lorenz_jacobian!(J, u, p, t)
    x, y, z = u
    σ, ρ, β = p
    J[1, 1] = -σ
    J[1, 2] = σ
    J[1, 3] = 0.0
    J[2, 1] = ρ - z
    J[2, 2] = -1.0
    J[2, 3] = -x
    J[3, 1] = y
    J[3, 2] = x
    J[3, 3] = -β
end

function rossler!(du, u, p, t)
    x, y, z = u
    a, b, c = p
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b + z * (x - c)
end

function rossler_jacobian!(J, u, p, t)
    x, y, z = u
    a, b, c = p
    J[1, 1] = 0.0
    J[1, 2] = -1.0
    J[1, 3] = -1.0
    J[2, 1] = 1.0
    J[2, 2] = a
    J[2, 3] = 0.0
    J[3, 1] = z
    J[3, 2] = 0.0
    J[3, 3] = x - c
end

function compare_clv_subspaces(
    ginelli, wolfe; tol, indices, offset_ginelli=0, offset_wolfe=0, subspace_dim=2
)
    k = size(ginelli.V[1], 2)
    dim = min(subspace_dim, k)
    for idx in indices
        G = ginelli.V[idx + offset_ginelli][:, 1:dim]
        W = wolfe.V[idx + offset_wolfe][:, 1:dim]
        QG = qr(G).Q[:, 1:dim]
        QW = qr(W).Q[:, 1:dim]
        svals = svd(transpose(QG) * QW).S
        @test minimum(svals) > tol
    end
end

function compare_flow_alignment(wolfe, f!; p, tol, indices, offset_wolfe=0, t_value=0.0)
    for idx in indices
        x = wolfe.x[idx + offset_wolfe]
        du = similar(x)
        f!(du, x, p, t_value)
        v = wolfe.V[idx + offset_wolfe][:, 2]
        du_norm = norm(du)
        v_norm = norm(v)
        @test du_norm > 0
        @test v_norm > 0
        du ./= du_norm
        v ./= v_norm
        @test abs(dot(du, v)) > tol
    end
end

@testset "Kuptsov-Parlitz matches Ginelli (Lorenz)" begin
    u0 = [1.0, 1.0, 1.0]
    p = (10.0, 28.0, 8.0 / 3.0)
    diffeq = (alg=Tsit5(), abstol=1e-9, reltol=1e-9)
    ds = CoupledODEs(lorenz!, u0, p; diffeq=diffeq)

    N = 10
    tau_steps = 100
    Δt = 0.1
    Ttr = 20.0
    Ttr_extra = 50
    N_ginelli = N + tau_steps
    ginelli = clv(TangentDynamicalSystem(ds; J=lorenz_jacobian!), N_ginelli;
        Δt=Δt, Ttr=Ttr, Ttr_bkw=Ttr_extra
    )
    wolfe_V, wolfe_x = clv_wolfe_samelson(TangentDynamicalSystem(ds; J=lorenz_jacobian!), N;
        Δt=Δt, Ttr=Ttr, Ttr_fsv=tau_steps, adjoint_substeps=10, traj=true
    )
    wolfe = (V=wolfe_V, x=wolfe_x)

    indices = unique([1, Int(ceil(N / 2)), N])
    compare_clv_subspaces(ginelli, wolfe;
        tol=0.99, indices=indices, offset_ginelli=tau_steps, offset_wolfe=1
    )
    compare_flow_alignment(wolfe, lorenz!; p=p, tol=0.3, indices=indices, offset_wolfe=1)
end

@testset "Kuptsov-Parlitz matches Ginelli (Rossler)" begin
    u0 = [0.1, 0.0, 0.0]
    p = (0.2, 0.2, 5.7)
    diffeq = (alg=Tsit5(), abstol=1e-9, reltol=1e-9)
    ds = CoupledODEs(rossler!, u0, p; diffeq=diffeq)

    N = 10
    tau_steps = 100
    Δt = 0.1
    Ttr = 20.0
    Ttr_extra = 50
    N_ginelli = N + tau_steps
    ginelli = clv(TangentDynamicalSystem(ds; J=rossler_jacobian!), N_ginelli;
        Δt=Δt, Ttr=Ttr, Ttr_bkw=Ttr_extra
    )
    wolfe_V, wolfe_x = clv_wolfe_samelson(TangentDynamicalSystem(ds; J=rossler_jacobian!), N;
        Δt=Δt, Ttr=Ttr, Ttr_fsv=tau_steps, adjoint_substeps=10, traj=true
    )
    wolfe = (V=wolfe_V, x=wolfe_x)

    indices = unique([1, Int(ceil(N / 2)), N])
    compare_clv_subspaces(ginelli, wolfe;
        tol=0.99, indices=indices, offset_ginelli=tau_steps, offset_wolfe=1
    )
    compare_flow_alignment(wolfe, rossler!; p=p, tol=0.3, indices=indices, offset_wolfe=1)
end
