module ShimizuMoriokaSystem
export derivatives!, default_parameters, parameter_ranges, default_opn_params, initial_condition

# Shimizu-Morioka System
# dx/dt = y
# dy/dt = x - λy - xz
# dz/dt = -αz + x²
# Parameters: p = (α, λ, B) where B is unused (set to 0)
function derivatives!(du, u, p, t)
    x, y, z = u
    α, λ, _ = p
    du[1] = y
    du[2] = x - λ * y - x * z
    du[3] = -α * z + x^2
end

const default_parameters = [0.4, 0.725, 0.0]
const parameter_ranges = [(0.15, 0.65), (0.5, 0.95), (0.0, 0.0)]
const default_opn_params = (3, 4, 2, 1, 3, 1)
const initial_condition = [0.1, 0.1, 0.1]

end
