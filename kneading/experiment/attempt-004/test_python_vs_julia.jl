using DelimitedFiles
using CairoMakie

ee_py = readdlm("kneading/experiment/attempt-004/python_ee.txt")[:, 1]
ee_jl = readdlm("kneading/experiment/attempt-004/julia_ee.txt")[:, 1]

fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, ee_py, color=:blue, label="Python EE")
lines!(ax, ee_jl, color=:red, label="Julia EE")
axislegend(ax)
save("kneading/experiment/attempt-004/ee_comparison.png", fig)
