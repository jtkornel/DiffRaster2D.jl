include("Diff2DRaster.jl")


c = circle(5.0f0, [10.0f0, 10.0f0], [64.0f0, 64.0f0, 64.0f0])
xs = [x/10.0f0 for x in 0:500]
ys = [y/10.0f0 for y in 0:500]

ps = (collect(xs'), ys)

function sdf_load(ps, c)
    d = sdf_coverage(ps,[c])
    return d
end