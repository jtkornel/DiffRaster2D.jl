using DiffRaster2D
using Flux

W, H = 40, 40

ref_tris = [triangle{Vertex}([10, 10], [30, 10], [20, 30], color_shade([mono_shade([-50,0,35.0f0]), mono_shade([-50,0,128.0f0]), mono_shade([-50,0,128.0f0])])
)]

tris = [triangle{Vertex}([3, 3], [33, 15], [20, 20], color_shade(64.0f0))]

points = raster_sampling_grid(W, H)

ref_img = render(ref_tris, points)

ps = Flux.params(tris)

opt = opt = AMSGrad(0.75)

images = []
for iti in 1:128
    gs = Flux.gradient(ps) do
        l = render_loss(ref_img, tris, points)
        return l
    end

    Flux.Optimise.update!(opt, ps, gs)
end

l = render_loss(ref_img, tris, points)

@test l < 0.001