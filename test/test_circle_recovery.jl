using DiffRaster2D

using Flux

W, H = 40, 40

ref_circ = circle([15.0f0], [20.0f0,20.0f0], color_shade([mono_shade([-80, 80, 180.0f0]), mono_shade([-40, 40, 90.0f0]), mono_shade([0, 0,0.0f0])]))

circ = circle([25.0f0], [10.0f0,10.0f0], color_shade([mono_shade([0, 0, 0.0f0]), mono_shade([0, 0, 128.0f0]), mono_shade([0, 0, 0.0f0])]))


points = image_sample_points(W, H)

ref_img = render_objects([ref_circ], points)


circs = [circ]
ps = Flux.params(circ)

opt = AMSGrad(0.75)

for iti in 1:192
    gs = Flux.gradient(ps) do
        l = render_loss(ref_img, circs, points)
        return l
    end

    Flux.Optimise.update!(opt, ps, gs)
end

l = render_loss(ref_img, circs, points)

@test l < 0.32