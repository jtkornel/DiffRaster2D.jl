using MosaicViews
using ImageShow
using Flux

using DiffRaster2D

W, H = 40, 40

ref_circ = circle([15.0f0], [20.0f0,20.0f0], color_shade([mono_shade([-80, 80, 180.0f0]), mono_shade([-40, 40, 90.0f0]), mono_shade([0, 0,0.0f0])]))

circ = circle([25.0f0], [10.0f0,10.0f0], color_shade([mono_shade([0, 0, 0.0f0]), mono_shade([0, 0, 128.0f0]), mono_shade([0, 0, 0.0f0])]))


points = raster_sampling_grid(W, H)

ref_img = render([ref_circ], points)


circs = [circ]
ps = Flux.params(circ)

opt = AMSGrad(0.75)

images = []
for iti in 1:192
    gs = Flux.gradient(ps) do
        l = render_loss(ref_img, circs, points)
        println(l)
        return l
    end

    Flux.Optimise.update!(opt, ps, gs)

    if ((iti-1) % 1 == 0)
        img = render(circs, points)
        im = (colortypes_image(img/255, W, H))
        push!(images, im)
    end
end

gs = Flux.gradient(ps) do
    l = render_loss(ref_img, circs, points)
    println(l)
    return l
end

mosaicview(images..., ncol=16,rowmajor=true)