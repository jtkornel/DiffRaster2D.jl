using ImageShow
using TestImages
using Flux

using DiffRaster2D

function nonneg_regularizer(v :: Float32)
    return v < 0 ? 128*v.^2 : v.^2
end

function nonneg_regularizer(col :: Vector{Float32})
    return sum([nonneg_regularizer(c) for c in col])
end

function nonneg_regularizer(cs :: mono_shade)
    return nonneg_regularizer(cs.s[3]/1280.0f0) + sum((cs.s[1:2]).^2)
end

function nonneg_regularizer(cs :: color_shade)
    return sum([nonneg_regularizer(c) for c in cs.ms])
end

function nonneg_regularizer(cir :: circle)
    return nonneg_regularizer(cir.r) + nonneg_regularizer(cir.csh)
end

function nonneg_regularizer(tr :: triangle)
    return nonneg_regularizer(tr.color/1280.0f0) + (nonneg_regularizer(tr.a) + nonneg_regularizer(tr.b) + nonneg_regularizer(tr.c))/3
end

function scene_regularizer(shapes) :: Float32
    
    crs = [nonneg_regularizer(o) for o in shapes]

    cr = sum(crs)/length(crs)

    reg = cr*0.01f0

    return reg
end


function render_to_colortypes(shapes, W, H)
    rn = render(shapes, W, H)
    return colortypes_image(rn/255, W, H)
end

function optimize_shapes_to_image(img)

    H, W = size(img)
    
    img_arr = array_image(img)

    # Create inital grid of circles
    N, M = 12, 12
    r = max(1.4f0*W/(N*2), 1.3f0*H/(M*2))
    shapes = [circle([r], [x*W/(N-1), y*H/(M-1)], color_shade([128.0f0, 128.0f0, 128.0f0])) for x in 0:(N-1) for y in 0:(M-1)]

    points = raster_sampling_grid(W, H)

    par = Flux.params(shapes)
    opt = AMSGrad(0.6)

    for ii in 1:64
        println("iteration $ii")

        gs = Flux.gradient(par) do
            l = render_loss(img_arr, shapes, points)
            r = scene_regularizer(shapes)
            println("Loss (RMS): ", sqrt(l))
            println("Regularizer: ", r)
            return l + r
        end

        Flux.Optimise.update!(opt, par, gs)
    end

    render_to_colortypes(shapes, W, H)
end

optimize_shapes_to_image(testimage("mandrill"))