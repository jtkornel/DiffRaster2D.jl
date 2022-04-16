using Zygote
using ImageView
using ImageShow
using ColorTypes
using TestImages
using Flux
using FileIO

include("Diff2DRaster.jl")
include("optim_util.jl")

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

function scene_regularizer(objs) :: Float32
    
    crs = [nonneg_regularizer(o) for o in objs]

    cr = sum(crs)/length(crs)

    reg = cr*0.01f0
    println("Regularizer: ", reg)
    return reg
end


function updated_mono_shade(x :: mono_shade, grad, d)
    return mono_shade(x.s - d*grad.s)
end

function updated_color_shade(x :: color_shade, grad, d)
    return color_shade([updated_mono_shade(x.ms[1], grad.ms[1], d),  updated_mono_shade(x.ms[2], grad.ms[2], d), updated_mono_shade(x.ms[3], grad.ms[3], d)]) # painful hack
end

function gradient_iteration(objs, img, points, d=1.0f0)
    println("Calculating gradient...")

    gs = gradient((os)->(render_loss(img, os, points) + scene_regularizer(os)), objs)[1]
    println("Updating objects...")

    upd_objs = [circle(objs[ci].r .- d*gs[ci].r, objs[ci].c .- d*gs[ci].c,  updated_color_shade(objs[ci].csh, gs[ci].csh, d)) for ci in 1:length(objs)]

    return upd_objs
end

function show_render(objs, W, H)
    rn = render_objects(objs, W, H)
    return colortypes_image(rn/255, W, H)
end

function optimize_objects_to_image(img)

    H, W = size(img)
    
    img_arr = [ [Float32(x.r) for x in img];;; [Float32(x.g) for x in img];;; [Float32(x.b) for x in img]]*255
    img_arr = permutedims(img_arr, (3, 1,2))

    N=8
    M=8

    r = max(1.4f0*W/(N*2), 1.3f0*H/(M*2))
    objs = [circle(r, [x*W/(N-1), y*H/(M-1)], color_shade([128.0f0, 128.0f0, 128.0f0])) for x in 0:(N-1) for y in 0:(M-1)]

    points = image_sample_points(W, H)

    for ii in 1:100
        println("iteration $ii")
        objs=gradient_iteration(objs, img_arr, points, 4.0f0)
    end

    show_render(objs, W, H)
end

optimize_objects_to_image(testimage("chelsea"))