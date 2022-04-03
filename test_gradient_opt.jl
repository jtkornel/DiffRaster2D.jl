using Zygote
using ImageView
using ImageShow
using ColorTypes
using TestImages
using Flux
using FileIO

include("Diff2DRaster.jl")


function render_loss(ref_img :: Array{Float32, 3}, sc :: scene, points) :: Float32
    l = mse(render_objects([sc.cs...], points), ref_img) :: Float32
    println("Render loss: ", l)
    return l 
end

function nonneg_regularizer(v :: Float32)
    return v < 0 ? 128*v.^2 : v.^2
end

function nonneg_regularizer(col :: Vector{Float32})
    return sum([nonneg_regularizer(c) for c in col])
end

function nonneg_regularizer(cs :: mono_shade)
    return nonneg_regularizer(cs.o/1280.0f0) + sum((cs.s).^2)
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

function scene_regularizer(sc :: scene) :: Float32

    sc_objs = [sc.cs...]
    
    crs = [nonneg_regularizer(o) for o in sc_objs]

    cr = sum(crs)/length(crs)

    reg = cr*0.01f0
    println("Regularizer: ", reg)
    return reg
end

function colortypes_image(img_arr, W, H)
    return [ ColorTypes.RGB{Float32}(img_arr[1,n,m], img_arr[2,n,m], img_arr[3,n,m]) for n=1:H, m=1:W]
end

function updated_mono_shade(x :: mono_shade, grad, d)
    return mono_shade(x.s - d*grad.s, x.o - d*grad.o)
end

function updated_color_shade(x :: color_shade, grad, d)
    return color_shade([updated_mono_shade(x.ms[1], grad.ms[1], d),  updated_mono_shade(x.ms[2], grad.ms[2], d), updated_mono_shade(x.ms[3], grad.ms[3], d)]) # painful hack
end

function gradient_iteration(scne, img, points, d=1.0f0)
    println("Calculating gradient...")

    gs = gradient((sc)->(render_loss(img, sc, points)+scene_regularizer(sc)), scne)[1]
    println("Updating scene...")
    #ts = [triangle(scne.ts[ti].a .- d*gs.ts[ti].a, scne.ts[ti].b .- d*gs.ts[ti].b, scne.ts[ti].c .- d*gs.ts[ti].c, scne.ts[ti].csh - d*gs.ts[ti].csh) for ti in 1:length(scne.ts)]
    dummy_tri = [triangle([10.0f0,10.0f0],[200.0f0,0.0f0],[100.0f0,300.0f0], color_shade([128.0f0, 12.0f0, 12.0f0]))]

    cs = [circle(scne.cs[ci].r .- d*gs.cs[ci].r, scne.cs[ci].c .- d*gs.cs[ci].c,  updated_color_shade(scne.cs[ci].csh, gs.cs[ci].csh, d)) for ci in 1:length(scne.cs)]

    return scene(dummy_tri, cs)
end

function show_render(scn, W, H)
    rn = render_objects([scn.cs...], W, H)
    return colortypes_image(rn/255, W, H)
end

function optimize_scene_to_image()

    img = testimage("chelsea")
    H, W = size(img)
    
    img_arr = [ [Float32(x.r) for x in img];;; [Float32(x.g) for x in img];;; [Float32(x.b) for x in img]]*255
    img_arr = permutedims(img_arr, (3, 1,2))

    N=8
    M=8

    r = max(1.4f0*W/(N*2), 1.3f0*H/(M*2))
    circs = [circle(r, [x*W/(N-1), y*H/(M-1)], color_shade([128.0f0, 128.0f0, 128.0f0])) for x in 0:(N-1) for y in 0:(M-1)]
    dummy_tri = [triangle([10.0f0,10.0f0],[200.0f0,0.0f0],[100.0f0,300.0f0], color_shade([128.0f0, 12.0f0, 12.0f0]))]

    scn = scene(dummy_tri, circs)

    xs = [Float32(x) for x in 0:W-1] :: Vector{Float32}
    ys = [Float32(y) for y in 0:H-1] :: Vector{Float32}
    
    points = (collect(xs'), ys)

    for ii in 1:100
        println("iteration $ii")
        scn=gradient_iteration(scn, img_arr, points, 4.0f0)
    end

    show_render(scn, W, H)
end

optimize_scene_to_image()