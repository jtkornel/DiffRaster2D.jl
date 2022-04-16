using TestImages
using ImageView
using ColorTypes
using ImageShow

include("Diff2DRaster.jl")

function colortypes_image(img_arr, W, H)
    return [ ColorTypes.RGB{Float32}(img_arr[1,n,m], img_arr[2,n,m], img_arr[3,n,m]) for n=1:H, m=1:W]
end

function render_loss(ref_img, ts, W, H) :: Float32
    l = mse(render_objects(ts, W, H), ref_img) :: Float32
    println("Render loss: ", l)
    return l 
end

function updated_mono_shade(x :: mono_shade, grad, d)
    return mono_shade(x.s - d*grad.s)
end

function updated_color_shade(x :: color_shade, grad, d)
    return color_shade([updated_mono_shade(x.ms[1], grad.ms[1], d),  updated_mono_shade(x.ms[2], grad.ms[2], d), updated_mono_shade(x.ms[3], grad.ms[3], d)]) # painful hack
end

img = testimage("chelsea")
H, W = size(img)
img_arr = [ [Float32(x.r) for x in img];;; [Float32(x.g) for x in img];;; [Float32(x.b) for x in img]]*255
img_arr = permutedims(img_arr, (3, 1,2))

N=16
M=16
ts = [triangle{Vertex}([x*W/(N-1), y*H/(M-1)], [(x+1)*W/(N-1), y*H/(M-1)], [(x+0.5)*W/(N-1), (y+1)*H/(M-1)], color_shade([128.0f0, 128.0f0, 128.0f0])) for x in 0:(N-1) for y in 0:(M-1)]

function gradient_iteration(ts, d=1)
    gs = gradient((t)->(render_loss(img_arr, t, W, H)), ts)[1]

    upd_ts = [triangle{Vertex}(ts[ti].a .- d*gs[ti].a, ts[ti].b .- d*gs[ti].b, ts[ti].c .- d*gs[ti].c, updated_color_shade(ts[ti].csh, gs[ti].csh, d)) for ti in 1:length(ts)]

    return upd_ts
end

for ii in 1:400
    global ts
    println("iteration $ii")
    @time ts=gradient_iteration(ts, 4.0f0)
end

rn = render_objects(ts, W, H)
colortypes_image(rn/255, W, H)