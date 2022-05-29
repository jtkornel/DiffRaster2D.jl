using TestImages
using ImageView
using ColorTypes
using ImageShow

include("../Diff2DRaster.jl")
include("../colortypes_img_util.jl")
include("../optim_util.jl")


img = testimage("chelsea")
H, W = size(img)

img_arr = array_image(img)

N=16
M=16
ts = [triangle{Vertex}([x*W/(N-1), y*H/(M-1)], [(x+1)*W/(N-1), y*H/(M-1)], [(x+0.5)*W/(N-1), (y+1)*H/(M-1)], color_shade([128.0f0, 128.0f0, 128.0f0])) for x in 0:(N-1) for y in 0:(M-1)]

points = image_sample_points(W, H)

par = Flux.params(objs)
opt = AMSGrad(0.6)

for ii in 1:30
    println("iteration $ii")

    gs = Flux.gradient(par) do
        l = render_loss(img_arr, objs, points)
        println("Loss (RMS): ", sqrt(l))
        return l 
    end

    Flux.Optimise.update!(opt, par, gs)
end


rn = render_objects(ts, W, H)
colortypes_image(rn/255, W, H)