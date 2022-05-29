using TestImages
using ImageShow
using Flux

include("Diff2DRaster.jl")
include("colortypes_img_util.jl")
include("triangle_mesh_util.jl")
include("optim_util.jl")


img = testimage("chelsea")

H, W = size(img)

fv_msh = triangulate_image(img)
img_arr = array_image(img)
fv_msh = sample_triangle_colors(img_arr, fv_msh)
points = image_sample_points(W, H)

ps = Flux.params(fv_msh)

opt = AMSGrad(0.6)


for iti in 1:128
    println("Iteration ", iti)
    println("Calculating gradient")
    gs = Flux.gradient(ps) do
        tr_msh = object_mesh{triangle{Vertex}}(fv_msh)
        l = render_loss(img_arr, tr_msh.objects, points)
        println("Loss (RMS) ", sqrt(l))
        return l
    end
    println("Updating")
    Flux.Optimise.update!(opt, ps, gs)
end

tr_msh = object_mesh{triangle{Vertex}}(fv_msh)
rn=render_objects(tr_msh.objects, points)
colortypes_image(rn/255, W, H)
