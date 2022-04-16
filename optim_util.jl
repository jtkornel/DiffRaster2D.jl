include("Diff2DRaster.jl")


function render_loss(ref_img :: Array{Float32, 3}, objects, points, loss_fun=mse) :: Float32
    l = loss_fun(render_objects(objects, points), ref_img) :: Float32
    return l 
end