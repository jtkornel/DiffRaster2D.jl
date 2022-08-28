function mae(x :: Array{Float32,3}, y :: Array{Float32,3}) :: Float32
    e = abs.(x .- y)
    mae = sum(e)/length(e)
    return mae
end

function mse(x :: Array{Float32,3}, y :: Array{Float32,3}) :: Float32
    e = (x .- y) .^2
    mse = sum(e)/length(e)

    return mse
end

function render_loss(ref_img :: Array{Float32, 3}, shapes, points, loss_fun=mse) :: Float32
    l = loss_fun(render(shapes, points), ref_img) :: Float32
    return l 
end