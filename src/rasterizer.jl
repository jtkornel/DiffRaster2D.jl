
function parabolic_kernel_integral(r :: Matrix{Float32}) :: Matrix{Float32}
    rc = clamp.(r, -1, 1)
    return 0.5f0 .+ 0.25f0(rc.^3 - 3rc)
end

function coverage(points, shape) :: Matrix{Float32}
    return parabolic_kernel_integral(signed_distance_function(points, shape))
end

function render(shapes, points) :: Array{Float32,3}

    W = length(points[1])
    H = length(points[2])

    r_sum = zeros(Float32, (H, W))
    g_sum = zeros(Float32, (H, W))
    b_sum = zeros(Float32, (H, W))
    cov_sum = zeros(Float32, (H, W))

    for s in shapes
        cov = coverage(points, s)

        cov_sum = cov_sum + cov

        shade = shading(points, s, W, H)

        r_sum = r_sum + shade[1] .* cov
        g_sum = g_sum + shade[2] .* cov
        b_sum = b_sum + shade[3] .* cov
    end

    cov_sum = max.(1.0f0, cov_sum)

    cov_sum = reshape(cov_sum, (1,H,W))
    return permutedims(reshape([r_sum; g_sum; b_sum], (H,3,W)), (2,1,3))./cov_sum
end

function raster_sampling_grid(W, H)
    xs = [Float32(x) for x in 0:W-1] :: Vector{Float32}
    ys = [Float32(y) for y in 0:H-1] :: Vector{Float32}
    
    points = (collect(xs'), ys)
    return points
end

function render(shapes, W, H) :: Array{Float32,3}
    points = raster_sampling_grid(W, H)
    return render(shapes, points) 
end