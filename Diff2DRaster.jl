using Zygote
using Flux

struct triangle
    a :: Vector{Float32}
    b :: Vector{Float32}
    c :: Vector{Float32}
    color :: Vector{Float32}
end

struct circle
    r :: Float32
    c :: Vector{Float32}
    color :: Vector{Float32}
end

struct scene
    ts :: Vector{triangle}
    cs :: Vector{circle}
end

# u projected onto v, clamped to length of v
@views function vector_projection_clamped(u, v)
    x_u =  u[1]
    y_u =  u[2]
    x_v =  v[1]
    y_v =  v[2]

    # w = v * dot(u,v)/dot(v,v) = v * t

    t = @. (y_u*y_v + x_u*x_v)/(y_v*y_v + x_v*x_v + 1.0f-12)

    t = clamp.(t, 0, 1)

    x_w = x_v .* t
    y_w = y_v .* t

    return [x_w, y_w]
end

function point_line_projection(p, a, b)
    v_ap = [p[1] .- a[1], p[2] .- a[2]]
    v_ab = [b[1] .- a[1], b[2] .- a[2]]

    v_apPab = vector_projection_clamped(v_ap, v_ab)

    return [v_apPab[1] .- v_ap[1], v_apPab[2] .- v_ap[2]]
end

function point_line_distance(p, a, b)
    v_pt = point_line_projection(p, a, b)
    return sqrt.(v_pt[1].^2 + v_pt[2].^2 .+ 1.0f-12)
end

function edge_determinant(p, a, b)
    return ((b[2] .- a[2]).*p[1] .+ (a[1] .- b[1]).*p[2] .+ (a[2].*b[1] .- a[1].*b[2]))
end

function edge_determinants(p, t :: triangle)
    F_ab = edge_determinant(p, t.a, t.b)
    F_bc = edge_determinant(p, t.b, t.c)
    F_ca = edge_determinant(p, t.c, t.a)
    return (F_ab, F_bc, F_ca)
end

function signed_distance_function(ps, t :: triangle) :: Matrix{Float32}

    d_ab = point_line_distance(ps, t.a, t.b)
    F_ab = edge_determinant(ps, t.a, t.b)

    d_bc = point_line_distance(ps, t.b, t.c)
    F_bc = edge_determinant(ps, t.b, t.c)

    d_ca = point_line_distance(ps, t.c, t.a)
    F_ca = edge_determinant(ps, t.c, t.a)

    s_ab = F_ab .< 0
    s_bc = F_bc .< 0
    s_ca = F_ca .< 0

    a_p = (s_ab .+ s_bc .+ s_ca) .== 3
    a_n = (s_ab .+ s_bc .+ s_ca) .== 0

    s_p = ( a_p .* -2) .+ 1
    s_n = ( a_n .* -2) .+ 1

    s = s_p .* s_n

    d = dropdims(minimum(cat(d_ab, d_bc, d_ca,dims=3), dims=3), dims=3)

    return (s .* d)
end

function signed_distance_function(ps, c :: circle) :: Matrix{Float32}
    cr = c.c :: Vector{Float32}
    r = c.r :: Float32
    xs = ps[1] :: Matrix{Float32}
    ys = ps[2] :: Vector{Float32}
    esq = ((cr[1] .- xs).^2 .+ (cr[2] .- ys).^2)
    return sqrt.(esq .+ 1.0f-12) .- r
end

function parabolic_kernel_integral(r :: Matrix{Float32}) :: Matrix{Float32}
    rc = clamp.(r, -1, 1)
    return 0.5f0 .+ 0.25f0(rc.^3 - 3rc)
end

function sdf_coverage(ps, obj) :: Matrix{Float32}
    return parabolic_kernel_integral(signed_distance_function(ps, obj))
end

function render_objects(objs, ps) :: Array{Float32,3}

    W = length(ps[1])
    H = length(ps[2])

    r_sum = zeros(Float32, (H, W))
    g_sum = zeros(Float32, (H, W))
    b_sum = zeros(Float32, (H, W))
    cov_sum = zeros(Float32, (H, W))

    for o in objs
        cov = sdf_coverage(ps, o)

        cov_sum = cov_sum + cov

        r_sum = r_sum + o.color[1] * cov
        g_sum = g_sum + o.color[2] * cov
        b_sum = b_sum + o.color[3] * cov
    end

    cov_sum = max.(1.0f0, cov_sum)

    cov_sum = reshape(cov_sum, (1,H,W))
    return permutedims(reshape([r_sum; g_sum; b_sum], (H,3,W)), (2,1,3))./cov_sum
end

function render_objects(objs, W, H) :: Array{Float32,3}
    xs = [Float32(x) for x in 0:W-1] :: Vector{Float32}
    ys = [Float32(y) for y in 0:H-1] :: Vector{Float32}
    
    points = (collect(xs'), ys)
    return render_objects(objs, points) 
end

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

function create_render_loss(ref_img)
    H, W = size(ref_img)    
    loss_fun = objects -> mae(render_objects(objects, H, W), ref_img)

    return loss_fun
end

