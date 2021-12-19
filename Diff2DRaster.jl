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
    t :: triangle
    cs :: Vector{circle}
end

# u projected onto v, clamped to length of v
function vector_projection_clamped(u, v)
    x_u = u[1]
    y_u = u[2]
    x_v = v[1]
    y_v = v[2]

    # w = v * dot(u,v)/dot(v,v) = v * t

    t = (y_u*y_v + x_u*x_v)/(y_v*y_v + x_v*x_v+1.0f-12)

    t = clamp(t, 0, 1)

    x_w = x_v * t
    y_w = y_v * t

    return [x_w, y_w]
end

function point_line_projection(p, a, b)
    v_ap = p .- a
    v_ab = b .- a
    return vector_projection_clamped(v_ap, v_ab) .- v_ap
end

function point_line_distance(p, a, b)
    v_pt = point_line_projection(p, a, b)
    return sqrt(sum(v_pt.^2)+1.0f-12)
end

function edge_determinant(p, a, b)
    F = (a[2] - b[2])*p[1] + (b[1] - a[1])*p[2] + (a[1]*b[2] - a[2]*b[1])
    return -F
end

function edge_determinants(p, t :: triangle)
    F_ab = edge_determinant(p, t.a, t.b)
    F_bc = edge_determinant(p, t.b, t.c)
    F_ca = edge_determinant(p, t.c, t.a)
    return (F_ab, F_bc, F_ca)
end

function signed_distance_function(ps, t :: triangle)
    num_ps = length(ps)
    as = repeat([t.a], num_ps)
    bs = repeat([t.b], num_ps)
    cs = repeat([t.c], num_ps)
    d_ab = point_line_distance.(ps, as, bs)
    F_ab = edge_determinant.(ps, as, bs)

    d_bc = point_line_distance.(ps, bs, cs)
    F_bc = edge_determinant.(ps, bs, cs)

    d_ca = point_line_distance.(ps, cs, as)
    F_ca = edge_determinant.(ps, cs, as)

    s_ab = F_ab .< 0
    s_bc = F_bc .< 0
    s_ca = F_ca .< 0

    a_p = s_ab .& s_bc .& s_ca
    a_n = .!s_ab .& .!s_bc .& .!s_ca

    s = ((a_p .| a_n) .* -2) .+ 1

    d = minimum([d_ab d_bc d_ca],dims=2)

    return Vector{Float32}(s .* d[:])
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

function sdf_coverage(ps, objs)
    return [parabolic_kernel_integral(signed_distance_function(ps, obj)) for obj in objs]
end

function render_objects(objs, ps)
    n_objs = length(objs)
    coverages = sdf_coverage(ps, objs)
    colors = [[c for c in o.color] for o in objs]

    cov_sum = sum(coverages)
    cov_sum = [max(1.0f0, cs) for cs in cov_sum]

    r = sum([colors[oi][1]*coverages[oi] for oi in 1:n_objs])./cov_sum
    g = sum([colors[oi][2]*coverages[oi] for oi in 1:n_objs])./cov_sum
    b = sum([colors[oi][3]*coverages[oi] for oi in 1:n_objs])./cov_sum

    return [r,g,b]
end

function render_objects(objs, W, H)
    xs = [x for x in 0:W-1]
    ys = [y for y in 0:H-1]
    
    points = (xs', ys)
    return render_objects(objs, points)
end

function mae(x,y) :: Float32
    e = x .- y
    ae = sum([abs.(ec) for ec in e])/length(e)
    mae = sum(ae)/length(ae)
    return mae
end

function mse(x,y) :: Float32
    e = x .- y
    se = sum([ec.^2 for ec in e])/length(e)
    mse = sum(se)/length(se)
    return mse
end

function create_render_loss(ref_img)
    H, W = size(ref_img)    
    loss_fun = objects -> mae(render_objects(objects, H, W), ref_img)

    return loss_fun
end

