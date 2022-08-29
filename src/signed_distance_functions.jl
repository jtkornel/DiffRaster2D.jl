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
    return sqrt.(v_pt[1].^2 .+ v_pt[2].^2 .+ 1.0f-12)
end

function edge_determinant(p, a, b)
    return ((b[2] .- a[2]).*p[1] .+ (a[1] .- b[1]).*p[2] .+ (a[2].*b[1] .- a[1].*b[2]))
end

function signed_distance_function(points, t :: triangle{Vertex}) :: Matrix{Float32}

    d_ab = point_line_distance(points, t.a, t.b)
    F_ab = edge_determinant(points, t.a, t.b)

    d_bc = point_line_distance(points, t.b, t.c)
    F_bc = edge_determinant(points, t.b, t.c)

    d_ca = point_line_distance(points, t.c, t.a)
    F_ca = edge_determinant(points, t.c, t.a)

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

function signed_distance_function(points, c :: circle)
    cr = c.c
    r = c.r
    xs = points[1]
    ys = points[2]
    esq = ((cr[1] .- xs).^2 .+ (cr[2] .- ys).^2)
    return sqrt.(esq .+ 1.0f-12) .- r
end