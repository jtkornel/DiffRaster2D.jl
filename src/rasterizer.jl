using Zygote
using Flux

import Base.+
import Base.*
struct mono_shade
    s :: Vector{Float32} # slope/offset
end

Flux.@functor mono_shade

function mono_shade(constant_intensity :: Float32)
    return mono_shade([0.0f0,0.0f0, constant_intensity])
end

(+)(x :: mono_shade, y :: mono_shade) = mono_shade(x.s + y.s)
(*)(a :: Float32, x :: mono_shade) = mono_shade(a * x.s)
struct color_shade
    ms :: Vector{mono_shade}
end

Flux.@functor color_shade
Flux.trainable(csh::color_shade) = (ms = csh.ms,)

function color_shade(constant_eq_rgb :: Float32)
    return color_shade([constant_eq_rgb for _ in 1:3])
end

function color_shade(constant_rgb :: Vector{Float32})
    return color_shade([mono_shade(c) for c in constant_rgb])
end

(+)(x :: color_shade, y :: color_shade) = color_shade(x.ms .+ y.ms)
(*)(a :: Float32, x :: color_shade) = color_shade( [a * m for m in x.ms])

struct circle
    r :: Vector{Float32} # Scalars/immutable parameters not supported in Flux 
    c :: Vector{Float32}
    csh :: color_shade  
end

Flux.@functor circle

VertexRef = UInt
Vertex = Vector{Float32}
struct triangle{PointType}
    a :: PointType
    b :: PointType
    c :: PointType
    csh :: color_shade 
end

Flux.@functor triangle{Vertex}

Flux.@functor triangle{VertexRef}
Flux.trainable(tr::triangle{VertexRef}) = (csh=tr.csh,)

function triangle{Vertex}(tri :: triangle{VertexRef}, vertices :: Matrix{Float32})
    return triangle{Vertex}(vertices[tri.a,:], vertices[tri.b,:], vertices[tri.c,:], tri.csh)
end

function centre_color(tri_vr :: triangle{VertexRef}, vertices :: Matrix{Float32}, img :: Array{Float32,3})
    tri_v = triangle{Vertex}(tri_vr, vertices)
    c = round.(centre(tri_v))

    color = img[:, UInt(c[2]), UInt(c[1])]
    return color
end

function sample(img :: Array{Float32, 3}, c :: Float32, r:: Float32)
    c = clamp(round(c), 1, size(img)[3])
    r = clamp(round(r), 1, size(img)[2])

    img[:, UInt(r), UInt(c)]
end

function average_color(tri_vr :: triangle{VertexRef}, vertices :: Matrix{Float32}, img :: Array{Float32,3})
    tri_v = triangle{Vertex}(tri_vr, vertices)
    cent = centre(tri_v)
    a = tri_v.a
    b = tri_v.b
    c = tri_v.c

    c_cent = sample(img, cent[1], cent[2])
    c_a = sample(img, a[1], a[2])
    c_b = sample(img, b[1], b[2])
    c_c = sample(img, c[1], c[2])

    return 0.25f0*(c_cent + c_a + c_b + c_c)
end

struct face_vertex_mesh{FaceType}
    faces :: Vector{FaceType}
    vertices :: Matrix{Float32}
end

Flux.@functor face_vertex_mesh{triangle{VertexRef}}
#Flux.trainable(fvm::face_vertex_mesh{triangle{VertexRef}}) = (vertices=fvm.vertices, )
struct shape_mesh{ShapeType}
    shapes :: Vector{ShapeType}
end

function shape_mesh{triangle{Vertex}}(fvmesh :: face_vertex_mesh{triangle{VertexRef}})
    return shape_mesh{triangle{Vertex}}([ triangle{Vertex}(f, fvmesh.vertices) for f in fvmesh.faces])
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

function centre(t :: triangle{Vertex}) :: Vector{Float32}
    return (t.a + t.b + t.c)/3
end

function centre(c :: circle) :: Vector{Float32}
    return c.c
end

function shading(points, shape, W, H) :: Vector{Matrix{Float32}}
    c = centre(shape)

    sd = signed_distance_function(([c[1]], [c[2]]), shape)
    d = abs(sd[1])
    xs = ((points[1].-c[1])/d) :: Matrix{Float32}
    ys = ((points[2].-c[2])/d) :: Vector{Float32}

    return [ m.s[1].*xs .+ m.s[2].*ys .+ m.s[3] for m in shape.csh.ms]  
end

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