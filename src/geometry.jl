using Flux

struct circle
    r :: Vector{Float32} # Scalars/immutable parameters not supported in Flux 
    c :: Vector{Float32}
    csh :: color_shade  
end

Flux.@functor circle

VertexRef = UInt
Vertex = Vector{Float32}
struct triangle{CornerType}
    a :: CornerType
    b :: CornerType
    c :: CornerType
    csh :: color_shade 
end

Flux.@functor triangle{Vertex}

Flux.@functor triangle{VertexRef}
Flux.trainable(tr::triangle{VertexRef}) = (csh=tr.csh,)

function triangle{Vertex}(tri :: triangle{VertexRef}, vertices :: Matrix{Float32})
    return triangle{Vertex}(vertices[tri.a,:], vertices[tri.b,:], vertices[tri.c,:], tri.csh)
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

function centre(t :: triangle{Vertex}) :: Vector{Float32}
    return (t.a + t.b + t.c)/3
end

function centre(c :: circle) :: Vector{Float32}
    return c.c
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