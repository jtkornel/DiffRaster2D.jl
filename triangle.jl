using Zygote
using ImageView
using ColorTypes
using TestImages
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

function signed_distance_function(ps, c :: circle) :: Vector{Float32}
    esq = [sum((c.c .- Vector{Float32}(p)).^2) for p in ps] :: Vector{Float32}
    return sqrt.(esq .+ 1.0f-12) .- c.r
end

function parabolic_kernel_integral(r :: Vector{Float32})
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
    points = [[floor(j/W), j%W] for j in 0:(W*H-1)]
    return render_objects(objs, points)
end

function mae(x,y) :: Float32
    e = x .- y
    ae = [abs.(x) for x in e]
    mae = sum(sum(ae))
    return mae
end

function mse(x,y)
    e = x .- y
    ae = [x.^2 for x in e]
    mse = sum(sum(ae))
    return mse
end

function create_render_loss(ref_img)
    H, W = size(ref_img)    
    loss_fun = objects -> mae(render_objects(objects, H, W), ref_img)

    return loss_fun
end

function colortypes_image(img_1d, W, H)
    return [ ColorTypes.RGB{Float32}(img_1d[1][(n-1)*W + m], img_1d[2][(n-1)*W + m], img_1d[3][(n-1)*W + m]) for n=1:H, m=1:W]
end



img = testimage("mandrill")
H, W = size(img)

img_1d = [ [Float32(x.r) for x in img'[:]], [Float32(x.g) for x in img'[:]], [Float32(x.b) for x in img'[:]]]*255

#lf = create_render_loss(img_1d)

#signed_distance_function([[0,0],[10,10],[12,11]],t)
#sdf_coverage([[0,0], [10,10], [12,11]],[t])

function sdf_coverage_test_loss(ps, objs)
    x=sdf_coverage(ps,objs)
    return sum(sum(x))
end

function render_test_loss(objs)
    ps=[[10.5,10.5], [10,10], [11,11]]
    x=render_objects(objs,ps)
    return sum(sum(x))
end

function mysum(v :: Vector{Float32}) :: Float32
    s = 0.0f0
    for ve in v
        s += ve :: Float32
    end
    return s
end

function render_loss(ref_img, sc :: scene) :: Float32
    l = mae(render_objects([sc.cs...], H, W), ref_img) :: Float32
    println("Render loss: ", l)
    return l 
end

function nonneg_regularizer(v :: Float32)
    return v < 0 ? 128*v.^2 : v.^2
end

function nonneg_regularizer(col :: Vector{Float32})
    return sum([nonneg_regularizer(c) for c in col])
end

function nonneg_regularizer(cir :: circle)
    return nonneg_regularizer(cir.r) + nonneg_regularizer(cir.color)
end

function nonneg_regularizer(tr :: triangle)
    return nonneg_regularizer(tr.color) + (nonneg_regularizer(tr.a) + nonneg_regularizer(tr.b) + nonneg_regularizer(tr.c))/3
end

function scene_regularizer(sc :: scene) :: Float32

    sc_objs = [sc.cs...]
    
    crs = [nonneg_regularizer(o) for o in sc_objs]

    cr = sum(crs)

    println("Regularizer: ",cr*0.1)
    return cr*0.1f0
end

#print(gradient(render_test_loss, [circ]))
#tim_1d = render_objects([circ,t], 90, 60)
#print(gradient(o -> sum(sum(render_objects(o,90,60))), [t,circ]))
#tim_2d = colortypes_image(tim_1d, 90, 60)
#imshow(tim_2d)



tria = triangle([10.0f0,10.0f0],[200.0f0,0.0f0],[100.0f0,300.0f0], [128.0f0, 12.0f0, 12.0f0])

circs = [circle(35.0f0, [y*80.0f0, x*80.0f0], [64.0f0, 64.0f0, 64.0f0]) for x in 1:5 for y in 1:5]

scn = scene(tria, circs)

function gradient_iteration()
    global scn
    gs = gradient((sc)->(render_loss(img_1d, sc)+scene_regularizer(sc)), scn)[1]

    d = 0.5f-3
#    t = triangle(scn.t.a .- d*gs.t.a, scn.t.b .- d*gs.t.b, scn.t.c .- d*gs.t.c, scn.t.color .- d*gs.t.color)
    
    cs = [circle(scn.cs[ci].r .- d*gs.cs[ci].r, scn.cs[ci].c .- d*gs.cs[ci].c,  scn.cs[ci].color .- d*gs.cs[ci].color) for ci in 1:length(scn.cs)]

    scn = scene(tria, cs)
    println(scn)
    rn = render_objects([scn.t,scn.cs...], H, W)
    return colortypes_image(rn/255, W, H);
end

for ii in 1:4
    println("iteration $ii")
    rn2 = gradient_iteration()
end