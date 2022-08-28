using DiffRaster2D
using Flux

W, H = 40, 40

ref_fv_msh = face_vertex_mesh{triangle{VertexRef}}([triangle{VertexRef}(1, 2, 3, color_shade([mono_shade([-50,0,35.0f0]), mono_shade([-50,0,128.0f0]), mono_shade([-50,0,128.0f0])]))],
                                                [10 10; 30 10; 20 30])

fv_msh = face_vertex_mesh{triangle{VertexRef}}([triangle{VertexRef}(1, 2, 3, color_shade(64.0f0))],
                                                   [3 3; 33 15; 20 20])

points = image_sample_points(W, H)

ref_tr_msh = object_mesh{triangle{Vertex}}(ref_fv_msh)
ref_img = render_objects(ref_tr_msh.objects, points)

ps = Flux.params(fv_msh)

opt = opt = AMSGrad(0.75)

images = []
for iti in 1:128
    gs = Flux.gradient(ps) do
        tr_msh = object_mesh{triangle{Vertex}}(fv_msh)
        l = render_loss(ref_img, tr_msh.objects, points)
        println(l)
        return l
    end

    Flux.Optimise.update!(opt, ps, gs)
end

tr_msh = object_mesh{triangle{Vertex}}(fv_msh)
l = render_loss(ref_img, tr_msh.objects, points)

@test l < 0.01