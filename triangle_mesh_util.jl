using Triangulate


function triangulate_image(img, N=8)
    H, W = size(img)

    ps = reshape([Float64(x*(1-d)+y*d) for x in range(0,W,length=N), y in range(0,H,length=N), d in 0:1], (:,2))

    triin = TriangulateIO() 
    triin.pointlist = ps'
    msh, _ = triangulate("vcDQ", triin)

    pts = msh.pointlist'
    fcs = [triangle{VertexRef}(vref[1], vref[2], vref[3], color_shade(128.0f0)) for (i,vref) in enumerate(eachcol(msh.trianglelist))]

    fv_msh = face_vertex_mesh{triangle{VertexRef}}(fcs, pts)

    return fv_msh
end