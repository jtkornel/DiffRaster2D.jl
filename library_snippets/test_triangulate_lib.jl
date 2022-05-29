
using Triangulate
using CairoMakie
using GeometryBasics 


W, H = 64, 64

ps = reshape([Float64(x*d+y*(1-d)) for y in 0:8:H, x in 0:8:W, d in 0:1], (:,2))

triin = TriangulateIO() 
triin.pointlist = ps'
msh, _ = triangulate("vcDQ", triin)

# Convert TriangleMesh.TriMesh to GeometryBasics.Mesh
pts = [Point(val[1], val[2]) for val in eachcol(msh.pointlist)]
fcs = [TriangleFace(val[1], val[2], val[3]) for val in eachcol(msh.trianglelist)]
gb_msh = GeometryBasics.Mesh(pts, fcs)

# Plot mesh
scn = mesh(gb_msh, color = 1 : length(gb_msh.position))
wireframe!(gb_msh)
current_figure()