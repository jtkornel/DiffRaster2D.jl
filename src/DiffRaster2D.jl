module DiffRaster2D

include("./shading.jl")
export mono_shade, color_shade
export shading

include("./geometry.jl")
export circle

export Vertex, VertexRef
export triangle

export face_vertex_mesh, shape_mesh

export centre
export centre_color, average_color

include("./signed_distance_functions.jl")
export signed_distance_function

include("./rasterizer.jl")
export parabolic_kernel_integral
export coverage
export render
export raster_sampling_grid

include("./colortypes_img_util.jl")
export array_image, colortypes_image

include("./triangle_mesh_util.jl")
export triangulate_image
export sample_triangle_colors

include("./optim_util.jl")
export mae, mse
export render_loss


end # module DiffRaster2D