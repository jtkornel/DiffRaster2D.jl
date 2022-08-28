module DiffRaster2D

include("./rasterizer.jl")
include("./colortypes_img_util.jl")
include("./triangle_mesh_util.jl")
include("./optim_util.jl")

export mono_shade, color_shade

export circle

export Vertex, VertexRef
export triangle

export centre_color, average_color

export face_vertex_mesh, shape_mesh

export signed_distance_function
export centre
export shading
export parabolic_kernel_integral
export coverage
export render
export raster_sampling_grid

export mae, mse

# optim_util
export render_loss

# triangle_mesh_util
export triangulate_image
export sample_triangle_colors

# colortypes_img_util
export array_image, colortypes_image




end # module DiffRaster2D