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

export face_vertex_mesh, object_mesh

export signed_distance_function
export centre
export color_gradient
export parabolic_kernel_integral
export sdf_coverage
export render_objects
export image_sample_points

export mae, mse

# optim_util
export render_loss

# triangle_mesh_util
export triangulate_image
export sample_triangle_colors

# colortypes_img_util
export array_image, colortypes_image




end # module DiffRaster2D