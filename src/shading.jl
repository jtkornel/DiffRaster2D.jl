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


function shading(points, shape, W, H) :: Vector{Matrix{Float32}}
    c = centre(shape)

    sd = signed_distance_function(([c[1]], [c[2]]), shape)
    d = abs(sd[1])
    xs = ((points[1].-c[1])/d) :: Matrix{Float32}
    ys = ((points[2].-c[2])/d) :: Vector{Float32}

    return [ m.s[1].*xs .+ m.s[2].*ys .+ m.s[3] for m in shape.csh.ms]  
end