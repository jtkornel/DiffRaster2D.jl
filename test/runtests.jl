using Test

@testset "Recover basic shapes" begin
include("./test_circle_recovery.jl")
include("./test_triangle_recovery.jl")
end;