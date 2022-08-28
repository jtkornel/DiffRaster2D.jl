# DiffRaster2D.jl - Differentiable 2d rasterizer in Julia

A basic rasterizer which lets you take any image and create a vector-graphics rendering from it, using the same techiques that form the backbone of deep learning.

By using [signed distance functions](https://en.wikipedia.org/wiki/Signed_distance_function) to represent geometric primitives, and rendering with basic anti-aliasing, one can get a fully differentiable graphics pipeline. That in turn makes it possible to set up a loss funtion for a rendered image against some reference image, and optimize the rendering parameters to minimize the loss using gradient search techniques. The SDF technique for graphics is not novel, but using it for differentiable 2d-rendering like here was recently described in [this paper by Tzu-Mao Li and collaborators](https://people.csail.mit.edu/tzumao/diffvg/diffvg.pdf) (Note that the SDF does not _necessarily imply_ conflation artifacts as the paper may seem to indicate, just that one has to take care not to _conflate_ pixel transparency, blending fraction, confidence or other uses of the alpha channel in the rasterizer. When targeting SVG this may be more challenging.)

This code demonstrates the concept using the [Zygote](https://github.com/FluxML/Zygote.jl) framework in [Julia](https://julialang.org). It is work in progress.

To test, run `examples/visualize_circle_recovery.jl`

This shows selected iterations when fitting a shaded circle:
<img width="449" alt="chelsea_circle_rendering" src="https://user-images.githubusercontent.com/8590187/146675817-8fd2c76a-5f85-4575-b3f2-12742e2cd54d.png">

For a more advanced example fitting a triangle mesh to a raster image, run `examples/triangle_mesh_optimization.jl`

By running for quite a few iterations you should get something like this:
<img width="449" alt="chelsea_circle_rendering" src="https://user-images.githubusercontent.com/8590187/146675817-8fd2c76a-5f85-4575-b3f2-12742e2cd54d.png">


