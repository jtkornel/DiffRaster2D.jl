![ci tests](https://github.com/jtkornel/DiffRaster2D.jl/actions/workflows/main.yml/badge.svg)  [![codecov](https://codecov.io/gh/jtkornel/DiffRaster2D.jl/branch/main/graph/badge.svg?token=E59A020AY9)](https://codecov.io/gh/jtkornel/DiffRaster2D.jl)

# DiffRaster2D.jl - Differentiable 2d rasterizer in Julia

A basic rasterizer which lets you take any image and and fit the parameters of a vector-graphics model to it, using the same techiques that form the backbone of deep learning.

By using [signed distance functions](https://en.wikipedia.org/wiki/Signed_distance_function) (SDF) to represent geometric primitives, and rendering with 1d anti-aliasing, one can get a fully differentiable graphics pipeline. That in turn makes it possible to set up a loss funtion for a rendered image against some reference image, and optimize the rendering parameters to minimize the loss using gradient search techniques. The SDF technique for graphics is not novel, but using it for differentiable 2d-rendering like here was recently described in [this paper by Tzu-Mao Li and collaborators](https://people.csail.mit.edu/tzumao/diffvg/diffvg.pdf) 


This toolbox shows the concept using the [Zygote](https://github.com/FluxML/Zygote.jl) framework in [Julia](https://julialang.org).

For now circle and triangle primitives are supported, in addition to face-vertex triangle meshes.

Each primitive can be shaded with a linear color gradient which can also be optimized.

To do a basic test, run `examples/visualize_circle_recovery.jl`

This shows selected iterations when fitting a shaded circle:  
<img alt="visualize_circle_recovery" src="https://raw.githubusercontent.com/jtkornel/DiffRaster2D.jl/main/media/visualize_circle_recovery.png">

For a more advanced example fitting a triangle mesh to a raster image, run `examples/triangle_mesh_optimization.jl`

By running for quite a few iterations you should get something like this:  
<img alt="triangle_mesh_optimization" src="https://raw.githubusercontent.com/jtkornel/DiffRaster2D.jl/main/media/triangle_mesh_optimization.png">

Note that since the vertex positions are unconstrained, the mesh quality may end up being so-so with some very elongated rectangles.

## Todo-list

* A better selection of SDF primitives
* Documentation
* Integration with other Julia geometry packages
* Mesh quality regularization
* Tailor for more efficient gradients from Zygote 
* Faster rendering with tiles or kD-tree
