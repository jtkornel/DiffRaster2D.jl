using TestImages
using ImageView
using ColorTypes
using ImageShow

include("Diff2DRaster.jl")

function colortypes_image(img_arr, W, H)
    return [ ColorTypes.RGB{Float32}(img_arr[1,n,m], img_arr[2,n,m], img_arr[3,n,m]) for n=1:H, m=1:W]
end

function render_loss(ref_img, sc :: scene, W, H) :: Float32
    l = mse(render_objects([sc.ts...], W, H), ref_img) :: Float32
    println("Render loss: ", l)
    return l 
end

img = testimage("chelsea")
H, W = size(img)
img_arr = [ [Float32(x.r) for x in img];;; [Float32(x.g) for x in img];;; [Float32(x.b) for x in img]]*255
img_arr = permutedims(img_arr, (3, 1,2))

N=16
M=16
ts = [triangle([x*W/(N-1), y*H/(M-1)], [(x+1)*W/(N-1), y*H/(M-1)], [(x+0.5)*W/(N-1), (y+1)*H/(M-1)], [128.0f0, 128.0f0, 128.0f0]) for x in 0:(N-1) for y in 0:(M-1)]
cs = [circle(10, [10,10], [128.0f0, 128.0f0, 128.0f0])]

scn = scene(ts, cs)

function gradient_iteration(scne, d=1)
    gs = gradient((sc)->(render_loss(img_arr, sc, W, H)), scne)[1]

    ts = [triangle(scne.ts[ti].a .- d*gs.ts[ti].a, scne.ts[ti].b .- d*gs.ts[ti].b, scne.ts[ti].c .- d*gs.ts[ti].c, scne.ts[ti].color .- d*gs.ts[ti].color) for ti in 1:length(scne.ts)]
    #cs = [circle(scne.cs[ci].r .- d*gs.cs[ci].r, scne.cs[ci].c .- d*gs.cs[ci].c,  scne.cs[ci].color .- d*gs.cs[ci].color) for ci in 1:length(scne.cs)]

    return scene(ts, cs)
end

for ii in 1:32
    global scn
    println("iteration $ii")
    @time scn=gradient_iteration(scn,4)
end

rn = render_objects([scn.ts...], W, H)
colortypes_image(rn/255, W, H)