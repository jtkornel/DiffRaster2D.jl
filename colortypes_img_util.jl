using ColorTypes


function array_image(coltyp_img) ::Array{Float32, 3}
    img_arr = [ [Float32(x.r) for x in coltyp_img];;; [Float32(x.g) for x in coltyp_img];;; [Float32(x.b) for x in coltyp_img]]*255
    img_arr = permutedims(img_arr, (3, 1,2))
    return img_arr
end

function colortypes_image(img_arr, W, H)
    return [ ColorTypes.RGB{Float32}(img_arr[1,n,m], img_arr[2,n,m], img_arr[3,n,m]) for n=1:H, m=1:W]
end