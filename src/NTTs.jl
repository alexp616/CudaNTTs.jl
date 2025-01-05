module NTTs

using BitIntegers
using CUDA

include("utils/utils.jl")
include("cpuntt.jl")
include("gpuntt.jl")

end
