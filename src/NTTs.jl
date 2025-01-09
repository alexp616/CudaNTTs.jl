module NTTs

using BitIntegers
using CUDA
using Primes

include("utils/utils.jl")
include("cpuntt.jl")
include("gpuntt.jl")

end
