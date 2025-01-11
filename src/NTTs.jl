module NTTs

using BitIntegers
using CUDA
using Primes

export primitive_nth_root_of_unity
export NTTPlan
export INTTPlan
export plan_ntt
export ntt!
export intt!

include("utils/utils.jl")
include("gpuntt.jl")
include("cpuntt.jl")

end
