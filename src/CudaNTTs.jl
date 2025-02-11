module CudaNTTs

using CUDA
using Primes

export primitive_nth_root_of_unity
export NTTPlan
export INTTPlan
export plan_ntt
export ntt!
export intt!

global const INTTYPES = Union{UInt32, UInt64, Int32, Int64}
global const o = Int32(1) # For kernel index arithmetic, LLVM usually optimizes it away anyways

include("reducers.jl")
include("utils/nttutils.jl")
include("gpuntt.jl")
include("nttkernels.jl")
include("oldgpuntt.jl")

end