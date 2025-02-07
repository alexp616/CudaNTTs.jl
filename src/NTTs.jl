module NTTs

using CUDA
using Primes

export primitive_nth_root_of_unity
export NTTPlan
export INTTPlan
export plan_ntt
export ntt!
export intt!

include("reducers.jl")
include("utils/nttutils.jl")
include("gpuntt.jl")
include("nttkernels.jl")
include("oldgpuntt.jl")

end
