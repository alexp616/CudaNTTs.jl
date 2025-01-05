include("src/NTTs.jl")
# using CUDA
function run()
    # arr1 = [i - 1 for i in 1:64]
    # arr2 = copy(arr1)
    # prime = 193
    # npru = 94
    arr1 = [i - 1 for i in 1:32]
    arr2 = copy(arr1)
    prime = 97
    npru = 51
    NTTs.ntt(arr1, npru, prime)
    NTTs.slow_ntt(arr2, npru, prime)

    display(arr1)
    # display(arr2)
    @assert arr1 == arr2 "expected output: $arr2"

    return nothing
end

run()

# function mykernel(arr::CuDeviceVector{UInt32})
#     idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1

#     @cuprintln(typeof(threadIdx()))
#     @cuprintln(typeof(blockIdx()))
#     @cuprintln(typeof(blockDim()))
#     @cuprintln(typeof(idx))

#     return nothing
# end

# function sus()

#     @cuda mykernel(CuArray(UInt32.([1])))

# end

# sus()

# function runtests()
#     arr1 = [i for i in 1:8]
#     arr2 = copy(arr1)
#     prime = 17
#     npru = 9
#     NTTs.ntt(arr1, npru, prime)
#     NTTs.slow_ntt(arr2, npru, prime)

#     # display(arr1)
#     # display(arr2)
#     @assert arr1 == arr2
# end

# function test1()
#     arr1 = [i for i in 1:16]
#     arr2 = copy(arr1)
#     NTTs.ntt(arr1, 3, 17)
#     NTTs.slow_ntt(arr2, 9, 17)

#     @assert arr1 == arr2
# end

# run()
# test1()