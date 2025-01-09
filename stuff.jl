include("src/NTTs.jl")
using CUDA

function run()
    len = 2^21
    # len = 2^13
    prime = UInt32(167772161)
    npru = NTTs.primitive_nth_root_of_unity(len, prime)

    arr1 = rand(UInt32(0):UInt32(prime - 1), len)
    # arr1 = ones(Int, len)
    # arr2 = copy(arr1)
    arr3 = CuArray(arr1)

    plan = NTTs.plan_cpuntt(len, prime, npru; numsPerThread = 2, threadsPerBlock = 256)
    plan2 = NTTs.plan_ntt(len, prime, npru)

    NTTs.ntt(arr1, plan)
    # @time NTTs.slow_ntt(arr2, npru, prime)
    NTTs.ntt!(arr3, plan2)

    # display(arr1)
    # display(arr3)

    # temp1 = sort(arr1)
    # temp2 = sort(arr2)
    # @assert temp1 == temp2 "values aren't even right"

    # display(arr1)
    # display(arr2)
    # perm = Int16.(get_perm(arr1, arr2))
    # println("perm: $perm)")
    # println("permbits: ")

    # print_perm_bits(perm)
    # for i in eachindex(perm)
    #     println("$(bitstring(perm[i])[4:8])\t$(bitstring(Int8(i - 1))[4:8])")
    # end
    # @assert arr1 == arr2 "output and expected output: \n$arr1 \n$arr2"
    @assert arr1 == Array(arr3)
    println("Test passed!")
    return nothing
end

function get_perm(arr1, arr2)
    perm = zeros(eltype(arr1), length(arr1))

    for i in eachindex(arr1)
        j = 1
        while true
            if arr2[j] == arr1[i]
                perm[i] = j - 1
                break
            end
            j += 1
        end
    end

    return perm
end


function print_perm_bits(arr)
    numBits = NTTs.intlog2(length(arr))
    for i in eachindex(arr)
        println("$(bitstring(Int16(arr[i]))[(16 - numBits + 1):16])\t$(bitstring(Int16(i - 1))[(16 - numBits + 1):16])")
    end
end

# function mykernel(arr::CuDeviceVector{UInt32})
#     idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x - 1

#     @cuprintln(typeof(threadIdx()))
#     @cuprintln(typeof(blockIdx()))
#     @cuprintln(typeof(blockDim()))
#     @cuprintln(typeof(idx))

#     return nothing
# end

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

run()
