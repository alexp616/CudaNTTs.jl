# Defining a new implementation
# To define a new FFT implementation in your own module, you should

# Define a new subtype (e.g. MyPlan) of AbstractFFTs.Plan{T} for FFTs and related transforms on arrays of T. This must have a pinv::Plan field, initially undefined when a MyPlan is created, that is used for caching the inverse plan.

# Define a new method AbstractFFTs.plan_fft(x, region; kws...) that returns a MyPlan for at least some types of x and some set of dimensions region. The region (or a copy thereof) should be accessible via fftdims(p::MyPlan) (which defaults to p.region), and the input size size(x) should be accessible via size(p::MyPlan).

# Define a method of LinearAlgebra.mul!(y, p::MyPlan, x) that computes the transform p of x and stores the result in y.

# Define a method of *(p::MyPlan, x), which can simply call your mul! method. This is not defined generically in this package due to subtleties that arise for in-place and real-input FFTs.

# If the inverse transform is implemented, you should also define plan_inv(p::MyPlan), which should construct the inverse plan to p, and plan_bfft(x, region; kws...) for an unnormalized inverse ("backwards") transform of x. Implementations only need to provide the unnormalized backwards FFT, similar to FFTW, and we do the scaling generically to get the inverse FFT.

# You can also define similar methods of plan_rfft and plan_brfft for real-input FFTs.

# To support adjoints in a new plan, define the trait AbstractFFTs.AdjointStyle. AbstractFFTs implements the following adjoint styles: AbstractFFTs.FFTAdjointStyle, AbstractFFTs.RFFTAdjointStyle, AbstractFFTs.IRFFTAdjointStyle, and AbstractFFTs.UnitaryAdjointStyle. To define a new adjoint style, define the methods AbstractFFTs.adjoint_mul and AbstractFFTs.output_size.

# multidimensional


# 
struct CPUNTTPlan
    
end

function slow_ntt(vec, npru::T, p::T) where T<:Integer
    n = length(vec)
    if !is_primitive_root(npru, p, n)
        throw(ArgumentError("$npru must be a primitive $n-th root of unity of p"))
    end
    mat = zeros(T, n, n)

    w = 1
    for r in 1:n
        temp = 1
        for c in 1:r
            mat[r, c] = temp
            temp = mul_mod(temp, w, p)
        end
        w = mul_mod(w, npru, p)
    end

    for r in 1:n
        for c in r+1:n
            mat[r, c] = mat[c, r]
        end
    end
    
    # temporary solution
    mat = BigInt.(mat)
    temp = BigInt.(vec)
    
    vec .= T.((mat * temp) .% p)

    return vec
end

function get_transposed_index(idx, rows, cols)
    originalRow = idx % rows
    originalCol = idx ÷ rows

    result = originalCol + originalRow * cols

    return result
end

# This way this works is:
# We want to perform an FFT of a vector with a power of 2 length.
# The simple radix-2 decimation in time algorithm will require an 
# auxiliary array, since the bit-reversal permutations need to be applied
# in order for the FFT to be done in-place.
# 
# The solution is to break the FFT into chunks bigger than 2, say
# R (for radix). We also have R be a power of 2.
# 
# Say R is 2^6, and our FFT is of length 2^20. We can convert this to a
# 2^6 x 2^6 x 2^6 x 2^2 FFT, with the additional step of applying 3 sets of
# twiddle factors.
# 
# What's so important about this is that 2^6 is a size small enough to fit
# into shared memory of a GPU thread block. This means that when we want to
# apply our bit reversal, we can just use shared memory as our auxiliary space.
# with the added benefit of removing the global memory access of the FFT.
# 
# Now comes the question of how to determine R, the optimal radix.
# The solution I use is to make R the # of numbers that are handled by each block.
# If one block does an entire FFT, this removes the need of inter-block syncing.
# 
# So how many numbers are handled by each block? This is decided by maximizing how many
# total threads we can run per SM. The bottleneck here is shared memory size.

# 3-dimensional doesn't work, 1-dimensional doesn't work
function ntt(vec::Vector{T}, npru::T, p::T) where T<:Integer
    n = length(vec)
    @assert ispow2(n)
    log2n = intlog2(n)

    original = vec
    aux = zeros(T, n)

    numsPerThread = 4
    threadsPerBlock = 2
    numsPerBlock = numsPerThread * threadsPerBlock

    numBlocks = length(vec) ÷ numsPerBlock

    npbpruPowerTable = generate_twiddle_factors(powermod(npru, n ÷ numsPerBlock, p), p, numsPerBlock)

    shared = zeros(T, numsPerBlock)
    numIterations = Int(ceil(log(numsPerBlock, n)))
    
    lastFFTLen = n ÷ numsPerBlock ^ (numIterations - 1)

    twiddleStride = 1
    for _ in 1:numIterations - 1
    # for itr in 1:numIterations
        fftLen = numsPerBlock

        log2fftLen = intlog2(fftLen)

        threadStride = numBlocks
        for blockIdx in 1:numBlocks
            for threadIdx in 1:threadsPerBlock
                # Load numbers into shared memory
                fftIdx = threadIdx - 1
                globalIdx = blockIdx + (threadIdx - 1) * threadStride
                added = threadStride * threadsPerBlock
                for _ in 1:numsPerThread
                    val = vec[globalIdx]
                    # println("blockIdx: $blockIdx, globalIdx: $globalIdx")
                    bitreversed = bit_reverse(fftIdx, log2fftLen)
                    shared[bitreversed + 1] = val
                    fftIdx += threadsPerBlock
                    globalIdx += added
                end
            end
            # display("before FFT: blockIdx: $blockIdx, shared: $shared")
            for i in 1:log2fftLen
                m = 1 << i
                m2 = m >> 1
                bits = log2fftLen - i
                mask = (1 << bits) - 1

                for threadIdx in 1:threadsPerBlock
                    threadStart = (threadIdx - 1) * (numsPerThread >> 1)

                    for j in 0:(numsPerThread >> 1) - 1
                        gh_j = j + threadStart
                        idx1 = m * (gh_j & mask) + (gh_j >> bits)
                        idx2 = idx1 + m2
                        # theta = twiddleFactors[(n >> i) * (gh_j >> bits) + 1]
                        theta = npbpruPowerTable[((fftLen >> i) * (gh_j >> bits)) + 1]
                        t = mul_mod(theta, shared[idx2 + 1], p)
                        u = shared[idx1 + 1]

                        shared[idx1 + 1] = add_mod(u, t, p)
                        shared[idx2 + 1] = sub_mod(u, t, p)
                    end
                end
            end

            # display("after FFT: blockIdx: $blockIdx, shared: $shared")
            
            blockTwiddleStride = ((blockIdx - 1) ÷ twiddleStride) * twiddleStride
            for threadIdx in 1:threadsPerBlock
                # Load numbers into shared memory
                fftIdx = threadIdx - 1
                globalIdx = fftIdx + (blockIdx - 1) * numsPerBlock
                for _ in 1:numsPerThread
                    # println("blockIdx: $blockIdx, globalIdx: $globalIdx, resultIdx: $resultIdx")
                    # shared[fftIdx + 1] = mul_mod(shared[fftIdx + 1], twiddleFactors[(blockIdx - 1) * fftIdx + 1], p)
                    shared[fftIdx + 1] = mul_mod(shared[fftIdx + 1], powermod(npru, fftIdx * blockTwiddleStride, p), p)
                    aux[globalIdx + 1] = shared[fftIdx + 1]
                    globalIdx += threadsPerBlock
                    fftIdx += threadsPerBlock
                end
            end
            # display("after twiddle: blockIdx: $blockIdx, shared: $shared")
        end

        display("after iteration: $aux")
        println()
        # println()
        # throw("skibidi")
        temp = aux
        aux = vec
        vec = temp

        twiddleStride *= numsPerBlock
    end

    for _ in 1:1
        fftLen = lastFFTLen
        log2fftLen = intlog2(fftLen)

        threadStride = numBlocks
        for blockIdx in 1:numBlocks
            fill!(shared, 0)
            for threadIdx in 1:threadsPerBlock
                # Load numbers into shared memory
                virtualThreadIdx = threadIdx - 1
                globalIdx = blockIdx + (threadIdx - 1) * threadStride
                added = threadStride * threadsPerBlock
                for _ in 1:numsPerThread
                    # transposedIdx = get_transposed_index(virtualThreadIdx, lastFFTLen, numsPerBlock ÷ lastFFTLen)
                    transposedIdx = get_transposed_index(virtualThreadIdx, numsPerBlock ÷ lastFFTLen, lastFFTLen)
                    fftNum = transposedIdx ÷ fftLen
                    val = vec[globalIdx]

                    fftIdx = transposedIdx % fftLen
                    bitreversed = bit_reverse(fftIdx, log2fftLen)
                    # bitreversed = fftIdx
                    # println("virtualThreadIdx: $virtualThreadIdx, original fftIdx: $(virtualThreadIdx % fftLen), fftNum: $fftNum, transposed fftIdx: $(fftIdx), bitreversed: $bitreversed, val: $val, shared idx: $(fftNum + bitreversed)")
                    shared[fftNum * fftLen + bitreversed + 1] = val
                    virtualThreadIdx += threadsPerBlock
                    globalIdx += added
                end
            end
            display("before FFT: blockIdx: $blockIdx, shared: $shared")
            for i in 1:log2fftLen
                m = 1 << i
                m2 = m >> 1
                bits = log2fftLen - i
                mask = (1 << bits) - 1

                for threadIdx in 1:threadsPerBlock
                    offset = 0
                    for j in 0:(numsPerThread >> 1) - 1
                        virtualThreadIdx = threadIdx + offset - 1
                        gh_j = virtualThreadIdx % (fftLen >> 1)
                        fftNum = virtualThreadIdx ÷ (fftLen >> 1)
                        idx1 = m * (gh_j & mask) + (gh_j >> bits) + ((fftLen) * fftNum)
                        idx2 = idx1 + m2
                        # println("threadIdx: $threadIdx, idx1: $idx1, idx2: $idx2, gh_j: $gh_j")
                        theta = npbpruPowerTable[((fftLen >> i) * (gh_j >> bits)) + 1]
                        t = mul_mod(theta, shared[idx2 + 1], p)
                        u = shared[idx1 + 1]

                        shared[idx1 + 1] = add_mod(u, t, p)
                        shared[idx2 + 1] = sub_mod(u, t, p)

                        offset += threadsPerBlock
                    end
                end
            end

            display("after FFT: blockIdx: $blockIdx, shared: $shared")
            println()
            # for threadIdx in 1:threadsPerBlock
            #     # Load numbers into shared memory
            #     fftIdx = threadIdx - 1
            #     # outputIdx = blockIdx + fftIdx * numBlocks
            #     outputIdx = (blockIdx - 1) * numsPerBlock + fftIdx

            #     for _ in 1:numsPerThread
            #         # println("fftIdx: $fftIdx outputted to $outputIdx")
            #         aux[digit_reverse(outputIdx, fftLen, Int(log(fftLen, n))) + 1] = shared[fftIdx + 1]
            #         outputIdx += threadsPerBlock
            #         fftIdx += threadsPerBlock
            #     end
            # end
        end

        # display("after iteration: $aux")
        # println()
    end

    if pointer(aux) != pointer(original)
        original .= aux
    end

    return nothing
end