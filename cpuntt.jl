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
    return nothing
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

function ntt(vec::Vector{T}, npru::T, p::T) where T<:Integer
    n = length(vec)
    @assert ispow2(n)
    original = vec
    aux = zeros(T, n)

    numsPerThread = 2
    threadsPerBlock = 4
    numsPerBlock = numsPerThread * threadsPerBlock

    numBlocks = length(vec) ÷ numsPerBlock

    twiddleFactors = generate_twiddle_factors(npru, p, n)
    
    smallTwiddleFactors = generate_twiddle_factors(powermod(npru, n ÷ numsPerBlock, p), p, numsPerBlock)

    shared = zeros(T, numsPerBlock)
    numIterations = Int(ceil(log(numsPerBlock, n)))
    threadStride = n
    # for _ in 1:numIterations - 1
    for _ in 1:numIterations - 1
        threadStride = numBlocks
        blockStride = 1
        fftLen = numsPerBlock
        log2fftLen = intlog2(fftLen)

        for blockIdx in 1:numBlocks
            blockStart = (blockIdx - 1) * blockStride
            for threadIdx in 1:threadsPerBlock
                # Load numbers into shared memory
                offset = 0
                for _ in 1:numsPerThread
                    fftIdx = (threadIdx - 1 + offset)
                    globalIdx = blockStart + fftIdx * threadStride
                    val = vec[globalIdx + 1]    
                    bitreversed = bit_reverse(fftIdx, log2fftLen)
                    shared[bitreversed + 1] = val
                    offset += threadsPerBlock
                end
            end

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
                        theta = twiddleFactors[(n >> i) * (gh_j >> bits) + 1]
                        t = mul_mod(theta, shared[idx2 + 1], p)
                        u = shared[idx1 + 1]

                        shared[idx1 + 1] = add_mod(u, t, p)
                        shared[idx2 + 1] = sub_mod(u, t, p)
                    end
                end
            end

            # display("blockIdx: $blockIdx, shared: $shared")
            
            # Need to transpose
            outputBlockStart = (blockIdx - 1) * numsPerBlock
            outputThreadStride = 1
            for threadIdx in 1:threadsPerBlock
                offset = 0
                for _ in 1:numsPerThread
                    fftIdx = (threadIdx - 1 + offset)
                    globalIdx = outputBlockStart + fftIdx * outputThreadStride
                    # Multiply by twiddle factor
                    shared[threadIdx + offset] = mul_mod(shared[threadIdx + offset], twiddleFactors[(blockIdx - 1) * fftIdx + 1], p)
                    aux[globalIdx + 1] = shared[threadIdx + offset]
                    offset += threadsPerBlock
                end
            end
        end
        temp = aux
        aux = vec
        vec = temp
    end

    for _ in 1:1
        threadStride = 1
        fftLen = div(n, numsPerBlock ^ (numIterations - 1))
        fftsPerBlock = numsPerBlock ÷ fftLen
        log2fftLen = intlog2(fftLen)

        for blockIdx in 1:numBlocks
            blockStart = (blockIdx - 1) * fftsPerBlock
            for threadIdx in 1:threadsPerBlock
                # Load numbers into shared memory
                offset = 0
                for _ in 1:numsPerThread
                    virtualIdx = threadIdx - 1 + offset
                    fftNum = virtualIdx % fftsPerBlock
                    fftIdx = virtualIdx ÷ fftLen
                    globalIdx = blockStart + fftNum + fftIdx * threadStride
                    val = vec[globalIdx + 1]
                    bitreversed = bit_reverse(fftIdx, log2fftLen)
                    shared[fftNum * fftLen + bitreversed + 1] = val
                    offset += threadsPerBlock
                end
            end
            # println("blockIdx: $blockIdx, shared: $(reshape(shared, 2, 2))")

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
                        theta = twiddleFactors[(n >> i) * (gh_j >> bits) + 1]
                        t = mul_mod(theta, shared[idx2 + 1], p)
                        u = shared[idx1 + 1]

                        shared[idx1 + 1] = add_mod(u, t, p)
                        shared[idx2 + 1] = sub_mod(u, t, p)

                        offset += threadsPerBlock
                    end
                end
                # println("blockIdx: $blockIdx, shared: $(reshape(shared, 2, 2))")
                # println()
            end

            blockStart = (blockIdx - 1) * fftsPerBlock
            for threadIdx in 1:threadsPerBlock
                # Load numbers into shared memory
                offset = 0
                for _ in 1:numsPerThread
                    fftIdx = (threadIdx - 1 + offset) % fftLen
                    fftNum = div(threadIdx - 1 + offset, fftLen)
                    globalIdx = blockStart + fftNum + fftIdx * threadStride
                    aux[globalIdx + 1] = shared[fftNum * fftLen + fftIdx + 1]
                    offset += threadsPerBlock
                end
            end
        end
    end

    if pointer(aux) != pointer(original)
        original .= aux
    end

    return nothing
end