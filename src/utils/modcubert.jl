# Just brute forces it for now, sizes aren't too big
# to warrant specialized algorithm
function modcubert(a::T, p::T) where T<:Integer
    for n in T(2):p-1
        if powermod(n, 3, p) == a
            return n
        end
    end
    throw("cube root of $a does not exist mod $p")
end