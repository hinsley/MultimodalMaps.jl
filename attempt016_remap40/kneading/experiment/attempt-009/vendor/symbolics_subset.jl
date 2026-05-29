module SymbolicsSubset

export BranchSymbol, itinerary_to_kneading_sequence, normalized_LZ76_complexity, SSCS_to_itinerary

@enum BranchSymbol begin
    SymbolA
    SymbolB
    SymbolC
    SymbolD
    SymbolE
    SymbolF
end

# Exact function bodies copied from PlantChaos/tools/symbolics.jl,
# reduced to the subset used by continuous_critical_itineraries.jl.

function itinerary_to_kneading_sequence(itinerary::Vector{BranchSymbol})::Vector{Int}
    kneading_sequence = Int[]
    kneading_symbol_accumulator = 2

    for symbol in itinerary
        if symbol == SymbolA
            push!(kneading_sequence, 1)
        elseif symbol == SymbolB
            kneading_symbol_accumulator = 2
        elseif symbol == SymbolD
            kneading_symbol_accumulator += 2
        elseif symbol == SymbolE
            kneading_symbol_accumulator += 1
            push!(kneading_sequence, kneading_symbol_accumulator)
        elseif symbol == SymbolF
            push!(kneading_sequence, kneading_symbol_accumulator)
        end
    end

    return kneading_sequence
end

function normalized_LZ76_complexity(sequence::Vector{Int})::Float64
    if isempty(sequence)
        return 0.0
    end

    complexity = 0
    i = 1
    n = length(sequence)
    b = length(unique(sequence))

    if b == 1
        return 0.0
    end

    while i <= n
        max_match_length = 0

        for j in 1:(i-1)
            match_length = 0
            while (i + match_length <= n) && (sequence[j + match_length] == sequence[i + match_length])
                match_length += 1
                if j + match_length > i - 1
                    break
                end
            end
            if match_length > max_match_length
                max_match_length = match_length
            end
        end

        if max_match_length > 0
            complexity += 1
            i += max_match_length + 1
        else
            complexity += 1
            i += 1
        end
    end

    normalized_complexity = complexity * log2(n) / (n * log2(b))
    return normalized_complexity
end

function SSCS_to_itinerary(SSCS::Vector{Int})::Vector{BranchSymbol}
    itinerary = BranchSymbol[]
    for symbol in SSCS
        if symbol == 0
            push!(itinerary, SymbolA)
        else
            push!(itinerary, SymbolB)
            for _ in 2:abs(symbol)
                push!(itinerary, SymbolD)
            end
            push!(itinerary, SymbolC)
            if symbol > 0
                push!(itinerary, SymbolE)
            else
                push!(itinerary, SymbolF)
            end
        end
    end
    return itinerary
end

end
