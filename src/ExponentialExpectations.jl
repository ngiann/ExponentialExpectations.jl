module ExponentialExpectations

    using Random, QuadGK, Distributions, Printf, HCubature

    include("expectations_exponential.jl")

    export E, B, V, Elognormal

    include("long_tests.jl")

    export long_tests

end
