function long_tests(TOLERANCE = 1e-3; seed = 1)

    rg = MersenneTwister(seed)

    bnd = 150 # the larger the better, but more expensive

    dx = 1e-5 # the smaller the better, but more expensive

    function randomcoefficients()

        rand(rg)*2, randn(rg)*3, rand(rg)*2.0+1e-2, randn(rg)*3
            # a          μ             σ                b
    end

    function numericalintegration(f, dx, bnd)
        local s = 0.0
        @inbounds for x in -bnd:dx:bnd
            s += f(x)
        end
        s*dx
    end

    a, μ, σ, b = randomcoefficients()

    @printf("Randomly picked coefficients a, μ, σ, b are: %2.5f, %2.5f, %2.5f, %2.5f\n", a, μ, σ, b)

    #--------------------------------------------------------------------------------#
    # Expectations of exponential function wrt Normal, i.e. ∫ exp(a*x+b) N(x|μ,σ) dx #
    #--------------------------------------------------------------------------------#

    let

        p = Normal(μ, σ)
        
        quadr = quadgk(x -> pdf(p,x) * exp(a*x+b), -bnd, bnd)[1]

        hquadr = hquadrature(x -> pdf(p,x) * exp(a*x+b), -bnd, bnd)[1]

        analytical = E(;a = a, μ = μ, σ = σ, b = b)

        numerical = numericalintegration(x -> pdf(p,x) * exp(a*x+b), dx, bnd)

        @printf("Testing ∫ exp(a*x+b) N(x|μ,σ) dx implemented via E(a = a, μ = μ, σ = σ, b = b)\n")

        
        discrepancy = abs(quadr - analytical)

        @printf("numerical  = %f\n", numerical)
        @printf("quadgk     = %f\n", quadr)
        @printf("hquadr     = %f\n", hquadr)
        @printf("Analytical = %f\n", analytical)
        @printf("Discrepancy is %f\n", discrepancy)
        
        
        @assert discrepancy < TOLERANCE
        
        TOLERANCE > discrepancy ? @printf("Test passed ✓\n") : @printf("Test failed ❗\n")

    end


    #-------------------------------------------------------------------------------------------#
    # Expectations of squared exponential function wrt Normal, i.e. ∫ [exp(a*x+b)]² N(x|μ,σ) dx #
    #-------------------------------------------------------------------------------------------#

    let

        p = Normal(μ, σ)
        
        quadr = quadgk(x -> pdf(p,x) * exp(a*x+b)^2,-bnd,bnd)[1]

        hquadr = hquadrature(x -> pdf(p,x) * exp(a*x+b)^2,-bnd,bnd)[1]

        numerical = numericalintegration(x -> pdf(p,x) * exp(a*x+b)^2, dx, bnd)

        analytical = B(a = a, μ = μ,σ = σ,b = b)

        @printf("\nTesting  ∫ [exp(a*x+b)]² N(x|μ,σ) dx implemented via B(a = a, μ = μ, σ = σ, b = b)\n")
       
        discrepancy =  abs(quadr - analytical)

        @printf("numerical  = %f\n", numerical)
        @printf("quadgk     = %f\n", quadr)
        @printf("hquadr     = %f\n", hquadr)
        @printf("Analytical = %f\n", analytical)
        @printf("Discrepancy is %f\n", discrepancy)

        @assert discrepancy < TOLERANCE
        
        TOLERANCE > discrepancy ? @printf("Test passed ✓\n") : @printf("Test failed ❗\n")

    end
    

    #-----------------------------------------------------------------------------------#
    # Variance of exponential function wrt Normal, i.e. ∫ [exp(a*x+b) - m]² N(x|μ,σ) dx #
    #-----------------------------------------------------------------------------------#

    let


        p = Normal(μ, σ); m = E(a=a, μ=μ, σ=σ, b=b)

        quadr  = quadgk(x -> pdf(p,x) * (exp(a*x+b)-m)^2, -bnd, bnd)[1]

        hquadr = hquadrature(x -> pdf(p,x) * (exp(a*x+b)-m)^2, -bnd, bnd)[1]
        
        numerical = numericalintegration(x -> pdf(p,x) * (exp(a*x+b)-m)^2, dx, bnd)

        analytical = V(a = a, μ = μ, σ = σ, b = b)
        
        @printf("\nTesting  ∫ [exp(a*x+b) - m]² N(x|μ,σ) dx  implemented via V(a = a, μ = μ, σ = σ, b = b)\n")

        discrepancy =  abs(quadr - analytical)

        @printf("numerical  = %f\n", numerical)
        @printf("quadgk     = %f\n", quadr)
        @printf("hquadr     = %f\n", hquadr)
        @printf("Analytical = %f\n", analytical)
        @printf("Discrepancy is %f\n", discrepancy)

    end
    
    


    #-------------------------------------------------#
    # ∫ N(x|μ, σ) log N(y | c ⋅ exp(a⋅x + b), β⁻¹) dx #
    #-------------------------------------------------#

    let

        y, c, β = randn(rg)*2, randn(rg)*2, exp(rand(rg)*3)

        p = Normal(μ, σ)

        quadr = quadgk(x -> pdf(p, x) * logpdf(Normal(c*exp(a*x+b), sqrt(1/β)), y),-bnd, bnd)[1]
        
        hquadr = hquadrature(x -> pdf(p, x) * logpdf(Normal(c*exp(a*x+b), sqrt(1/β)), y),-bnd, bnd)[1]
        
        numerical = numericalintegration(x -> pdf(p, x) * logpdf(Normal(c*exp(a*x+b), sqrt(1/β)), y), dx, bnd)

        analytical = Elognormal(y=y, a=a, μ=μ, b=b, c=c, σ=σ, β=β)


        @printf("\nTesting  ∫ N(x|μ, σ) log N(y | c ⋅ exp(a⋅x + b), β⁻¹) dx  implemented via Elognormal(y=y, a=a, μ=μ, b=b, c=c, σ=σ, β=β)\n")

        discrepancy =  abs(quadr - analytical)

        @printf("numerical  = %f\n", numerical)
        @printf("quadgk     = %f\n", quadr)
        @printf("hquadr     = %f\n", hquadr)
        @printf("Analytical = %f\n", analytical)
        @printf("Discrepancy is %f\n", discrepancy)
    end
    
end