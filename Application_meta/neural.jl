

function generate_windows(ρ; window_bins)
    ρ_windows = Zygote.Buffer(zeros(Float32, window_bins, length(ρ))) # We use a Zygote Buffer here to keep autodifferentiability
    pad = window_bins ÷ 2 - 1 # a number
    ρpad = vcat(ρ[end-pad:end], ρ, ρ[1:1+pad]) 
    for i in 1:length(ρ)
        ρ_windows[:,i] = ρpad[i:i+window_bins-1] 
    end
    copy(ρ_windows)  # copy needed due to Zygote.Buffer
end

function generate_phi(ϕ,ρ)
    ϕ_func = Zygote.Buffer(zeros(Float32, length(ϕ), length(ρ)))
    for i in 1:length(ρ)
        ϕ_func[:,i] = ϕ
    end
    copy(ϕ_func)
end


# pair potential from discretized values
function ϕ(x,params) 
    l = length(params)
    Δ = 1.5/(l-1)
    if 0 <= x < 1.5
        bin = x ÷ Δ
        bin = Integer(bin)
        ϵ1 = params[bin+1]
        ϵ2 = params[bin+2]
        return (ϵ2-ϵ1)*(x-bin*Δ)/(Δ) + ϵ1
    else
        return 0
    end
end

# discrete values of pair potential
function get_params(f::Function)
    grid = collect(0:0.01:1.5-0.01)
    params = f.(grid)
    for i=1:length(params)
        if params[i] > 9 || isinf(params[i]) == true 
            params[i] = 9
        end
    end 
    for i=1:length(params)
        if isnan(params[i]) == true 
            params[i] = 9
        end
    end 
    return params
end



###############################################################################################################
# DFT
#planar 
function get_c1_neural_plan(model,ϕ) 
    window_bins = length(model.layers[1].weight[1,:])-size(ϕ)[1] -1
    model = model |> gpu
    function (ρ,ϕ, epsilon)
        ϕ_func = generate_phi(ϕ,ρ) |> gpu 
        ρ_windows = generate_windows(ρ; window_bins) |> gpu  
        ϵ = transpose(zeros(length(ρ)) .+ epsilon)
        input = vcat(ρ_windows,ϕ_func, ϵ)
        c11 = model(input) |> cpu |> vec
        
        input = vcat(reverse(ρ_windows,dims = 1),ϕ_func, ϵ)
        c13 = model(input) |> cpu |> vec
        
        ρ_windows = generate_windows(reverse(ρ); window_bins) |> gpu  
        ϵ = transpose(zeros(length(ρ)) .- epsilon)
        input = vcat(ρ_windows,  ϕ_func, ϵ)
        c12 = model(input) |> cpu |> vec
        
        input = vcat(reverse(ρ_windows,dims = 1),  ϕ_func, ϵ)
        c14 = model(input) |> cpu |> vec
                
        (c11 + c12 + c13 + c14) ./4
    end
end

function minimize_plan(L::Number, μ::Number, T::Number, ϕ::Vector{Float32}, Vext::Function, get_c1::Function; ϵ::Number=1, α::Number=0.03, maxiter::Int=10000, dx::Number=0.01, floattype::Type=Float32, tol::Number=max(eps(floattype(1e3)), 1e-8))
    L, μ, T, ϵ = floattype.((L, μ, T, ϵ))
    ϕ = vec(ϕ);
    ϕ = floattype.(ϕ)
    xs = collect(floattype, dx/2:dx:L)  
    Vext = Vext.(xs) 
   
    βVext = Vext ./ T
    βμ = μ / T
    βϕ = ϕ ./ T #WARN IF CUTOFF
    if maximum(ϕ) > 7 && T != 1
        @warn("cutoff of infinite potential changes with thermal scaling")
    end
    
    infiniteVext = isinf.(Vext)  
    ρ, ρEL = zero(xs), zero(xs) 
    fill!(ρ, 0.5) 
    c1 = get_c1(xs) 
    i = 0
    while true
        ρEL .= exp.((βμ .- βVext) .+ c1(ρ,βϕ,ϵ)) 
        ρ .= (1 - α) .* ρ .+ α .* ρEL 
        ρ[infiniteVext] .= 0 
        clamp!(ρ, 0, Inf)  
        Δρmax = maximum(abs.(ρ - ρEL)[.!infiniteVext]) 
        i += 1
        if Δρmax < tol
            println("Converged (step: $(i), ‖Δρ‖ = $(Δρmax) < $(tol) = tolerance)")
            break 
        end
        if !isfinite(Δρmax) || i >= maxiter
            println("Did not converge (step: $(i) of $(maxiter), ‖Δρ‖: $(Δρmax), tolerance: $(tol))")
            return nothing 
        end
    end
    xs, ρ
end

# radial

function R_meta(r) # positional input preprocessing
    if r > 0
        return ((tanh(0.5*r-1)+1)/2-(tanh(-1)+1)/2)*1/0.8807970779778824
    else
        return - ((tanh(-0.5*r-1)+1)/2-(tanh(-1)+1)/2)*1/0.8807970779778824 
    end
end


function h(x,p) 
    p = abs(p)
    exp(-p*x^2)
end

function scale_windows(ρ_windows,r,scale_dict) #return fρ
    for i =1:length(r)
        rval = round(r[i], digits=3)
        correcture = scale_dict[rval]
        ρ_windows[:,i] =  correcture .* ρ_windows[:,i] 
    end
    return ρ_windows
end

function construct_scale_dict(rvals;dx = 0.01,window_width=2.0,mp=h) #function f(r,r')
    xs =  [-reverse(rvals);  rvals]
    Rw = collect(-window_width:0.01:window_width)
    Dict(rval =>  mp.(Rw,0.01/rval) .* collect(round(rval-(window_width+10e-5),digits=3):0.01:round(rval+(window_width+10e-5),digits=3))./rval for rval in xs)
end

