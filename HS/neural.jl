

function generate_windows(ρ; window_bins)
    ρ_windows = Zygote.Buffer(zeros(Float32, window_bins, length(ρ))) # We use a Zygote Buffer here to keep autodifferentiability
    pad = window_bins ÷ 2 - 1
    ρpad = vcat(ρ[end-pad:end], ρ, ρ[1:1+pad]) 
    for i in 1:length(ρ)
        ρ_windows[:,i] = ρpad[i:i+window_bins-1] 
    end
    copy(ρ_windows)  # copy needed due to Zygote.Buffer
end

function h(x,p) 
    p = abs(p)
    exp(-p*x^2)
end

function R_HS(r) #function R(r)
    tanh(0.3*r)
end

function get_c1_neural_rad(model,scale_windows,scale_dict)
    window_bins = 401 #input bins for density
    model = model |> gpu
    function (ρ,xs)
        l = length(ρ)
        ρ_windows = generate_windows(ρ; window_bins)
        r = [-reverse(xs);  xs] #extendet to -r
        ρ_windows = scale_windows(ρ_windows,r,scale_dict) |> gpu #preprocessing density
        
        rint = R_HS.(r)|> gpu #preprocessing position
        input = vcat(ρ_windows, rint')|> gpu  
        c1 = model(input) |> cpu |> vec #evaluate model
        c11 = c1[Integer(l/2+1):end]
        c12 = reverse(c1[1:Integer(l/2)]) #mirrored output
        (c11+c12)/2
  
    end
end


function minimize(L::Number, μ::Number, T::Number, Vext::Function, c1neural::Function, model; scalewindows=nothing, α::Number=0.03, maxiter::Int=10000, dx::Number=0.01, floattype::Type=Float32, tol::Number=max(eps(floattype(1e3)), 1e-8))
    L, μ, T = floattype.((L, μ, T))
    xs = collect(floattype, dx/2:dx:L)
    Vext = Vext.(xs) #evaluate external potential
    infiniteVext = isinf.([reverse(Vext);Vext])  
    ρ, ρEL = zero([xs;xs]), zero([xs;xs]) #preallocate density
    fill!(ρ, 0.5) 
    βVext = Vext ./ T #thermal scaling
    βμ = μ / T
    rdict = construct_scale_dict(xs);  
    if scalewindows != nothing
        c1_func = c1neural(model,scalewindows,rdict)
    else
        c1_func = c1neural(model)
    end
    i = 0
    while true #Picard iteration
        c1 = c1_func(ρ,xs)
        ρEL .= exp.(βμ .- [reverse(βVext);βVext] .+ [reverse(c1);c1])  # Evaluate the RHS of the Euler-Lagrange equation
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
            return xs, ρ[Integer(L/dx+1):end]  
        end
    end
    xs, ρ[Integer(L/dx+1):end]
end


function scale_windows(ρ_windows,r,scale_dict)
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



function get_c1_neural_plan(model) #uses only length of pair potential, not the values
    window_bins = length(model.layers[1].weight[1,:]) -1 # Get the number of input bins from the shape of the first layer
    model = model |> gpu
    function (ρ; epsilon = 1)
        ρ_windows = generate_windows(ρ; window_bins) |> gpu  
        ϵ = transpose(zeros(length(ρ)) .+ epsilon)
        input = vcat(ρ_windows, ϵ) |> gpu  
        c11 = model(input) |> cpu |> vec
                
        ρ_windows = generate_windows(reverse(ρ); window_bins) |> gpu  
        ϵ = transpose(zeros(length(ρ)) .- epsilon)
        input = vcat(ρ_windows, ϵ) |> gpu  
        c12 = model(input) |> cpu |> vec
        (c11 + c12) ./2
    end
end

function minimize_plan(L::Number, μ::Number, T::Number, Vext::Function, get_c1::Function; α::Number=0.03, maxiter::Int=10000, dx::Number=0.01, floattype::Type=Float32, tol::Number=max(eps(floattype(1e3)), 1e-8))
    L, μ, T = floattype.((L, μ, T))
    xs = collect(floattype, dx/2:dx:L)  # Construct the numerical grid
    Vext = Vext.(xs)  # Evaluate the external potential on the grid
    infiniteVext = isinf.(Vext)  # Check where Vext is infinite to set ρ = 0 there
    ρ, ρEL = zero(xs), zero(xs)  # Preallocate the density profile and an intermediate buffer for iteration
    fill!(ρ, 0.5)  # Start with a bulk density of 0.5
    c1 = get_c1(xs)  # Obtain the c1 functional for the given numerical grid
    i = 0
    while true
        ρEL .= exp.((μ .- Vext) ./ T .+ c1(ρ))  # Evaluate the RHS of the Euler-Lagrange equation
        ρ .= (1 - α) .* ρ .+ α .* ρEL  # Do a Picard iteration step to update ρ
        ρ[infiniteVext] .= 0  # Set ρ to 0 where Vext = ∞
        clamp!(ρ, 0, Inf)  # Make sure that ρ does not become negative
        Δρmax = maximum(abs.(ρ - ρEL)[.!infiniteVext])  # Calculate the remaining discrepancy to check convergence
        i += 1
        if Δρmax < tol
            println("Converged (step: $(i), ‖Δρ‖ = $(Δρmax) < $(tol) = tolerance)")
            break  # The remaining discrepancy is below the tolerance: break out of the loop and return the result
        end
        if !isfinite(Δρmax) || i >= maxiter
            println("Did not converge (step: $(i) of $(maxiter), ‖Δρ‖: $(Δρmax), tolerance: $(tol))")
            return nothing  # The iteration did not converge, there is no valid result
        end
    end
    xs, ρ
end



function c1_single_rad(model,mp)
    model = model |> gpu 
    function (ρ,r,cor,trafo)
        ρ = ρ .* trafo .* cor
        input = vcat(ρ,R_HS(r))|> gpu 
        out = model(input) |> cpu |> vec 
        out
    end
end

function get_c2_from_ρ(ρ,xs,model;mp=h)
    window_bins = 401
    rho = [reverse(ρ);ρ]
    ρ_windows = generate_windows(rho; window_bins)
    c1srp = c1_single_rad(model,mp)
    l = length(ρ)
    c2 = []

    for i =1:l
        xr = xs[i]
        rho_el = ρ_windows[:,l+i]
        #+r
        rp = collect(round(xr-2.0001,digits=3):0.01:round(xr+2.0001,digits=3))
        correcture = rp./xr
        Rw = collect(-2:0.01:2)
        trafo = mp.(Rw,0.01/xr)
        craddiff =  Flux.jacobian(c1srp,rho_el,xr,correcture,trafo)[1]*100
        
        #-r
        
        xr = - xs[i]
        rp = collect(round(xr-2.0001,digits=3):0.01:round(xr+2.0001,digits=3))
        correcture = rp./xr
        craddiff2 =  Flux.jacobian(c1srp,reverse(rho_el),xr,correcture,trafo)[1]*100
 
        
        push!(c2,(craddiff+reverse(craddiff2))/2)
    end
    reduce(vcat,c2)
end


function get_Fexc_funcintegral_radial(c1_function, xs; num_a=100)
    dx = xs[2] - xs[1]
    da = 1 / num_a
    as = da/2:da:1
    function (ρ)
        aintegral = zero(ρ)
        for a in as
            aintegral .+= c1_function([reverse(a .* ρ); a .* ρ],xs)* da
        end
        -sum(ρ .* aintegral .* xs.^2) * dx * 4*π
    end
end




