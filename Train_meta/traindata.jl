function generate_inout_plan(ρ_profiles, c1_profiles, ϕ_profiles, r_vals; window_width=2.0, dx=0.01)
    window_bins = 2 * round(Int, window_width / dx) + 1
    ρ_windows_all = Vector{Vector{Float32}}()
    c1_values_all = Vector{Float32}()
    ϕ_functions_all = Vector{Vector{Float32}}()
    r_values_all = Vector{Float32}()
    for (ρ, c1, ϕ, r) in zip(ρ_profiles, c1_profiles, ϕ_profiles, r_vals)
        ρ_windows = generate_windows(ρ; window_bins)
        ϕ_func = generate_phi(ϕ,ρ)
        s = 3
        for i in collect(1:s:length(c1)) 
            if !isfinite(c1[i])
                continue
            end
            push!(ρ_windows_all, ρ_windows[:,i])
            push!(c1_values_all, c1[i])
            push!(ϕ_functions_all, ϕ_func[:,i])
            push!(r_values_all, r)

        end
    end
    reduce(hcat, ρ_windows_all), permutedims(c1_values_all), reduce(hcat, ϕ_functions_all), permutedims(r_values_all)
end


function read_sim_data(file)
    ρ_profiles = Vector{Vector{Float64}}()
    c1_profiles = Vector{Vector{Float64}}()
    ϕ_profiles = Vector{Vector{Float64}}()
    r_profiles = Vector{Vector{Float64}}()
    data = load(file) 
    gs = data["gs"]
    ϕs = data["ϕs"]
    ρbs = data["ρbs"]
    μs = data["μs"]
    rs = data["rs"]
    phi = zero.(rs)
    for i =1:length(gs)
        u = ϕs[i]
        ρ = ρbs[i]*gs[i]
        µ = μs[i]
        phi[1:150] .= u
        Vext = 0.5*phi
        Vext[1:end-1] += Vext[2:end]
        ρ = [reverse(ρ);ρ]
        Vext = [reverse(phi);phi]
        uloc = µ .- Vext
        c1 = log.(ρ) .- uloc
        push!(ρ_profiles, ρ)
        push!(c1_profiles, c1)
        push!(ϕ_profiles, u)
        push!(r_profiles, [-reverse(rs);rs])
    end
    ρ_profiles, c1_profiles, ϕ_profiles, r_profiles
end




function read_sim_data_const(file)
    ρ_profiles = Vector{Vector{Float64}}()
    c1_profiles = Vector{Vector{Float64}}()
    ϕ_profiles = Vector{Vector{Float64}}()
    r_profiles = Vector{Vector{Float64}}()
    
    data = load(file) 
    gs = data["gs"]
    ϕs = data["ϕs"]
    ρbs = data["ρbs"]
    μs = data["μs"]
    rs = data["rs"]    
    
    for i =1:length(gs)
        u =  ϕs[i]
        ρ = ρbs[i]*ones(700)
        µ = μs[i]
        ρ = [reverse(ρ);ρ]
        c1 = log.(ρ) .- µ
        push!(ρ_profiles, ρ)
        push!(c1_profiles, c1)
        push!(ϕ_profiles, u)
        push!(r_profiles, [-reverse(rs);rs])
    end
    ρ_profiles, c1_profiles, ϕ_profiles, r_profiles
end




function generate_inout_tp(ρ_profiles, c1_profiles, ϕ_profiles, r_vals; window_width=2.0, dx=0.01)
    window_bins = 2 * round(Int, window_width / dx) + 1
    ρ_windows_all = Vector{Vector{Float32}}()
    c1_values_all = Vector{Float32}()
    ϕ_functions_all = Vector{Vector{Float32}}()
    r_values_all = Vector{Float32}()
    Rw = collect(-2:0.01:2)
    for (ρ, c1, ϕ, eps) in zip(ρ_profiles, c1_profiles, ϕ_profiles, r_vals)
        ρ_windows = generate_windows(ρ; window_bins)
        ϕ_func = generate_phi(ϕ,ρ)
        s = 1
        for i in collect(401:s:length(c1)-401) 
            if !isfinite(c1[i])
                continue
            end
            r = eps[i]
            rp = collect(round(r-2,digits=3):0.01:round(r+2,digits=3))
            rint = R_meta(eps[i])
            correcture = rp./r
            Rw = collect(-2:0.01:2)
            tra = h.(Rw,0.01/r)        
            
            if abs(rint) > 0.5
            rn = rand()
                if rn > 0.3*abs(rint)
                    push!(ρ_windows_all, correcture .* ρ_windows[:,i] .* tra)
                    push!(c1_values_all, c1[i])
                    push!(ϕ_functions_all, ϕ_func[:,i])
                    push!(r_values_all, rint)
                end 
            else 
                push!(ρ_windows_all, correcture .* ρ_windows[:,i] .* tra )
                push!(c1_values_all, c1[i])
                push!(ϕ_functions_all, ϕ_func[:,i])
                push!(r_values_all, rint)
                  
            end       
        end
    end
    reduce(hcat, ρ_windows_all), permutedims(c1_values_all), reduce(hcat, ϕ_functions_all), permutedims(r_values_all)
end