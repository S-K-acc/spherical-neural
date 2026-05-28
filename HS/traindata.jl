function generate_inout(ρ_profiles, c1_profiles, eps_vals; window_width=2.0, dx=0.01)
    
    xs = collect(0.005:0.01:10) #maximal grid
    scale_dict = construct_scale_dict(xs;dx = 0.01)
    rabs = collect(0.005:0.01:12)
    dR = diff(R_HS.(rabs))
    push!(dR,0);
       
    window_bins = 2 * round(Int, window_width / dx) + 1
    ρ_windows_all = Vector{Vector{Float32}}()
    c1_values_all = Vector{Float32}()
    r_values_all = Vector{Float32}()
    for (ρ, c1, eps) in zip(ρ_profiles, c1_profiles, eps_vals)
        ρ_windows = generate_windows(ρ; window_bins)
        s = 1
        for i in collect(101:s:length(c1)-101)
            if !isfinite(c1[i])
                continue
            end
            r = eps[i]
            j = findall(x->x≈abs(r), rabs)
            rint = R_HS(eps[i])
            correcture = scale_dict[r]
            if rand() < dR[j[1]]/dR[1] # ≈ equal distribution of training data
                push!(ρ_windows_all, correcture .* ρ_windows[:,i])
                push!(c1_values_all, c1[i])
                push!(r_values_all, rint)
            end
        end
    end
    reduce(hcat, ρ_windows_all), permutedims(c1_values_all), permutedims(r_values_all)
end

function generate_inout_plan(ρ_profiles, c1_profiles, r_vals; window_width = 2.0, dx=0.01)
    window_bins = 2 * round(Int, window_width / dx) + 1
    ρ_windows_all = Vector{Vector{Float32}}()
    c1_values_all = Vector{Float32}()
    r_values_all = Vector{Float32}()
    for (ρ, c1, r) in zip(ρ_profiles, c1_profiles, r_vals)
        ρ_windows = generate_windows(ρ; window_bins)
        s = 10
        for i in collect(1:s:length(c1)) 
            if !isfinite(c1[i])
                continue
            end
            push!(ρ_windows_all, ρ_windows[:,i])
            push!(c1_values_all, c1[i])
            push!(r_values_all, r)
        end
    end
    reduce(hcat, ρ_windows_all), permutedims(c1_values_all), permutedims(r_values_all)
end

function read_sim_data_const(EOS)
    ρ_profiles = Vector{Vector{Float64}}()
    c1_profiles = Vector{Vector{Float64}}()
    r_profiles = Vector{Vector{Float64}}()
    rhon = EOS[:,1]
    c1n = EOS[:,2]
    for i =1:length(rhon)
        n = 450
        r = collect(0.005:0.01:n*0.01)
        ρ = rhon[i]*ones(2*n)
        c1 = c1n[i]*ones(2*n)
        push!(ρ_profiles, ρ)
        push!(c1_profiles, c1)
        push!(r_profiles, [-reverse(r);r])
    end
    ρ_profiles, c1_profiles, r_profiles
end 



function generate_windows(ρ; window_bins)
    ρ_windows = Zygote.Buffer(zeros(Float32, window_bins, length(ρ))) 
    pad = window_bins ÷ 2 - 1 
    ρpad = vcat(ρ[end-pad:end], ρ, ρ[1:1+pad]) 
    for i in 1:length(ρ)
        ρ_windows[:,i] = ρpad[i:i+window_bins-1] 
    end
    copy(ρ_windows)  
end


function construct_scale_dict(rvals;dx = 0.01,window_width=2.0) #function f(r,r')
    xs =  [-reverse(rvals);  rvals]
    Rw = collect(-window_width:0.01:window_width)
    Dict(rval =>  h.(Rw,0.01/rval) .* collect(round(rval-(window_width+10e-5),digits=3):0.01:round(rval+(window_width+10e-5),digits=3))./rval for rval in xs)
end
