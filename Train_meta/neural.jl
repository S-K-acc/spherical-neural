#machine learning





function generate_windows(ρ; window_bins)
    ρ_windows = Zygote.Buffer(zeros(Float32, window_bins, length(ρ)))
    pad = window_bins ÷ 2 - 1 
    ρpad = vcat(ρ[end-pad:end], ρ, ρ[1:1+pad]) 
    for i in 1:length(ρ)
        ρ_windows[:,i] = ρpad[i:i+window_bins-1] 
    end
    copy(ρ_windows)  
end

function generate_phi(ϕ,ρ)
    ϕ_func = Zygote.Buffer(zeros(Float32, length(ϕ), length(ρ)))
    for i in 1:length(ρ)
        ϕ_func[:,i] = ϕ
    end
    copy(ϕ_func)
end




function R_meta(ϵ)
    if ϵ > 0
        return ((tanh(0.5*ϵ-1)+1)/2-(tanh(-1)+1)/2)*1/0.8807970779778824
    else
        return - ((tanh(-0.5*ϵ-1)+1)/2-(tanh(-1)+1)/2)*1/0.8807970779778824 
    end
end

function h(x,p)
    p = abs(p)
    exp(-p*x^2)
end


















