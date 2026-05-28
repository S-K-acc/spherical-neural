function Inversion(ρgoal, T, ρb, dx)
    α = 0.01 #mixing parameter
    dx = 0.01
    rs = Float32.(0.005:0.01:length(ρgoal)*dx)
    rdict = construct_scale_dict(rs)
    us = zero(ρgoal)
    ur = zero(ρgoal)

    ρgoal[findall(x -> x == 0, ρgoal)] .= 10e-6 #because of log
    if dx != 0.01
        @warn("function written for dx=0.01")
    end
    if !(50 <= argmax(ρgoal) <= 140)
        @warn("pair potential might not be compartible with finite cutoff, rescale system!")
    end
    
    for i=1:20
        c1b = get_c1_neural_rad(model,scale_windows,rdict)(ρb .* one.([reverse(ρgoal);ρgoal]),us[1:150],rs) #bulk c1
        µ = log.(ρb) - c1b[end] #chemical potential from equation of state
        ur = get_c1_neural_rad(model,scale_windows,rdict)([reverse(ρgoal);ρgoal],us[1:150],rs) - log.(ρgoal) .+ µ
        us = (1-α) .* ur + α .* us #Picard-mixing
        us .-= us[end] #pair potential vanishes for r > r_cutoff
    end
    rs = collect(0.005:0.01:1.5-0.005)
    return rs, us[1:150].*T
end





function solve_OZ4ρ2(barc2,ρ;dx = 0.01)
    @warn("function written for ,simple' situations")
    r = (0.005:0.01:length(ρ)*dx-0.005)
    f0 = findfirst(x -> x != 0, ρ)
    l0  = findlast(x -> x != 0, ρ)
    c2 = zero.(barc2[1:l0,1:l0])
    c2[f0:l0,f0:l0] = barc2[f0:l0,f0:l0]
    Vol = 4*π*Diagonal(r[1:l0].^2)
    lhs_h2 = I - ρ[1:l0].*c2 * Vol .*dx
    h_sol = lhs_h2 \ c2
    ρ2sol = (h_sol .+ 1) .* (ρ[1:l0]*transpose(ρ[1:l0]))
    r[1:l0], ρ2sol
end
