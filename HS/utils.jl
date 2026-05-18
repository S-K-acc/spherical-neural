function construct_c2(fullc2;dx=0.01) 
    #c2(r,r') - c2(r,-r')
    allc2 = []
    allxc2 = []
    xs = collect(dx/2:dx:size(fullc2)[1]*dx)
    for k=0:size(fullc2)[1]-1
        
        xc2 = collect(xs[1+k]*100-200:1:xs[1+k]*100+200)*dx
        c2r = fullc2[1+k,:]
        c2c = []
        x2c = []
        if k <= 200
            for i = 0:199-k 
                R = k*dx
                Rp = i*dx
                c2 = (c2r[201-k+i] + c2r[200-k-i])
                push!(c2c,c2)
                push!(x2c,xc2[201-k+i])
            end
            for i = 401-2*k:401
                R = k*dx
                Rp = i*dx
                push!(c2c,c2r[i])
                push!(x2c,xc2[i])
                
            end
        else
            c2c = c2r
            x2c = xc2
        end
        push!(allc2,c2c)
        push!(allxc2,x2c)
    end
    return allxc2,allc2
end

function construct_c2mat(allrs, allc2;dx = 0.01)
    ydims = size(allc2)[1] + 200
    xdims = size(allc2)[1]
    c2mat = zeros((ydims,xdims))
    for i=1:size(allc2)[1]
        if i <= 200
            lb = 1
            ub = length(allrs[i])
            c2mat[lb:ub,i] = allc2[i] 
        else
            lb = i-200
            ub = lb + 400
            c2mat[lb:ub,i] = allc2[i] 
        end
    end
    rr = collect(dx/2:dx:size(allc2)[1]*dx)
    rp = (collect(0.5:1:ydims-0.5))/100
    return rr,rp,c2mat
end

function solve_OZ4ρ2(ringc2,ρ;dx = 0.01)
    @warn("function written for ,simple' situations")
    r = (0.005:0.01:length(ρ)*dx-0.005)
    f0 = findfirst(x -> x != 0, ρ)
    l0  = findlast(x -> x != 0, ρ)
    c2 = zero.(ringc2[1:l0,1:l0])
    c2[f0:l0,f0:l0] = ringc2[f0:l0,f0:l0]
    Vol = 4*π*Diagonal(r[1:l0].^2)
    lhs_h2 = I - ρ[1:l0].*c2 * Vol .*dx
    h_sol = lhs_h2 \ c2
    ρ2sol = (h_sol .+ 1) .* (ρ[1:l0]*transpose(ρ[1:l0]))
    r[1:l0], ρ2sol
end