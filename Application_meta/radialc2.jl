function c1_single_rad(model,u)
    model = model |> gpu 
    function (ρ,r,cor,trafo)
        ρ = ρ .* trafo .* cor
        input = vcat(ρ,Float32.(u),R_meta(r))|> gpu 
        out = model(input) |> cpu |> vec 
        out
    end
end

function get_c2_from_ρ(ρ,u,xs,model,mp) # automatic differentiation for density windows
    window_bins = 401
    rho = [reverse(ρ);ρ]
    ρ_windows = generate_windows(rho; window_bins)
    c1srp = c1_single_rad(model,u)
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



function construct_c2(fullc2;dx=0.01) # take into account changes at mirrored position
    #c2(r,r') + c2(r,-r')
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

function construct_c2mat(allrs, allc2;dx = 0.01) #construct c_2(r,r')
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