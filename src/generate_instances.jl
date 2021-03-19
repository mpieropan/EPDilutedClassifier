function generate_weights(N,ρ)
    K=round(Int64,ρ*N)
    w=zeros(N)
    indices=shuffle(1:N)
    w[indices[1:K]]=randn(K)
    return w
end

function generate_weights_recnet(N,ρ)
    W=zeros(N,N-1)
    for i=1:N
        W[i,:]=generate_weights(N-1,ρ)
    end
    return W
end

function generate_gaussian_iid_patterns(N,α)
    M = round(Int64,α*N);
    G = randn(M,N)
    return G
end

function generate_gaussian_cor_patterns(N,α)
    M = round(Int64,α*N);
    Y = randn(1,N); SIG = Y'*Y + diagm(0 => abs.(randn(N)))
    distr = MvNormal(SIG)
    G = (rand(distr))'
    for m = 2:M
        G = vcat(G,(rand(distr))')
    end
    return G
end

function recurrent_net_asynchronous_patterns(W,ξ_old,randperc)
    N=length(ξ_old)
    ξ_new=copy(ξ_old)
    indices=union(1:(randperc-1),(randperc+1):N)
    ξ_new[randperc]=sign(ξ_old[indices]'W[randperc,:])
    return ξ_new
end

function generate_patterns_recnet_dynamics(W,αmax)
    # asynchronous update
    Nperceptrons,N=size(W)
    Mmax=round(Int64,αmax*N)
    perc_seq=1 .+ randperm(Mmax).%Nperceptrons # list of perceptrons chosen at random
    Tmax=Mmax+1
    ξ=Array{Float64,2}(undef,Tmax,Nperceptrons)
    ξ[1,:]=sign.(randn(Nperceptrons))
    t=0
    for m=1:Mmax
        t=t+1
        ξ_old=copy(ξ[t,:])
        my_rand_perc=perc_seq[m]
        ξ_new=recurrent_net_asynchronous_patterns(W,ξ_old,my_rand_perc)
        ξ[t+1,:]=ξ_new
    end
    return ξ
end
