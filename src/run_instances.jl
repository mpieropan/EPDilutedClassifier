function run_instance(w,ξ,η; epsconv=1.0e-4,callback=(x...)->nothing,ρ0=0.25,
    λ0=1.0,δρ=0.0,η0=1.0,δη=0.0,damp=0.99,maxiter=50000,EPInputState=nothing)
    println("ρ0=$ρ0, δρ=$δρ, η0=$η0, δη=$δη")
    ret,(Γ,Λ),runtime,MSEdB,Overlap=EP_Regress_ZeroTemp(ξ,w,
        η,λ=λ0,ρ0=ρ0,λ0=λ0,δρ=δρ,δλ=0.0,η0=η0,δη=δη,damp=damp,maxiter=maxiter,
        epsconv=epsconv,callback=callback,EPInputState=EPInputState)
    println("Teacher-student overlap: $Overlap")
    println("Teacher-student mean squared error (dB): $MSEdB")
    return ret,(Γ,Λ),runtime,MSEdB,Overlap
end

function run_instance_gaussian_iid(w,ρ,α,η; callback=(x...)->nothing,
        ρ0=ρ,λ0=1.0,δρ=0.0,η0=η,δη=0.0,damp=0.99,epsconv=1e-4,maxiter=50000)
    prior=Lambda_prior(η)
    N=length(w)
    ξ=generate_gaussian_iid_patterns(N,α)
    M=size(ξ)[1];
    initial_EP_state=initialize(M,N,η)
    ret,(Γ,Λ),runtime,MSEdB,Overlap = run_instance(w,ξ,η,epsconv=epsconv,
        callback=callback,ρ0=ρ,λ0=λ0,δρ=δρ,η0=η0,δη=δη,damp=damp,maxiter=maxiter,EPInputState=initial_EP_state)
    return ret,(Γ,Λ),runtime,ρ0,MSEdB,Overlap
end

function run_instance_gaussian_cor(w,ρ,α,η; callback=(x...)->nothing,
        ρ0=ρ,λ0=1.0,δρ=0.0,η0=η,δη=0.0,damp=0.99,epsconv=1e-4,maxiter=50000)
    prior=Lambda_prior(η)
    N=length(w)
    ξ=generate_gaussian_cor_patterns(N,α)
    M=size(ξ)[1];
    initial_EP_state=initialize(M,N,η)
    ret,(Γ,Λ),runtime,MSEdB,Overlap = run_instance(w,ξ,η,epsconv=epsconv,
        callback=callback,ρ0=ρ,λ0=λ0,δρ=δρ,η0=η0,δη=δη,damp=damp,maxiter=maxiter,EPInputState=initial_EP_state)
    return ret,(Γ,Λ),runtime,ρ0,MSEdB,Overlap
end
