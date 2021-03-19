function initialize(M,N,η)
    Ntot=M+N
    Nx=N
    state=EPState{Float64}(Ntot,Nx)
    if η<1 # noisy scenario
       	@extract state A y Σ v av va a μ b s
        b[1:Nx] .*= 1/Nx
        state=EPState{Float64}(A,y,Σ,v,av,va,a,μ,b,s)
    end
    return state
end

function normalize_weights(w,ret,N)
    teacher = w./norm(w);
    student = ret.av[1:N]./norm(ret.av[1:N]);
    return teacher, student
end

function signflip(o,η)
    M=length(o); println("M=$M");
    K_flip=floor(Int64,(1-η)*M); #number of flipped labels
    println("K_flip=$K_flip");
    #flipped = 1:K_flip; o[flipped]*=-1
    flipped = randperm(M)[1:K_flip]; o[flipped]*=-1
    #for i=1:M
    #    if rand() > η
    #        o[i] *= -1
    #    end
    #end
    return o
end

function Lambda_prior(η)
    @assert 0.0<=η<=1.0
    if η==1.0
        println("Noiseless case")
        prior="theta"
    else
        println("Noisy case")
        prior="theta-mixture"
    end
    return prior
end

function Reporter(w)
    function report(iter,state,Δav,Δva,epsconv,maxiter,H,P0)
        av=state.av
        teacher=w./norm(w)
        N=length(w)
        student=av[1:N]./norm(av[1:N])
        overlap=teacher'student
        MSE=Distances.msd(student,teacher)
        MSEdB=10.0*log10(MSE)
        if iter % 100 == 0
            println("$iter $Δav $MSEdB $overlap")
        end
        Δav < epsconv && return true
    end
end

function EP_Regress_ZeroTemp(ξ,w,η;λ=1.0,ρ0=0.25,λ0=1.0,δρ=0.0,δλ=0.0,
η0=0.5,δη=0.0,damp=0.9,maxiter=10000,epsconv=1.0e-4,offset=0.0,callback=(x...)->nothing,
maxvar=1e50,minvar=1e-50,EPInputState::Union{EPState{Float64},Nothing}=nothing,
prior::String="theta_mixture")
    K = length(findall(w.!=0.0))
    M,N = size(ξ)
    o = sign.(ξ*w)
    o_new = signflip(o,η)
    ξ.*=o_new
    Γ = SpikeSlabPrior(ρ0,λ0,δρ,δλ)
    (prior=="theta") && (Λ=ThetaPrior())
    (prior=="theta_mixture") && (Λ=ThetaMixturePrior(η0,δη))
    P0 = [ [Γ for i=1:N]; [Λ for i=1:M] ]
    if typeof(EPInputState) == Nothing
        EPInputState = EPState{Float64}(M+N,N)
    end
    callback==Reporter(w) && println("Printing (iter, Δav, MSEdB, overlap):")
    runtime =
        @elapsed ret=
        expectation_propagation(Vector{Term{Float64}}(undef,0),
        P0,ξ,offset*ones(M),damp=damp,maxiter=maxiter,epsconv=epsconv,maxvar=maxvar,
        minvar=minvar,callback=callback,state=EPInputState,inverter=x->inv(cholesky(Hermitian(x))))
    teacher,student=normalize_weights(w,ret,N);
    Overlap=teacher'student; println("α=$(M/N), $(ret.converged)");
    MSEdB=10.0*log10(Distances.msd(teacher,student))
    return ret, (Γ, Λ), runtime, MSEdB, Overlap
end
