# EP simulations with correlated patterns generated from a recurrent neural network and noiseless labels

using EPDilutedClassifier

Nperceptrons=128; N=Nperceptrons-1
ith_perceptron=1
ρ=0.25
α=6.0
stepsize=25 # the lower the stepsize, the more correlated the patterns
αmax=α*stepsize
maxiter=50000
epsconv=1e-4
damp=0.999

prior=Lambda_prior(1.0)
teacher_net_weights=generate_weights_recnet(Nperceptrons,ρ);
pattern_set=generate_patterns_recnet_dynamics(teacher_net_weights,αmax)

i=ith_perceptron
w=teacher_net_weights[i,:]
K = length(findall(w.!=0.0))
report=Reporter(w)

X=pattern_set[1:stepsize:end,:]
T=size(X)[1]-1
ρ=K/N
α=T/N
println("ρ=$ρ, α=$α")
indices = union(1:i-1,i+1:Nperceptrons)
ξ_except_i = X[1:T,indices]
ξ_out_i = sign.(ξ_except_i*w)
ξ = ξ_out_i.*ξ_except_i

ret,(Γ,Λ),runtime,MSEdB,Overlap = run_instance(w,ξ,1.0,epsconv=epsconv,
    callback=report,ρ0=ρ,δρ=0.0,damp=damp,maxiter=maxiter)
