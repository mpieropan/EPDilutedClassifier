# EP simulations with i.i.d. and correlated Gaussian patterns and noisy labels

using EPDilutedClassifier

N=128; ρ=0.25; α=6.0;
epsconv=1e-4; maxiter=10000; η=0.95;
damp=0.99;
λ0=1e4
w=generate_weights(N,ρ);
report=Reporter(w)

# iid patterns
println("Test with i.i.d. Gaussian patterns")
run_instance_gaussian_iid(w,ρ,α,η,callback=report,ρ0=ρ,λ0=λ0,δρ=0.0,η0=η,δη=0.0,
        damp=damp,epsconv=epsconv,maxiter=maxiter)

# gaussian correlated patterns
println("Test with correlated Gaussian patterns")
run_instance_gaussian_cor(w,ρ,α,η,callback=report,ρ0=ρ,λ0=λ0,δρ=0.0,η0=η,δη=0.0,
        damp=damp,epsconv=epsconv,maxiter=maxiter)
