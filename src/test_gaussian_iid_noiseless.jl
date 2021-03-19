# EP simulations with i.i.d. patterns and noiseless labels


using .EPDilutedClassifier

N=128; ρ=0.25; α=6.0;
epsconv=1e-4; maxiter=50000; η=1.0; damp=0.9995;
w=generate_weights(N,ρ);
report=Reporter(w);

run_instance_gaussian_iid(w,ρ,α,η,callback=report,ρ0=ρ,
   λ0=1.0,δρ=0.0,η0=η,δη=0.0,damp=damp,epsconv=epsconv,maxiter=maxiter)
