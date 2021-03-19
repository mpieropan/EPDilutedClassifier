module EPDilutedClassifier

using Random, LinearAlgebra, Distances, Distributions

using GaussianEP: EPState, SpikeSlabPrior, ThetaMixturePrior, ThetaPrior, expectation_propagation, Term, @extract



include("generate_instances.jl")
include("run_instances.jl")
include("epwrapper.jl")

export run_instance, run_instance_gaussian_iid, run_instance_gaussian_cor
export run_instance_cor_from_recnet
export EP_Regress_ZeroTemp, initialize, Lambda_prior
export generate_weights
export generate_gaussian_iid_patterns, generate_gaussian_cor_patterns
export generate_weights_recnet
export generate_patterns_recnet_dynamics
export Reporter

end # end module SparsePerceptron
