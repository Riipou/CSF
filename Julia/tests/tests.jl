include("linear_euclidean_distance_matrices.jl")
include("multiple_slackgon.jl")
include("CD_extrapoled_test.jl")
include("multiple_test.jl")
include("CSFandNMF.jl")
using Dates

# Execution of tests
start = time()
# Comparing CD and extrapoled CD 
#= extrapoledCD_test() =#
# Test on euclidian distance matrices
multiple_distance_matrix()
# Test on slack matrices of regular n-gons
multiple_slackgon_matrix()
# Tests on synthetic data
multiple_test()
random_test()
# Data set tests
CBCL_test()
TDT2_test()
CBCLfacialfeatures_test()
sparse_matrices_test()
sparse_matrices_test2()
println(time()-start)