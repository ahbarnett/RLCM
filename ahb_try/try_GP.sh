#!/bin/bash
#
# AHB trying to call a single GP, simplest case, controling params. 3/21/22

# uses the applications under ../app/GP : GP_RLCM.ex

# seems more threads doesn't speed up :( (maybe factor <2 in Kriging time)...
export NumThreads=1
# regardless, it's about 700 pts/sec for sampling, same for kriging.

# if empty, doesn't do slow mem leak checks, I guess...
export Valgrind=
# NOTE: Valgrind check is very slow. Reduce the number of ell, nu, and
# tau.

export OMP_NUM_THREADS=${NumThreads}

export Dir="../app/GP"
# overall dim...?
export d=2
# grid size:   N = N_train = prod(Dim) * PortionTrain  ,  eg N = 1e4
export Dim=(100 200)
# grid lies over a rectangle apparently...
export Lower=(-0.8 -1.0)
export Upper=(+0.8 +1.0)
# Matern params: ell=lengthscale.   sigma^2 ("nugget") = 10^tau
export ell=0.1
export nu=1.5
export tau=-3
# do a single log-lik eval at the true params (no grid search)...
export List_ell=(${ell})
export List_nu=(${nu})
export List_tau=(${tau})

export Seed=1
# this randomly samples from the grid (see GP_Common.hpp)...
export PortionTrain=0.5

# not sure what used for (checking grad of L?) ... 3 steps is for 3 params
export IsCheckFiniteDiff=0
export DiffStepSize=(1e-5 1e-5 1e-5)

export Num_ell=${#List_ell[*]}
export Num_nu=${#List_nu[*]}
export Num_tau=${#List_tau[*]}

# some HODLR rank, method params
export r=250
export DiagCorrect=1e-8

# just run the RLCM alg...  (not the Standard, whatever that is)
# this output flag works...
export OutputRandomField=0
export RandomFieldFileBasename="try_GP_RLCM_RF"
export OutputLogLik=0
export LogLikFileName="try_GP_RLCM_LogLik.txt"
export ComputeFisher=0
export OutputFisher=0
export FisherFileName="try_GP_RLCM_Fisher.txt"
# but output flags here seem to have no effect...
export OutputKrigedRandomField=0
export KrigedRandomFieldFileBasename="try_GP_RLCM_RF_kriged"
export OutputPredictions=1
export PredictionsFileName="try_GP_RLCM_Pred.txt"

(set -x; ${Valgrind} ${Dir}/GP_RLCM.ex ${NumThreads} ${d} ${Dim[*]} ${Lower[*]} ${Upper[*]} ${ell} ${nu} ${tau} ${Num_ell} ${List_ell[*]} ${Num_nu} ${List_nu[*]} ${Num_tau} ${List_tau[*]} ${r} ${DiagCorrect} ${Seed} ${PortionTrain} ${IsCheckFiniteDiff} ${DiffStepSize[*]} ${OutputRandomField} ${RandomFieldFileBasename} ${OutputLogLik} ${LogLikFileName} ${ComputeFisher} ${OutputFisher} ${FisherFileName} ${OutputKrigedRandomField} ${KrigedRandomFieldFileBasename} ${OutputPredictions} ${PredictionsFileName})
