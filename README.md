# RLCM: Recursively Low-Rank Compressed Matrices

Jie Chen, MIT-IBM Watson AI Lab, IBM Research

[Hacked fork version by Alex Barnett, with MATLAB interface to KRR tasks, which is the same as GP regression. This is done by system-calling new KRR_Standard and KRR_RLCM versions in app/KRR that read in files and write out the regressed mean vector at targets. Also Chen's code-generated KRR_binary directory is switched off. For tests see RLCM.m in ahb_try/ ]

RLCM is a library, written in C++, that offers the low-level linear algebra routines for a type of structured matrices [(Chen 2014a)](#Chen2014a), [(Chen 2014b)](#Chen2014b) resulting from hierarchical compressions of fully dense covariance/kernel matrices, as well as the high-level application routines for statistics and machine learning, including Gaussian processes [(Chen and Stein 2017)](#Chen2017b) and kernel methods [(Chen et al. 2017)](#Chen2017a).

The motivation of this matrix structure is to reduce the complexity of computing with large-scale fully dense covariance/kernel matrices from O(n<sup>3</sup>) to O(n), while maintaining a similar performance to the use of the original kernel function and achieving stable matrix computations without approximation. The supported matrix operations include

- matrix-vector multiplication;
- matrix inversion;
- matrix determinant;
- Cholesky-like factorization; and
- other special inner products and quadratic forms inherent to Gaussian processes and kernel methods.

The inverted matrix and the Cholesky-like factor have exactly the same matrix structure as the original matrix.

Additionally, this library offers

- support for dense and sparse matrices;
- iterative preconditioned linear solvers PCG and GMRES;
- various covariance kernels and test functions;
- sampling a Gaussian process;
- kriging;
- log-likelihood calculation; and
- kernel ridge regression.

## Philosophy

We inevitably need to do approximation. In order to reduce the cost of dense linear algebra to O(n), all these matrix computations can be done only approximately: matrix-vector multiplication, inversion, determinant, Cholesky, etc. We are faced with two options: approximately compute each of these, or approximate the matrix itself only and compute each of these exactly (well, a more precise wording is "numerically stably"). Whereas existing methods generally explore the first option, the work/software here takes the latter approach. We approximate the kernel function (that defines the matrix) first, so that the resulting matrix possesses the "recursive low-rank" structure, and then design numerical algorithms that perform matrix computations stably.

## Platforms

This library has been tested successfully under

- macOS High Sierra Version 10.13.6, Darwin Kernel Version 17.7.0, x86_64
- Red Hat Enterprise Linux 6.9, Kernel Version 2.6.32-696, x86_64
- AHB 3/21/22 got going: Ubuntu (really Pop!OS) 21.04, GCC+openBLAS, x86_64

## Dependencies

- C++ compiler (optionally with OpenMP for multithreading)
- LAPACK
- (Optional) One of OpenBLAS, ESSL, MKL, and Accelerate for multithreaded LAPACK
- (Optional) Fortran compiler for the use of Matern kernel
- (Optional) Valgrind for debugging

## How to Use

1. Install the dependencies.

1. Edit `configure.sh` and execute it.

	```shell
	$ configure.sh
	```

1. Compile the library.

	```shell
	$ make
	```

1. Install the library. This step will generate two directories: `lib` and `include`. For convenience, these directories are placed under the same directory as is the source code.

	```shell
	$ make install
	```

1. Compile application codes. The directory `app` contains several example codes for Gaussian processes (see `app/GP`) and kernel ridge regression (see `app/KRR`). These codes include not only methods using the RLCM matrix structure, but also  other compared methods.

	```shell
	$ make apps
	```

1. Compile test codes. The directory `test` contains numerous test codes, including unit tests for library functions and the tests of application codes with toy data.

	```shell
	$ make tests
	```

1. Test.

	```shell
	$ cd test
	$ Test_All.sh
	```
	These tests serve two purposes: verify the correctness of the installation and the correctness of the implementation. Hence, they are more complex than the unit tests in the usual software engineering practice using `assert`. Specifically, one needs the knowledge of numerical linear algebra to understand whether the codes perform normally. For example, the discrepancy of two correct implementations of matrix-vector multiplications under double precision may be on the level of `1.0e-16` because of numerical roundoff; if the discrepancy reads `1.0e-10`, the algorithm/implementation is rather suspicious. Whereas the 2-norm difference between AA<sup>-1</sup> and the identity may be as high as `1.0e-06` in normal circumstances, because of the ill-conditioning of the matrix A.

	The tests are rather lengthy (but fast). One may want to redirect the screen output to log files for examination. The following command does so (assuming using `bash`). One may compare the log file with the one shipped by this library (`Test_All.log.reference`).
	
	```shell
	$ Test_All.sh 2>&1 | tee Test_All.log
	```
	
1. (Optional) This step will remove all the files generated in the previous steps (including the newly generated directories `lib` and `include`) and restore the source code to the factory state.

	```shell
	$ make uninstall
	```

1. Write your own application codes by following the code examples under `app/GP`, the makefile example `app/GP/Makefile.in`, and the usage example `test/Test_GP.sh`.

## Bibliography

- <a name="Chen2017b"></a>Jie Chen and Michael Stein. [Linear-cost covariance functions for Gaussian random fields](http://arxiv.org/abs/1711.05895). Preprint arXiv:1711.05895, 2017. To appear in Journal of the American Statistical Association.
- <a name="Chen2017a"></a>Jie Chen, Haim Avron, and Vikas Sindhwani. [Hierarchically compositional kernels for scalable nonparametric learning](http://jmlr.org/papers/v18/15-376.html). Journal of Machine Learning Research, 18(66):1–42, 2017.
- <a name="Chen2014b"></a>Jie Chen. [Computing square root factorization for recursively low-rank compressed matrices](https://jiechenjiechen.github.io/pub/rlcm_sqrt_root.pdf). Technical Report RC25499, IBM Thomas J. Watson Research Center, 2014.
- <a name="Chen2014a"></a>Jie Chen. [Data structure and algorithms for recursively low-rank compressed matrices](https://jiechenjiechen.github.io/pub/rlcm.pdf). Technical Report ANL/MCS-P5112-0314, Argonne National Laboratory, 2014.

