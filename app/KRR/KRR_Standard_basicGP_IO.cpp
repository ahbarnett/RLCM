// This program performs kernel ridge regression by using the Standard
// method. The actual machine learning task can be either regression,
// binary classification, or multiclass classification.
//
// For large data sets, this program may not be able to handle the
// test data all at once. Hence, a parameter Budget is used to split
// the test data into batches, each handled at a time. The number of
// test data in one batch is approximately Budget/N, where N is the
// number of training data.
//
// The current implementation supports the following kernels:
// isotropic Gaussian, isotropic Laplace, product Laplace, and inverse
// multiquadric. For their definitions, see the corresponding .hpp
// files under ${CMATRIX_DIR}/src/Kernels/. The required paramters are
// sigma (bandwidth) and lambda (regularization). The global scaling s
// is always 1.
//
// The current implementation supports two data types for storing
// points: dense (DPoint) and sparse (SPoint). The supported data file
// format is LibSVM format. Even though LibSVM format always treats
// the data sparse, some data are in fact (almost) fully dense. The
// linear algebra is different for dense and sparse points. From a
// practical stand point, using the dense format is often more
// efficient, unless memory explodes.
//
// A note on LibSVM format: The LibSVM reader implemented in this
// software treats the attribute index starting from 1. For binary
// classifications, the labels must be +1/-1. For multiclass
// classifications, the labels must start from 0 and must be
// consecutive integers.
//
// The current implementation supports parallelism. One may set the
// USE_OPENBLAS flag at compile-time so that all the BLAS and LAPACK
// routines are threaded. Additionally, one may set the USE_OPENMP
// flag so that other parts are threaded.
//
// Compile-time macros:
//
//   KernelType:       One of IsotropicGaussian, IsotropicLaplace,
//                     ProductLaplace, InvMultiquadric
//   PointFormat:      One of DPoint, SPoint
//   PointArrayFormat: One of DPointArray, SPointArray. Must be
//                     consistent with PointFormat
//   USE_OPENBLAS:     Either use this macro or not (no value)
//   USE_OPENMP:       Either use this macro or not (no value)
//
// Usage:
//
//   KRR_Standard_basicGP_IO_<kerneltype>_<pointtype>.ex NumThreads Ntrain FileTrain Ntest FileTest FilePred d sigma var lambda
//
//      note (ahb): only does G regression, and one sigma, lambda (=?)
//
//   NumThreads:  Number of threads
//   Ntrain : # training pts
//   FileTrain:   File name (including path) of the data file for
//                training (format is raw binary double-prec interleaved:
//                   coords_1 y_1 coords_2 y_2 ... coords_NTrain y_NTrain)
//                    where coords_j = [x1_j x2_j ... xd_j] is a point.
//   Ntest : # test pts
//   FileTest:    File name (including path) of the data file for
//                testing (just coords of points xtrg).
//   FilePred:   Output file name for pred means at test pts (raw doubles, binary)
//   d:           Dimension of the data
//   sigma:   param sigma (lengthscale of kernel, usually called \ell).
//   var :  param k(0) prior variance
//   lambba:  param lambda = nugget I think ?  Ie what GPs call sigma^2 ?
//               inferred by reading test/Test_IsotropicGaussian.cpp

#include "KRR_Common.hpp"

#include <cstdio>


int main(int argc, char **argv) {

  //---------- Parameters from command line --------------------
  if (argc!=11) {
    printf("wrong number of cmd args!\n");
    exit(1);
  }
  INTEGER idx = 1;
  int NumThreads = atoi(argv[idx++]);               // Number of threads
  INTEGER Ntrain = String2Integer(argv[idx++]); // Number of training pts
  char *FileTrain = argv[idx++];                    // Training data
  INTEGER Ntest = String2Integer(argv[idx++]); // Number of test pts
  char *FileTest = argv[idx++];                     // Testing data
  char *FilePred = argv[idx++];                     // pred output file
  INTEGER d = String2Integer(argv[idx++]);          // Data dimension
  double sigma = atof(argv[idx++]);                // ell
  double var0 = atof(argv[idx++]);                   // aka s, var k(0)
  double lambda = atof(argv[idx++]);              // aka sigma^2 nugget

  //---------- Read in data --------------------

  PointArrayFormat Xtrain; // Training points
  DVector ytrain;          // Training labels
  PointArrayFormat Xtest;  // Testing points
  DVector ypred;   // Predictions ie output

  Xtrain.Init(Ntrain,d);
  ytrain.Init(Ntrain);
  double *px = Xtrain.GetPointer();     // not sure how template to single etc
  double *py = ytrain.GetPointer();
  FILE *fp=NULL;
  fp = fopen(FileTrain, "rb");
  for (INTEGER i=0;i<Ntrain;++i) {
    fread(px + i*d,d,sizeof(double),fp);  // read a point (all coords)
    fread(py + i,1,sizeof(double),fp);   // read corresp y val
    if (0 && i==0) {
      for (INTEGER j=0;j<d;++j) printf("%g ",px[i*d + j]);
      printf(" y=%g\n",py[i]);
    }
  }
  fclose(fp);
  
  Xtest.Init(Ntest,d);
  px = Xtest.GetPointer();
  fp = fopen(FileTest, "rb");
  for (INTEGER i=0;i<Ntest;++i) {
    fread(px + i*d,d,sizeof(double),fp);  // read a point (all coords)
    if (0 && i<2) {
      for (INTEGER j=0;j<d;++j) printf("%g ",px[i*d + j]);
      printf("\n");
    }
  }
  fclose(fp);
  ypred.Init(Ntest);

  
  //---------- Threading --------------------

#ifdef USE_OPENBLAS
  openblas_set_num_threads(NumThreads);
#elif defined USE_OPENMP
  omp_set_num_threads(NumThreads);
#else
  NumThreads = 1; // To avoid compiler warining of unused variable
#endif

  //---------- Main computation --------------------

  PREPARE_CLOCK(true);

  KRR_Standard<KernelType, PointFormat, PointArrayFormat> mKRR_Standard;

  // var0 = k(0) the prior var
  KernelType mKernel(var0, sigma);
  
  // Training (factorizes K+lamda*I only - see src/KRR/KRR_Standard.hpp)
  // Note that the kernel k(x,y) when x==y gives var0+lambda (adds "nugget")
  // see src/Kernels/IsotropicGaussian.tpp
  START_CLOCK;
  double MemEst = mKRR_Standard.Train(Xtrain, mKernel, lambda);
  END_CLOCK;
  double TimeTrain = ELAPSED_TIME;
  // we are bad since we should not really clobber stdout like this...
  printf("\tKRR_Standard.Train: (Ntrain=%d, dim=%d) param = %g %g, time = %g, mem_per_pt = %g\n", Ntrain, d, sigma, lambda, TimeTrain, MemEst);
  

  // do predictions? I guess.  There's no doc for KRR_Standard.Test....
  START_CLOCK;
  mKRR_Standard.Test(Xtrain, Xtest, ytrain, mKernel, ypred);
  END_CLOCK;
  double TimeTest = ELAPSED_TIME;
  printf("\tKRR_Standard.Test: (Ntest=%d) time = %g\n", Ntest, TimeTest);

  
  //----------- Write out  predicted mean y values at test pts ......
  py = ypred.GetPointer();
  fp = fopen(FilePred, "wp");
  for (INTEGER i=0;i<Ntest;++i) {
    // if (i<2) printf("ypred[%d] = %g\n",i,py[i]);      // checked ok
    fwrite(py + i,1,sizeof(double),fp);  // write a val
  }
  fclose(fp);
  
  //---------- Clean up --------------------      no free of Xtrain etc?
  return 0;

}
