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
// The current implementation (ahb hack) supports DPoint, and a binary
// disk format. ahb hacked so NumClasses = 1 = regression.
//
// The current implementation supports parallelism. One may set the
// USE_OPENBLAS flag at compile-time so that all the BLAS and LAPACK
// routines are threaded. Additionally, one may set the USE_OPENMP
// flag so that other parts are threaded.
//
// Compile-time macros: (*** ahb not updated)
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
//      note (ahb): only does GP regression, single set of kernel params
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
//   FilePred:   Output file for pred means at test pts (raw doubles, binary)
//   d:           Dimension of the data
//   verb:        verbosity (0 = silent, 1 = diagnostics to stdout).
//   sigma:   param sigma (lengthscale of kernel, usually called \ell).
//   var :  param k(0) prior variance
//   lambba:  param lambda = nugget I think ?  Ie what GPs call sigma^2 ?
//               inferred by reading test/Test_IsotropicGaussian.cpp

#include "KRR_Common.hpp"

#include <cstdio>


int main(int argc, char **argv) {

  //---------- Parameters from command line --------------------
  if (argc!=12) {
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
  int verb = String2Integer(argv[idx++]);          // verbosity
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
  INTEGER count;
  FILE *fp=NULL;
  fp = fopen(FileTrain, "rb");
  for (INTEGER i=0;i<d;++i) {
    count = (INTEGER)fread(px + i*Ntrain,sizeof(double),Ntrain,fp);  // read a coord (all pts)
    if (count!=Ntrain) {
      fprintf(stderr,"error reading train coordinates i=%d!\n",i);
      return 1;
    }
  }
  count = (INTEGER)fread(py,sizeof(double),Ntrain,fp);   // corresp y vals
  if (count!=Ntrain) {
    fprintf(stderr,"error reading train y vals!\n");
    return 1;
  }
  fclose(fp);
  
  Xtest.Init(Ntest,d);
  px = Xtest.GetPointer();
  fp = fopen(FileTest, "rb");
  for (INTEGER i=0;i<d;++i) {
    count = (INTEGER)fread(px + i*Ntest,sizeof(double),Ntest,fp);  // read a coord (all points)
    if (count!=Ntest) {
      fprintf(stderr,"error reading test coordinate i=%d!\n",i);
      return 1;
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
  if (verb)
    printf("\tKRR_Standard.Train: (Ntrain=%d, dim=%d) param = %g %g, time = %g, mem_per_pt = %g\n", Ntrain, d, sigma, lambda, TimeTrain, MemEst);
  

  // do predictions? I guess.  There's no doc for KRR_Standard.Test....
  START_CLOCK;
  mKRR_Standard.Test(Xtrain, Xtest, ytrain, mKernel, ypred);
  END_CLOCK;
  double TimeTest = ELAPSED_TIME;
  if (verb)
    printf("\tKRR_Standard.Test: (Ntest=%d) time = %g\n", Ntest, TimeTest);

  
  //----------- Write out  predicted mean y values at test pts ......
  py = ypred.GetPointer();
  fp = fopen(FilePred, "wp");
  count = (INTEGER)fwrite(py,sizeof(double),Ntest,fp);  // write all vals
  if (count!=Ntest) {
    fprintf(stderr,"error writing pred y vals!\n");
    return 1;
  }
  fclose(fp);
  if (verb)
    printf("\tKRR_Standard done writing output file.\n");
  
  //---------- Clean up --------------------      no free of Xtrain etc?
  return 0;

}
