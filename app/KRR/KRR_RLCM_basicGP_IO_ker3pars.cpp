// This is a 3-kernel-param version of KRR_RLCM_basicGP_IO.cpp [ahb hack]

// This program performs kernel ridge regression by using the RLCM
// method. The actual machine learning task can be either regression,
// binary classification, or multiclass classification.
//
// The RLCM method requires a parameter Rank, which is approximately
// equal to N0, the leaf size in a hierarchical partitioning of the
// training data. Meanwhile, for large data sets, this program may not
// be able to handle the test data all at once. Hence, a parameter
// Budget is used to split the test data into batches, each handled at
// a time. The number of test data in one batch is approximately
// Budget/N0.
//
// The RLCM method also offers the option Par, which specifies the
// method for partitioning the data set when building kernel matrix.
// RAND is much more efficient than PCA.
//
// The current implementation supports the following kernels:
// isotropic Matern, or any kernel with class constructor taking 3 params.
// files under ${CMATRIX_DIR}/src/Kernels/. The required paramters are
// nu (power), sigma (bandwidth = ell lengthscale) and lambda (regularization).
// The global scaling s is always 1.
//
// The current implementation (ahb hack) supports DPoint, and a binary
// disk format. ahb hacked so NumClasses = 1 = regression.
//
// The current implementation supports parallelism. One may set the
// USE_OPENBLAS flag at compile-time so that all the BLAS and LAPACK
// routines are threaded. Additionally, one may set the USE_OPENMP
// flag so that other parts are threaded.
//
// Compile-time macros:
//
//   KernelType:       Any 3-param kernel, eg IsotropicMatern
//   PointFormat:      One of DPoint, SPoint
//   PointArrayFormat: One of DPointArray, SPointArray. Must be
//                     consistent with PointFormat
//   USE_OPENBLAS:     Either use this macro or not (no value)
//   USE_OPENMP:       Either use this macro or not (no value)
//
// Usage:
//
//   KRR_RLCM_basicGP_IO_ker3pars_<kerneltype>_<pointtype>.ex NumThreads Ntrain FileTrain Ntest FileTest FilePred d verb nu sigma var lambda seed rank par diagcorrent refinement
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
//   verb:        0 (silent) or 1 (diagnostic to stdout)
// [three kernel params + one nugget param:]
//   nu:      power for Matern
//   sigma:   param sigma (lengthscale of kernel, usually called \ell).
//   var :  param k(0) prior variance
//   lambba:  param lambda = nugget I think ?  Ie what GPs call sigma^2 ?
// [five RLCM method params:]
//   Seed:        If >= 0, will use this value as the seed for RNG;
//                otherwise, use the current time to seed the RNG.
//   Rank:        Rank
//   Par:         Partitioning method. One of RAND, PCA
//   DiagCorrect: Diagonal correction for CMatrix construction, e.g., 1e-8
//   Refinement:  Refine the linear solves? Either 0 or 1.


#include "./KRR_Common.hpp"

#include <cstdio>


int main(int argc, char **argv) {

  //---------- Parameters from command line --------------------
  if (argc!=18) {
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
  // kernel & nugget pars...
  double nu = atof(argv[idx++]);                   // nu
  double sigma = atof(argv[idx++]);                // ell
  double var0 = atof(argv[idx++]);                   // aka s, var k(0)
  double lambda = atof(argv[idx++]);              // aka sigma^2 nugget
  // RLCM pars....
  INTEGER sSeed = String2Integer(argv[idx++]);      // Seed for randomization
  unsigned Seed;
  if (sSeed < 0) {
    Seed = (unsigned)time(NULL);
  }
  else {
    Seed = (unsigned)sSeed;
  }
  INTEGER Rank = String2Integer(argv[idx++]);       // Rank
  char *ParString = argv[idx++];                    // Partitioning method
  PartMethod Par;
  if (strcmp(ParString, "RAND") == 0) {
    Par = RAND;
  }
  else if (strcmp(ParString, "PCA") == 0) {
    Par = PCA;
  }
  else {
    fprintf(stderr,"KRR_RLCM. Error: Unknown partitioning method!\n");
    return 0;
  }
  double DiagCorrect = atof(argv[idx++]);           // DiagCorrect
  bool Refinement = atoi(argv[idx++]) ? true : false; // Refinement

  
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
  for (int i=0;i<d;++i) {
    count = (INTEGER)fread(px + i*Ntrain,sizeof(double),Ntrain,fp);  // read a coord (all pts)
    if (count!=Ntrain) {
      fprintf(stderr,"error reading train coordinate i=%d!\n",i);
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
  for (int i=0;i<d;++i) {
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
  if (verb)
    printf("\tStarting RLCM ker3pars version, NumThreads=%d ...\n",NumThreads);
  //  *** problem, even when NumThreads=1, observe all threads used :(
  // Also, NumThreads = 16, say, causes terrible slow-down to a halt :(

  
  //---------- Main computation --------------------

  PREPARE_CLOCK(true);

  // note here KernelType is a compiler macro giving the kernel class, eg IsotropicGaussian...
  KRR_RLCM<KernelType, PointFormat, PointArrayFormat> mKRR_RLCM;
  INTEGER *Perm = NULL, *iPerm = NULL;
  INTEGER N = Xtrain.GetN();
  New_1D_Array<INTEGER, INTEGER>(&Perm, N);
  New_1D_Array<INTEGER, INTEGER>(&iPerm, N);

  // Pre-training
  START_CLOCK;
  double MemEst;
  MemEst = mKRR_RLCM.PreTrain(Xtrain, ytrain, Perm, iPerm, Rank, DiagCorrect,
                                Refinement, Seed, Par);
  END_CLOCK;
  double TimePreTrain = ELAPSED_TIME;
  if (verb) {
    printf("\tKRR_RLCM: pretrain time = %g, MemEst=%g\n", TimePreTrain, MemEst); fflush(stdout);
  }
    
  // set up the kernel: var0 = k(0) = the prior var. sigma = lengthscale
  KernelType mKernel(var0, nu, sigma);  // Matern only, really
  
  // Training
  // Note that the kernel k(x,y) when x==y gives var0+lambda (adds "nugget")
  // see src/Kernels/IsotropicGaussian.tpp
  START_CLOCK;
  mKRR_RLCM.Train(Xtrain, mKernel, lambda);
  END_CLOCK;
  double TimeTrain = ELAPSED_TIME;
  // we are bad since we should not really clobber stdout like this...
  if (verb)
    printf("\tKRR_RLCM.Train: (Ntrain=%d, dim=%d), ker params = %g %g %g, time = %g\n", Ntrain, d, nu, sigma, lambda, TimeTrain);
  
  // do predictions? (not "tests") I guess.  There's no doc for KRR_*.Test....
  START_CLOCK;
  mKRR_RLCM.Test(Xtrain, Xtest, ytrain, mKernel, ypred);
  END_CLOCK;
  double TimeTest = ELAPSED_TIME;
  if (verb)
    printf("\tKRR_RLCM.Test: (Ntest=%d) time = %g\n", Ntest, TimeTest);
  
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
    printf("\tKRR_RLCM done writing output file.\n");
  
  //---------- Clean up --------------------      no free of Xtrain etc?
  Delete_1D_Array<INTEGER>(&Perm);
  Delete_1D_Array<INTEGER>(&iPerm);
  return 0;
}
