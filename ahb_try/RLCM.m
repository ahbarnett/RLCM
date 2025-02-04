function [y, ytrg, info] = RLCM(x, meas, sigmasq, ker, xtrg, opts)
% RLCM   GP regression via recursively low-rank compressed matrices, Chen-Stein
%
% [y, ytrg, info] = RLCM(x, meas, sigmasq, ker, xtrg, opts)
%  performs Gaussian process regression using certain kernels (SE and Matern),
%  in arbitrary dimension (eg 1d, 2d, or 3d).
%
% Inputs:
%  x    - points (ordinates) where observations taken, d*N real array for d dims
%  meas - observations at the data points, length-N real (or complex?) array
%  sigmasq - noise variance at data points, nonnegative scalar
%  ker  - covariance kernel struct of gp-shootout form, at least the fields:
%         k    - kernel covariance function of separation distance
%         fam  - string name
%         l    - lengthscale
%  xtrg - [optional, or may be empty] targets points, d*n real array for d dims.
%         If non-empty, attempts to compute ytrg outputs.
%  opts - [optional] struct controlling method params including:
%         dense - true uses Chen's dense Standard O(N^3) method; false uses RLCM
%         verb  - 0 (default) for silent; >0 for text diagnostic output
%         seed - 0 for random, or use as RLCM seed.
%         rank - RLCM matrix block rank param (see paper)
%         par = 'RAND' or 'PCA' (string), partitioning method.
%         diageps - diagonal added value for CMatrix construction, eg 1e-8 ?
%         refine - refine the linear solves? (0 or 1).
%         only_trgs - only compute posterior mean at targets
%         nthreads - how many threads for RLCM executables (default all avail)
%
% Outputs:
%  y - struct with fields of regression results corresp to given data points x:
%     mean - posterior mean vector, N*1
%  ytrg - [optional; otherwise empty] struct of regression at new targets xtrg:
%     mean - posterior mean vector, n*1
%  info - diagnostic struct containing fields:
%     cpu_time - struct with fields including (in seconds): total
%
% If called without arguments, does a self-test.

% SE ker first; added Matern 5/5/22
if nargin==0, test_RLCM; return; end
if nargin<5, xtrg = []; end
if nargin<6, opts = []; end
if ~isfield(opts,'dense'), opts.dense = 0; end
if ~isfield(opts,'verb'), opts.verb = 0; end
if ~isfield(opts,'seed'), opts.seed = 0; end     % defaults for RLCM pars
if ~isfield(opts,'rank'), opts.rank = 125; end    % (see RLCM .cpp source)
if ~isfield(opts,'par'), opts.par='PCA'; end
if ~isfield(opts,'diageps'), opts.diageps = 1e-8; end
if ~isfield(opts,'refine'), opts.refine=1; end

[dim,N] = size(x);
if numel(meas)~=N, error('sizes of meas and x must match!'); end
n = size(xtrg,2);   % # new targets

if ~isfield(opts, 'only_trgs'), xtrg = [x, xtrg]; end
ntrgs = size(xtrg, 2);

tmpkey = 1; %randi([0 intmax('uint32')], 'uint32');
filetrain = sprintf('/tmp/RLCM_train_%x.tmp',tmpkey);
filextrg = sprintf('/tmp/RLCM_xtest_%x.tmp',tmpkey);
fileypred = sprintf('/tmp/RLCM_ypred_%x.tmp',tmpkey);
%fileytrgmean = sprintf('/tmp/RLCM_ytrgmean_%x.tmp',tmpkey);

fid = fopen(filetrain,'wb');
fwrite(fid,[x', meas(:)],'float64');  % stack flipped x and ymeas and write out
fclose(fid);                          % format is points-fast, dims-slow order
fid = fopen(filextrg,'wb');
fwrite(fid,xtrg','float64');          % similar format
fclose(fid);

% [x(:,1); meas(1)]   % debug

h = fileparts(mfilename('fullpath'));   % this dir
dir = [h '/../app/KRR'];     % app dir of ahb-hacked RLCM executables

if isfield(opts,'nthread')
  dir = sprintf('OMP_NUM_THREADS=%d %s',opts.nthread,dir);  % for system call
end

if opts.dense              % toy small-N dense test
  exechead = 'KRR_Standard_basicGP_IO';
  methargs = [];
else                       % the real thing: RLCM fast alg for large N
  exechead = 'KRR_RLCM_basicGP_IO';
  methargs = sprintf('%d %d %s %g %d', opts.seed, opts.rank,opts.par, opts.diageps, opts.refine);
end

% note the following only changes the command args which seems not to affect
% actual # threads (hence the above hack on dir to control via OMP_NUM_THREADS)
nthread = 1; %maxNumCompThreads;   % 1 is best but still uses all threads
             % lots of threads cause terrible slow-down in executable :(

if strcmp(ker.fam,'squared-exponential')  % 2 pars: var, ell (also sigma^2 nugg)
  cmd = sprintf('%s/%s_IsotropicGaussian_DPoint.ex %d %d %s %d %s %s %d %d %.15g %.15g %.15g %s',dir,exechead,nthread,N,filetrain,ntrgs,filextrg,fileypred,dim,opts.verb,ker.l,ker.k(0),sigmasq,methargs);
  % cmd arg ordering must match RLCM/app/KRR/KRR_RLCM_basicGP_IO.cpp
elseif strcmp(ker.fam(1:6),'matern')      % 3 pars: var, nu, ell; also sigma^2
  if opts.dense, error('KRR_Standard (dense) not implemented for Matern!'); end
  % notice since Matern has 1 extra param, need different executable than SE:
  cmd = sprintf('%s/%s_IsotropicMatern_DPoint.ex %d %d %s %d %s %s %d %d %.15g %.15g %.15g %.15g %s',dir,exechead,nthread,N,filetrain,ntrgs,filextrg,fileypred,dim,opts.verb,ker.nu,ker.l,ker.k(0),sigmasq,methargs);
  % cmd arg ordering must match RLCM/app/KRR/KRR_RLCM_basicGP_IO_ker3pars.cpp
end
if opts.verb, cmd, end          % report
t0=tic;
status = system(cmd);
if status~=0, error('executable had error exit code, stopping!'); end
info.cpu_time.total = toc(t0);

fid = fopen(fileypred,'rb');
[ypred count] = fread(fid,ntrgs,'float64');
fclose(fid);
if count~=ntrgs, error('cannot read correct number ypred vals!'); end

if isfield(opts, 'only_trgs')
    y.mean = [];
    ytrg.mean = ypred; 
else
    y.mean = ypred(1:N);   % hack for now to split out posterior means into two types
    ytrg.mean = ypred(N+1:end);
end
if opts.verb, disp('RLCM done'); end


%%%%%%%%%%
function test_RLCM   % basic tests for now, duplicates naive_gp
N = 5e3;        % problem size (checking vs naive if <=1e4)
l = 0.1;        % SE kernel scale rel to domain [0,1]^dim, ie hardness of prob
sigma = 0.3;    % noise level used for GP regression
sigmadata = sigma;   % meas noise, consistent case
freqdata = 3.0;   % how oscillatory underlying func? freq >> 0.3/l misspecified
L=1; shift = 0;   % nodes in [0,1]^dim
%L = 50.0; shift = 200;   % arbitary, tests correct centering and L-box rescale
opts.dense = 0;   % 0 = RLCM; 1 = force KRR_Standard (dense = not RLCM)
opts.verb = 0;
opts.diageps = 1e-10;
%opts.par = 'PCA';  % seems better than RAND

for dim = 1:3   % ..........
  fprintf('\n ======== DIM %d =========\n',dim);
  % the main convergence params for RLCM... d>1 acc seems poor even for big rank
  opts.rank = 120*dim;  % 125 std = fast (1e4 pts/s), 1000 = slow (300 pts/s)
  for kertype = 1:4   % ========
  % pick your kernel... (note L scales its scale as well as the coords x)
    if kertype==1
      ker = SE_ker(dim,L*l);
    else
      ker = Matern_ker(dim,kertype-3/2,L*l);   % type=2,3,4 gives nu .5,1.5,2.5
    end  
    fprintf('\ntest RLCM: ker=%s, ell=%.3g, sigma=%.3g...\n',ker.fam, ker.l, sigma)
    unitvec = randn(dim,1); unitvec = unitvec/norm(unitvec);
    wavevec = freqdata*unitvec;    % col vec
    f = @(x) cos(2*pi*x'*wavevec + 1.3);   % underlying func, must give col vec
    rng(1); % set seed
    [x, meas, truemeas] = get_randdata(dim, N, f, sigmadata);  % x in [0,1]^dim
    x = L*x + (2*rand(dim,1)-1)*shift;                         % scale & shift
    [y, ~, info] = RLCM(x, meas, sigma^2, ker, [], opts);
    fprintf('CPU time (s):\t%.3g\t(%.3g pts/s)\n',info.cpu_time.total, N/info.cpu_time.total);
    fprintf('y.mean: rms err vs meas data   %.3g\t(should be about sigmadata=%.3g)\n', rms(y.mean-meas),sigmadata)
    % estim ability to average away noise via # pts in rough kernel support...
    fprintf('        rms truemeas pred err  %.3g\t(should be sqrt(l^d.N) better ~ %.2g)\n', rms(y.mean-truemeas),sigmadata/sqrt(l^dim*N))
    % make sure we're computing gp regression accurately
    if N<=1e4        % compare O(n^3) naive gp regression, our implementation
      [ytrue, ytrg, ~] = naive_gp(x, meas, sigma^2, ker, [], opts);
      % fprintf('first 10 entries of y.mean vec and ytrue.mean vec...\n');
      % disp([y.mean(1:10) ytrue.mean(1:10)])  % debug
      fprintf('        rms RLCM vs naive      %.3g\n', rms(y.mean-ytrue.mean))
    end

    if 0       % show pics
      figure;
      if dim==1, plot(x,meas,'.'); hold on; plot(x,y.mean,'-');
      elseif dim==2
        subplot(1,2,1); scatter(x(1,:),x(2,:),[],meas,'filled');
        caxis([-1 1]); axis equal tight
        subplot(1,2,2); scatter(x(1,:),x(2,:),[],y.mean,'filled');
        caxis([-1 1]); axis equal tight
      elseif dim==3
        subplot(1,2,1); scatter3(x(1,:),x(2,:),x(3,:),[],meas,'filled');
        caxis([-1 1]); axis equal tight
        subplot(1,2,2); scatter3(x(1,:),x(2,:),x(3,:),[],y.mean,'filled');
        caxis([-1 1]); axis equal tight
      end
      title(sprintf('RLCM test %dd',dim)); drawnow;
    end

  end          % ============
end             % ..........
