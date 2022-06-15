% driver script for RLCM wrapper timing, posterior mean @ data. Barnett 6/15/22
clear; verb = 1;

N = 1e5;        % problem size (rank=300 takes 1 min for N=1e5)
l = 0.1;        % kernel length-scale
sigma = 0.1;    % noise sqrt(variance) in regression
dim = 3;        % spatial dim
% data = noisy version of known underlying function...
sigmadata = sigma;   % meas noise, consistent case
freqdata = 3.0;   % how oscillatory underlying func? freq >> 0.3/l misspecified
unitvec = randn(dim,1); unitvec = unitvec/norm(unitvec);
wavevec = freqdata*unitvec;    % col vec
f = @(x) cos(2*pi*x'*wavevec + 1.3);   % underlying func, must give col vec
[x, meas, truemeas] = get_randdata(dim, N, f, sigmadata);

% GP kernel
ker = SE_ker(dim,l);
%ker = Matern_ker(dim,1/2,l);   % kernel has little difference on speed

% RLCM params
opts.nthread = 4;      % little effect on speed  (slows for eg >8 threads)
opts.diageps = 1e-10;   % ? eg claims 1e-8
opts.refine = 1;       % v little effect on speed
opts.verb = 0;   % 0 silent; 1 show RLCM system calls & timing breakdown
                 % (75% of time is KRR_RLCM.Test)

ranks = [100 200 300];    % convergence test
for i=1:numel(ranks)      % .....
  opts.rank = ranks(i);
  fprintf('\nRLCM dim=%d, N=%d, sigma=%.3g, rank=%d, refine=%d...\n',dim,N,sigma,opts.rank,opts.refine)
  [y, ~, info] = RLCM(x, meas, sigma^2, ker, [], opts);     % regress
  mu{i} = y.mean;   % save the result (posterior mean at data pts)
  fprintf('\tRLCM total time %.3g s\n',info.cpu_time.total)
  fprintf('\ty.mean: rms err vs meas data %.3g\t(should be about sigmadata=%.3g)\n', rms(y.mean-meas),sigmadata)
end                       % .....

if numel(mu)>1,
  mudiff = mu{end}-mu{end-1};
  fprintf('\nest rel err (diff btw last 2 runs): %.3g (l2),\t%.3g (l-infty).\n',norm(mudiff)/norm(mu{end}), norm(mudiff,inf)/norm(mu{end},inf))
end
