% PK: 17/08/2019

% preamble
close all
clear
rng('default');

% Import MPC tools
mpc = import_mpctools();

% Parameters and sizes for the nonlinear system
Nx = 21;
Nu = 11;
Nz = 14; % Number of algebraic states.
Np = 1;
Ny = 13;
Nd = Ny; % Number of disturbance states for closed-loop MPC simulation. 
Delta = 600;

% Plant functions.
% fxzup
fxzup = mpc.getCasadiFunc(@(x, z, u, p) PolymerizationModel_fxzup(x, z, u, p), ...
                           [Nx, Nz, Nu, Np], {'x', 'z', 'u', 'p'}, {'fxzup'});

% gxzup                          
gxzup = mpc.getCasadiFunc(@(x, z, u, p) PolymerizationModel_gxzup(x, z, u, p), ...
                          [Nx, Nz, Nu, Np], {'x', 'z', 'u', 'p'}, {'gxzup'});

% hxzup                      
hxzup = mpc.getCasadiFunc(@(x, z, u, p) PolymerizationModel_hxzup(x, z, u, p), ...
                          [Nx, Nz, Nu, Np], {'x', 'z', 'u', 'p'}, {'hxzup'});

% DAE integrator.
dae_integrator = mpc.getCasadiDAE('f', fxzup, 'g', gxzup, 'Delta', Delta,...
                                  'funcname', ['PolymerizationModel']); 
                                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Guestimate of steady-state values. 
% Steady-state values.
xs = [442; 97.68; 174.38;56.73;1.13;1.13e-2;
      356;277;283.71;3.27;3.17;4.8e+3;2.51e+3;
      1.41e+7;3.99e+6;1.22e+6;1.21e+6;3.59e+9;
      1.93e+9;2.39e+6;5.12e+4];

%xs(12:19) = xs(12:19)*(1e-10);
zs = ones(Nz, 1)*(1e+4);

us = [220; 7; 2; 0.9; 1.3e-2; 1e-4; 9e-2;
      8.5e+5; 0.5; 276; 0.005];
  
ps = 300;

% Simulate a few steps so we actually get dx/dt = 0 at steady state.
Nt = 10000;
for i = 1:Nt
     outputs = full(dae_integrator('x', xs, 'z', zs, 'u', us, 'p', ps));
     xs = full(outputs.xplus);
     zs = full(outputs.zplus);
end        



dzbydt = full(gxzup(xs, zs, us, ps));
dxbydt = full(fxzup(xs, zs, us, ps));

ys = full(hxzup(xs, zs, us, ps));

%% Firstly let us assume that we have full state feedback. 
% The variables to store the closed-loop parameters. 
xcl = NaN(Nx, Nsim + 1);
zcl = NaN(Nz, Nsim + 1);
ycl = NaN(Ny, Nsim);
ucl = NaN(Nu, Nsim);

% The state and the disturbance estimates. 
xhat = NaN(Nx, Nsim);
dhat = NaN(Nd, Nsim);
%% Trajectory lengths
Ntr = 350; 
Ncontroller = 30; % Controller Forecasting Horizon
Nmhe = 30; % Moving Horizon Estimator Horizon

% The Q, R, and the S matrices. 
Q = (1e-2)*diag(1./xs.^2);
R = (1e-2)*diag(1./us.^2);
S = (1e-2)*diag(1./us.^2);

% The upper and lower bounds on the control inputs.
% TODO: --
ulb 
uub

%% Build the Controller. 
lb = struct('u', ulb);
ub = struct('u', uub);

% Regulator
N = struct('x', Nx, 'u', Nu, 't', Ncontroller);
l = mpc.getCasadiFunc(@(x, u, xs, us, Du) stagecost(x, u, xs, us, Du, Q, R, S),...
           [Nx, Nu, Nx, Nu, Nu], {'x', 'u','xs','us', 'Du'}, {'l'});
Vf = mpc.getCasadiFunc(@(x, xsp) stagecost(x, zeros(Nu,1), xsp, zeros(Nu,1), zeros(Nu,1), Q, R, S),...
           [Nx, Nx], {'x', 'xsp'}, {'Vf'});
% Parameters for the controllers
pars = struct();
pars.d = zeros(Nd, 1);
pars.xs = repmat(xs, 1, Ncontroller + 1);
pars.us = repmat(us, 1, Ncontroller);
pars.p = ps; % Only used by the nominal controller
% Initial Guess
guess = struct();
guess.x = pars.xs;
guess.u = pars.us;
% Gather everything to pass everything to MPC tools at once
kwargs = struct();
kwargs.N = N;
kwargs.x0 = zeros(Nx, 1);
kwargs.l = l;
kwargs.Vf = Vf;
kwargs.lb = lb;
kwargs.ub = ub;
kwargs.uprev = us;
kwargs.par = pars;
kwargs.guess = guess;
% Controllers
controller = mpc.nmpc('f', Fnom, '**', kwargs);

%% Assembling the target selector. 
N = struct('x', Nx, 'u', Nu, 'y', Nx);
hxd = mpc.getCasadiFunc(@(x, d) x + Cd*d, ...
                         [Nx, Nd], {'x', 'd'}, {'hxd'});
lstarg = mpc.getCasadiFunc(@(y, u, ysp, usp) (y(CVs)-ysp(CVs))'*(y(CVs)-ysp(CVs)), ...
                         [Nx, Nu, Nx, Nu], {'y', 'u', 'ysp', 'usp'}, {'lstarg'});
% Guess for the target selector
guess = struct();
guess.x = xs;
guess.u = us;
guess.y = xs;
% Parameters 
pars = struct();
pars.d = zeros(Nd, 1);
pars.ysp = xs;
pars.usp = us;
pars.p = ps;
% Gather everything to pass to target selectors
kwargs = struct();
kwargs.h = hxd;
kwargs.l = lstarg;
kwargs.N = N;
kwargs.lb = lb;
kwargs.ub = ub;
kwargs.guess = guess;
kwargs.par = pars;                      
% Target selectors
sstargs = struct();
sstargs.nom = mpc.sstarg('f', Fnom, '**', kwargs);
sstargs.lin = mpc.sstarg('f', Flin, '**', kwargs);
sstargs.nn = mpc.sstarg('f', Fnn, '**', kwargs);

% Assembling nonlinear estimators
Qinv = diag([1, 1, 1, 1]);
Rinv = Qinv;
Qdinv = Qinv;
Pinv = blkdiag(Qinv, Qdinv);
% Stage cost
lmhe = mpc.getCasadiFunc(@(w, v, Dd) w'*Qinv*w + v'*Rinv*v + Dd'*Qdinv*Dd, ...
                          [Nx, Nx, Nd], {'w', 'v', 'Dd'}, {'lmhe'});
% Prior cost
lx = mpc.getCasadiFunc(@(x,xhat, d, dhat) priorcost(x, xhat, d, dhat, Pinv), ...
                        [Nx, Nx, Nd, Nd], {'x', 'xbar', 'd', 'dbar'}, {'lx'});
N = struct('x', Nx, 'u', Nu, 'y', Nx, 'd', Nd, 't', Nmhe);
% Initial Guess for the MHE optimizer
guess = struct();
guess.x = repmat(xs, 1, Nmhe + 1);
guess.d = repmat(zeros(Nd, 1), 1, Nmhe + 1);
% Structs for constraints
lb = struct('x', xlb);
ub = struct('x', xub);
% Parameters for the optimizer
pars = struct();
pars.y = repmat(xs, 1, Nmhe + 1);
pars.u = repmat(us, 1, Nmhe);
pars.xbar = zeros(Nx, 1);
pars.dbar = zeros(Nd, 1);
pars.p = ps;
% Gather Everything
kwargs = struct();
kwargs.h = hxd;
kwargs.l = lmhe;
kwargs.lx = lx;
kwargs.N = N;
kwargs.guess = guess;
kwargs.lb = lb;
kwargs.ub = ub;
kwargs.par = pars;
kwargs.wadditive = true();
% Finally, the three estimators
estimators = mpc.nmhe('f', Fnom, '**', kwargs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Start loop over time. 
for t = 1:Nsim 
        
    % Get the measurement with noise. 
    ycl(:, t) = full(hxzup(xcl(:, t), zcl(:, t), ucl(:, t), p(:, t))) + ;
    ycl(:, t) = mvnrnd(zeros(1, Ny), Rv)';
    
    % MHE 
    estimator.newmeasurement(ycl(:, t), uprev);
    estimator.par.xbar = xprior(:, 1);
    estimator.par.dbar = dprior(:, 1);
    estimator.solve();
    fprintf('Estimator: %s, \n', estimator.status);
    if ~isequal(estimator.status, 'Solve_Succeeded')
        fprintf('\n');
        warning('mhe failed at time %d!', t);
        break
    end
    xhat(:, t) = estimator.var.x(:, end);
    dhat(:, t) = estimator.var.d(:, end);
   
    %xhatd(:,i)
    %dhat(:,i)
    % xd(:,i)
    % Checking disturbances    
    % mhed.var.d(:,end)
    % mhed.var.Dd(:,end)+mhed.var.d(:,end-1)

    % Checking state equations
    % mhed.var.x(:,end)
    % mhed.var.w(:,end)+full(fxud(mhed.var.x(:,end-1),uprev,mhed.var.d(:,end-1)))

    % Checking measurement equations
    % yd(:,i)
    % mhed.var.x(:,end)+Cd*mhed.var.d(:,end)+mhed.var.v(:,end)    
 
    estimator.saveguess();
    xprior = [xprior(:, 2:end), xhat(:, t)];
    dprior = [dprior(:, 2:end), dhat(:, t)];
    
    %xhat(:,i)
    %dhat(:,i)
    % Use steady-state target selector.
    %sstargd.fixvar('y', 1, ysp(CVs,i), CVs);
    sstarg.par.ysp = ysp(:, t);
    sstarg.par.d = dhat(:, t);
    sstarg.solve();
    %sstargd.var.x
    %sstargd.var.u
    fprintf('Target: %s, ', sstarg.status);
    if ~isequal(sstarg.status, 'Solve_Succeeded')
         fprintf('\n');
         warning('sstarg failed at time %d!', t);
         xtarg = xs;
         utarg = us;
    else
    xtarg = sstarg.var.x; 
    utarg = sstarg.var.u;
    end
    % xtarg(:,i)
    % utarg(:,i)

    % Compute optimal control.
    controller.fixvar('x', 1, xhat(:, t));
    controller.par.xs = repmat(xtarg, 1, Ncontroller + 1);
    controller.par.us = repmat(utarg, 1, Ncontroller);
    controller.par.d = dhat(:, t);
    controller.par.uprev = uprev;
    controller.solve();
    fprintf('Controller: %s, ', controller.status);
    if ~isequal(controller.status, 'Solve_Succeeded')
        fprintf('\n');
        warning('controller failed at time %d', t);
        break
    end
    
    ucl(:, t) = controller.var.u(:, 1);
    controller.saveguess();
    uprev = ucl(:, t); % Save previous u.
    
    
    
    % Evolve plant. Our variables are deviation but cstrsim needs positional.
    next_plant_state = full(dae_integrator('x', xcl(:, t), 'z', zcl(:, t), ...
                        'u', ucl(:, t), 'p', p(:, t)));
    
    % Compute the next internal and the algebraic state.                 
    xcl(:, t + 1) = full(next_plant_state.xplus);
    zcl(:, t + 1) = full(next_plant_state.zplus);

    
end

%% Stage Cost (for the regulator)
function l = stagecost(x, u, xs, usp, Du, Q, R, S)

  l = (x-xs)'*Q*(x-xs)+(u-usp)'*R*(u-usp) + Du'*S*Du;  
  
end%function

% Prior cost
function cost = priorcost(x, xhat, d, dhat, Pinv)
    z = [x; d];
    zhat = [xhat; dhat];
    dz = z - zhat;
    cost = dz'*Pinv*dz;
end%function
