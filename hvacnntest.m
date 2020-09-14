% PK: 11/14/2018
% [depends] hvacnntrain.mat
% [makes] mat

% preamble
close all
clear
randn('state',12);

% Load training data, this contains the parameters in the network.
nndata = load('hvacnntrain.mat');

% Load some information about system which is being controlled
load('hvacnndata.mat');

% Import MPC tools
mpc = import_mpctools();
custompath();

% Parameters and sizes for the nonlinear system
[Nx, Nu] = size(B);
Np = size(Bp, 2);
Nd = Nx;

% Plant
fxup = mpc.getCasadiIntegrator(@(x, u, p) A*x+B*((c1.*u)./(c2+u))+ Bp*p, ...
                                Delta, [Nx, Nu, Np], {'x', 'u', 'p'}, {'fxup'});

%%%%%%%%% Compare Model predictions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                
% Total number of time steps for visual predictions
Nt = 500;

% Variables to store the state trajectories
x = zeros(Nx, Nt + 1);
xlin = zeros(Nx, Nt + 1);
xnnp = zeros(Nx, Nt +1);
u = zeros(Nu, Nt);

x(:,1) = xs;
xlin(:,1) = xs;
xnnp(:,1) = xs;

u(1,:) = rand([1,Nt]);
u(2,:) = rand([1,Nt]);

% For loop to observe the plant and predicted output
for i = 1:Nt
  
  % Plant
  x(:, i+1) = full(fxup(x(:,i), u(:,i), ps));
  
  % Neural Network
  xnnp(:,i+1) = forward_propagation(struct('x', [xnnp(:,i);u(:,i)], ...
                 'data', nndata));  
                 
  % Linear Model
  xlin(:,i+1) = Alin*(xlin(:,i)-xs)+ Blin*(u(:,i)-us)+xs;

end 

tpred = 0:Delta:Delta*(Nt-1);
disp('Done Visualizing model predictions')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% MPC tools stuff %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Three models with same disturbance model
Bd = [0 0 0 0; 0 0 1 0; 0 0 0 0;0 0 0 1];
Cd = [1 0 0 0; 0 0 0 0; 0 1 0 0;0 0 0 0];

% Nominal 
Fnom = mpc.getCasadiFunc(@(x, u, p, d) A*x+B*((c1.*u)./(c2+u)) + Bp*p + Bd*d, ...
                          [Nx, Nu, Np, Nd], {'x', 'u', 'p', 'd'}, {'Fnom'}, ...
                          'rk4', true(), 'Delta', Delta, 'M', 10);

% Linear                          
Flin = mpc.getCasadiFunc(@(x, u, d) Alin*(x-xs)+Blin*(u-us) + Bd*d + xs, ...
                   [Nx, Nu, Nd], {'x', 'u', 'd'}, {'Flin'});

% Neural network                   
Fnn = mpc.getCasadiFunc(@(x, u, d) forward_propagation(struct('x', [x;u], 'data', nndata)) + Bd*d, ...
                   [Nx, Nu, Nd], {'x', 'u', 'd'}, {'Fnn'});


% Stage Cost (for the regulator)
function l = stagecost(x, u, xsp, usp, Du)

  Q = (1e-4)*diag([1, 1, 1, 1]);
  R = (1e-4)*diag([1, 1]);
  S = 1e-2;
  l = (x-xsp)'*Q*(x-xsp)+(u-usp)'*R*(u-usp) + Du'*S*Du;  
  
end%function

%% Trajectory lengths
Ntr = 350; 
Ncontroller = 10;
Nmhe = 10;

% Structs for the controller
lb = struct('x', xlb, 'u', ulb, 'Du', -0.05);
ub = struct('x', xub, 'u', uub, 'Du', 0.05);

% Struct to store closed-loop trajectories
xcl = struct();
ycl = struct();
ucl = struct();

% Regulator
N = struct('x', Nx, 'u', Nu, 't', Ncontroller);
l = mpc.getCasadiFunc(@(x, u, xsp, usp, Du) stagecost(x, u, xsp, usp, Du),...
           [Nx, Nu, Nx, Nu, Nu], {'x', 'u','xsp','usp', 'Du'}, {'l'});
Vf = mpc.getCasadiFunc(@(x, xsp) stagecost(x, zeros(Nu,1), xsp, zeros(Nu,1), zeros(Nu,1)),...
           [Nx, Nx], {'x', 'xsp'}, {'Vf'});
% Parameters for the controllers
pars = struct();
pars.d = zeros(Nd, 1);
pars.xsp = repmat(xs, 1, Ncontroller + 1);
pars.usp = repmat(us, 1, Ncontroller);
pars.p = ps; % Only used by the nominal controller
% Initial Guess
guess = struct();
guess.x = pars.xsp;
guess.u = pars.usp;
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
controllers = struct();
controllers.nom = mpc.nmpc('f', Fnom, '**', kwargs);
controllers.lin = mpc.nmpc('f', Flin, '**', kwargs);
controllers.nn = mpc.nmpc('f', Fnn, '**', kwargs);


% Assembling Target selectors
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
function cost = priorcost(x, xhat, d, dhat, Pinv)
    z = [x; d];
    zhat = [xhat; dhat];
    dz = z - zhat;
    cost = dz'*Pinv*dz;
end%function
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
estimators = struct();
estimators.nom = mpc.nmhe('f', Fnom, '**', kwargs);
estimators.lin = mpc.nmhe('f', Flin, '**', kwargs);
estimators.nn = mpc.nmhe('f', Fnn, '**', kwargs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

solvers = fieldnames(controllers);
for j = 1:length(solvers)
  
  s = solvers{j};
  
  % Extracting controller, target selector, and estimator
  controller = controllers.(s);
  sstarg = sstargs.(s);
  estimator = estimators.(s);

  % Priors and state estimates
  xprior = repmat(xs, 1, Nmhe);
  dprior = repmat(zeros(Nd,1), 1, Nmhe);
  xhat = zeros(Nx, Ntr + 1);
  dhat = zeros(Nd, Ntr + 1);

  % Previous u
  uprev = us;
  controller.par.uprev = uprev;
  
  % Variables to store trajectories
  xcl.(s) = zeros(Nx, Ntr + 1);
  xcl.(s)(:, 1) = xs + [1; 2; 1; -0.5];
  ucl.(s) = zeros(Nu, Ntr);
  ycl.(s) = zeros(Nx, Ntr + 1);

  
  for t = 1:Ntr

    fprintf('Model: %s, Time: %d \n', s, t);
    ycl.(s)(:, t) = xcl.(s)(:, t) + mvnrnd(zeros(1, Nx), Rv)';
    
    % MHE 
    estimator.newmeasurement(ycl.(s)(:, t), uprev);
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
    controller.par.xsp = repmat(xtarg, 1, Ncontroller + 1);
    controller.par.usp = repmat(utarg, 1, Ncontroller);
    controller.par.d = dhat(:, t);
    controller.par.uprev = uprev;
    controller.solve();
    fprintf('Controller: %s, ', controller.status);
    if ~isequal(controller.status, 'Solve_Succeeded')
        fprintf('\n');
        warning('controller failed at time %d', t);
        break
    end
    
    ucl.(s)(:, t) = controller.var.u(:, 1);
    controller.saveguess();
    uprev = ucl.(s)(:, t); % Save previous u.
    
    % Evolve plant.
    xcl.(s)(:, t + 1) = full(fxup(xcl.(s)(:, t), ucl.(s)(:, t), p(:, t)));

    fprintf('\n');

  end

end

% Just recreating these (used for plotting)
ti = 0:Delta:Delta*(Ntr-1);

% Saving the data set
save('-v7', 'hvacnntest.mat','x','xlin','xnnp', 'u', 'tpred', ...
    'ycl', 'ucl', 'ti', ... 
    'Ntr', 'ysp');
