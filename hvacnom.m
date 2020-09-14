% PK: 11/14/2018
% [depends] hvacnndata.mat
% [makes] mat

% preamble
clear 
close all
randn('state', 12);

% Import MPC tools
mpc = import_mpctools();
custompath();

% Parameters and sizes for the nonlinear system
Delta = 0.5;
Nx = 4;
Nu = 2;
Ny = Nx;
Np = 1;
Nd = 4;

load('hvacnndata.mat')
% Loading continious time matrices

% Stage Cost (for the regulator)
function l = stagecost(x, u, xsp, usp, Du)

  Q = (1e-4)*diag([1, 1, 1, 1]);
  R = (1e-4)*diag([1, 1]);
  S = 1e-2;
  l = (x-xsp)'*Q*(x-xsp)+(u-usp)'*R*(u-usp) + Du'*S*Du;  
  
end%function

%% Solving a MPC trajectory
Ntr = 350;
Ncontroller = 10;
Nmhe = 10;
Ncollocation = 10;

% Constraints
xlb = xs - 10*ones(Nx,1);
xub = xs + 40*ones(Nx,1);
ulb = zeros(Nu,1);
uub = ones(Nu,1);

% Set points
CVs = [1, 3];
xsp = repmat(xs, 1, Ncontroller + 1);
usp = repmat(us, 1, Ncontroller);
ysp = NaN(Ny, Ntr + 1);
ys = xs;
ysp(CVs, :) = repmat(ys(CVs), 1, Ntr + 1);
ysp(CVs, 101:200) = repmat([295;288], 1, 100);
ysp(CVs, 201:end) = repmat([310;300], 1, 151);

% Disturbance and setpoint.
p = repmat(ps,1, Ntr);
p(:, 21:end) = ps - 2;

% Structs for the controller
lb = struct('x', xlb, 'u', ulb, 'Du', -0.05);
ub = struct('x', xub, 'u', uub, 'Du', 0.05);

% MPC trajectory and actual plant trajectory
x = zeros(Nx, Ntr+1);
u = zeros(Nu, Ntr);
y = zeros(Nx, Ntr);
x(:,1) = xs + [1; 2; 1; -0.5];

% Nominal controller
% Plant
fxup = mpc.getCasadiIntegrator(@(x, u, p) A*x+B*((c1.*u)./(c2+u))+ Bp*p, ...
                                Delta, [Nx, Nu, Np], {'x', 'u', 'p'}, {'fxup'});
%c1
%c2                               
%full(fxup(xs, us, ps))
%xs

% Model
Bd = [0 0 0 0; 0 0 1 0; 0 0 0 0;0 0 0 1];
f = mpc.getCasadiFunc(@(x, u, p, d) A*x+B*((c1.*u)./(c2+u)) + Bp*p + Bd*d, ...
                   [Nx, Nu, Np, Nd], {'x', 'u', 'p', 'd'}, {'f'});

fs = mpc.getCasadiFunc(@(x, u, p, d) A*x+B*((c1.*u)./(c2+u)) + Bp*p + Bd*d + x, ...
                   [Nx, Nu, Np, Nd], {'x', 'u', 'p', 'd'}, {'f'});

% Grabbing stuff for MPC tools
N = struct('x', Nx, 'u', Nu, 'y', Ny, 't', Ncontroller, 'c', Ncollocation);
l = mpc.getCasadiFunc(@(x,u,xsp,usp, Du) stagecost(x, u, xsp, usp, Du),...
		       [N.x, N.u, N.x, N.u, N.u], {'x', 'u','xsp','usp', 'Du'}, {'l'});
Vf = mpc.getCasadiFunc(@(x, xsp) stagecost(x, zeros(Nu,1), xsp, zeros(Nu,1), zeros(Nu,1)),...
		       [N.x, N.x], {'x', 'xsp'}, {'Vf'});
controller = mpc.nmpc('f', f, 'N', N, 'x0', x(:,1), 'l', l, ...
		                 'Vf', Vf, 'lb', lb, 'ub', ub, 'uprev', us, 'Delta', Delta, ...
                     'par', struct('xsp', xsp, 'usp', usp, 'd', zeros(Nd,1), 'p', ps),...
		                 'guess', struct('x', xsp, 'u', usp));

% Assembling Target selector
Cd = [1 0 0 0; 0 0 0 0; 0 1 0 0;0 0 0 0];
N = struct('x', Nx, 'u', Nu, 'y', Ny);
guess = struct('x', xs, 'u', us, 'y', ys);
par = struct('d', zeros(Nd,1), 'ysp', ys, 'usp', us, 'p', ps);
hxd = mpc.getCasadiFunc(@(x, d) x + Cd*d, ...
                         [Nx, Nd], {'x', 'd'}, {'hxd'});
lstarg = mpc.getCasadiFunc(@(y, u, ysp, usp) (y(CVs)-ysp(CVs))'*(y(CVs)-ysp(CVs)), ...
                         [Ny, Nu, Ny, Nu], {'y', 'u', 'ysp', 'usp'}, {'lstarg'});
sstarg = mpc.sstarg('f', fs, 'h', hxd, 'l', lstarg, 'N', N, ...
                     'lb', lb, 'ub', ub, 'guess', guess, 'par', par);

% Assembling Moving Horizon Estimation
Qinv = diag([1, 1, 1, 1]);
Rinv = Qinv;
Qdinv = Qinv;
Pinv = blkdiag(Qinv, Qdinv);
% Stage cost
lmhe = mpc.getCasadiFunc(@(w, v, Dd) w'*Qinv*w + v'*Rinv*v + Dd'*Qdinv*Dd, ...
                          [Nx, Ny, Nd], {'w', 'v', 'Dd'}, {'lmhe'});
% Prior cost
function cost = priorcost(x, xhat, d, dhat, Pinv)
    z = [x; d];
    zhat = [xhat; dhat];
    dz = z - zhat;
    cost = dz'*Pinv*dz;
end%function
%disp('hey1')
lx = mpc.getCasadiFunc(@(x,xhat, d, dhat) priorcost(x, xhat, d, dhat, Pinv), ...
                        [Nx, Nx, Nd, Nd], {'x', 'xbar', 'd', 'dbar'}, {'lx'});
N = struct('x', Nx, 'u', Nu, 'y', Ny, 'd', Nd, 't', Nmhe, 'c', Ncollocation);
guess = struct('x', repmat(xs, 1, N.t + 1), 'd', repmat(zeros(Nd,1),1, N.t+1));
lb = struct('x', xlb);
ub = struct('x', xub);
par = struct('y', repmat(ys, 1, N.t + 1), 'u', repmat(us, 1, N.t), ...
             'xbar', xs, 'dbar', zeros(Nd,1), 'p', ps);
mhe = mpc.nmhe('f', f, 'h', hxd, 'l', lmhe, 'lx', lx, 'N', N, 'Delta', Delta, ...
               'guess', guess, 'lb', lb, 'ub', ub, 'par', par, ...
               'wadditive', true());

uprev = us;
xprior = repmat(xs, 1, Nmhe);
dprior = repmat(zeros(Nd,1), 1, Nmhe);

% xtarg = zeros(Nx, Ncontroller + 1);
% utarg = zeros(Nu, Ncontroller + 1);

% Starting the trajectory
for i=1:Ntr
    i
    y(:,i) = x(:,i) + mvnrnd(zeros(1,Ny),mn)';
    
    % MHE 
    mhe.newmeasurement(y(:,i), uprev);
    mhe.par.xbar = xprior(:,1);
    mhe.par.dbar = dprior(:,1);
    mhe.solve();
    
    fprintf('Estimator: %s, \n', mhe.status);
    if ~isequal(mhe.status, 'Solve_Succeeded')
        fprintf('\n');
        warning('mhe failed at time %d!', i);
        break
    end
    
    %mhe.var.x
    %mhe.var.d
    %mhe.var.w
    %mhe.var.v
    %mhe.var.Dd

    xhat(:,i) = mhe.var.x(:,end);
    dhat(:,i) = mhe.var.d(:,end);

    %xhat(:,i)
    %dhat(:,i)
    %x(:,i)

    % Checking disturbances    
    % mhe.var.d(:,end)
    % mhe.var.Dd(:,end)+mhe.var.d(:,end-1)

    % Checking state equations
    % mhe.var.x(:,end)
    % mhe.var.w(:,end)+full(f(mhe.var.x(:,end-1),uprev,ps, mhe.var.d(:,end-1)))

    % Checking measurement equations
    % y(:,i)
    % mhe.var.x(:,end)+Cd*mhe.var.d(:,end)+mhe.var.v(:,end)    

    mhe.saveguess();
    xprior = [xprior(:,2:end), xhat(:,i)];
    dprior = [dprior(:,2:end), dhat(:,i)];
    
    %xhat(:,i)
    %dhat(:,i)
    % Computing all steady state targets for future horizon
    % assuming the disturbance estimate is dhat
    %sstarg.fixvar('y', 1, ysp(CVs,i), CVs);
    sstarg.par.ysp = ysp(:,i);
    sstarg.par.d = dhat(:,i);
    sstarg.solve();
    fprintf('Target: %s, ', sstarg.status);
    if ~isequal(sstarg.status, 'Solve_Succeeded')
         fprintf('\n');
         warning('sstarg failed at time %d!', i);
         xtarg = xs;
         utarg = us;
    else
      xtarg = sstarg.var.x;
      utarg = sstarg.var.u;
      %full(f(xtarg, utarg, ps, dhat(:,i)))
      % xtarg
      % utarg
    end
      
    %sstarg.var.x
    %sstarg.var.u
    %xtarg(:,i)

    % Compute optimal control.
    controller.fixvar('x', 1, xhat(:,i));
    controller.par.xsp = repmat(xtarg, 1, Ncontroller+1);
    controller.par.usp = repmat(utarg, 1, Ncontroller);
    controller.par.d = dhat(:,i);
    controller.par.uprev = uprev;
    controller.solve();
    
    fprintf('Controller: %s, ', controller.status);
    if ~isequal(controller.status, 'Solve_Succeeded')
        fprintf('\n');
        warning('controller failed at time %d', i);
        break
    end
    
    u(:, i) = controller.var.u(:,1);
    controller.saveguess();
    uprev = u(:,i);
    
    % Evolve plant.
    x(:, i + 1) = full(fxup(x(:,i), u(:,i), p(:,i)));
    
    fprintf('\n');

end

%u
%y

rank([eye(Nx)-A, -Bd;eye(Nx), Cd])

ti=0:Delta:Delta*(Ntr-1);
xsp = repmat(xsp(:,1), 1, Ntr+1);
xsp(CVs, :) = ysp(CVs, :);

% Saving the data set
save('-v7', 'hvacnom.mat','x','u', 'y', 'ysp', 'ti', 'Ntr');