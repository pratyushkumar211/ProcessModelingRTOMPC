% Generates relevant data to train neural network
% and subsequently do offset free MPC
% [makes] mat
clear
mpc = import_mpctools();
rng('default');

%%%%%%%%% Loading continuous time matrices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('buildingmats.mat')
[Nx, Nu] = size(B);
Bp = [0;0.5;0;0.25]; % Matrix that determines how the ambient temperature
Np = size(Bp, 2);
B = B*(1e+5);
Delta = 0.5;
Rv = diag([1e-2 1e-2 1e-2 1e-2]);
Nt = 5000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Casadi functions for the plant  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
c2 = [0.03; 0.05];
c1 = [0.385; 0.79];
fxup = mpc.getCasadiIntegrator(@(x,u,p) A*x+B*((c1.*u)./(c2+u))+ Bp*p, ...
                           Delta, [Nx, Nu, Np], {'x', 'u', 'p'}, {'fxup'});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% One particular steady state %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Would like to maintain the air temperature zones and the first 
% valve position
CVs = [1, 3, 5]; 
n = null([A, B, Bp]);
TsQsTas = n*(n(CVs, :)\[290; 290; 0.36]);
xs = TsQsTas(1:4);
ps = TsQsTas(7);
Qs = TsQsTas(5:6);
us = (Qs.*c2)./(c1-Qs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[A, B, Bp]*[xs;Qs;ps]
%full(fxup(xs,us,ps))
%xs

%%%%%%%%% Collect open loop data-set for linear ID %%%%%%%%%%%%%%%%%%%%%%%%
% Variables to store data for the nonlinear system
x = zeros(Nx, Nt+1);
x(:,1) = xs;
u = zeros(Nu, Nt);
y(:,1) = x(:,1) + mvnrnd(zeros(1,Nx), Rv)';

% PRBS input
exn = 0.2;
% Need to perturb the plant only in a small region around the steady state
u(1,:) = us(1) + exn*us(1)*(2*rand([1,Nt])- 1);
u(2,:) = us(2) + exn*us(2)*(2*rand([1,Nt])- 1);

% Loop over timesteps
for i=1:Nt

    % Plant
    x(:, i+1) = full(fxup(x(:,i), u(:,i), ps));
    y(:, i+1) = x(:,i+1) + mvnrnd(zeros(1,Nx), Rv)';
    
end	

time = 0:Delta:Delta*(Nt-1);
y = y - xs;
u = u - us;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%max(u, [], 2)
%min(u, [], 2)
%max(y, [], 2)
%min(y, [], 2)
%y = y - xs;
%u = u - us;

%%% Using the entire data to estimate a linear model %%%%%%%%%%%%%%%%%%%%%%
iddata = iddata(y(:,1:end-1)',u(:,1:end)', Delta);
parameters = {'A',zeros(Nx,Nx);'B',zeros(Nx,Nu)};
fcn_type = 'd';         
init_sys = idgrey(@myGreyModel,parameters, fcn_type);         
opt = greyestOptions('InitialState', zeros(Nx,1));         
estsys = greyest(iddata, init_sys,opt);
Ahat = estsys.A;
Bhat = estsys.B;
estsys.Report.Fit.MSE;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Data set for neural network training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x(:,1) = xs;
y(:,1) = x(:,1) + mvnrnd(zeros(1,Nx), Rv)';

% PRBS input, valve positions in the range 0 to 1
u(1,:) = rand([1,Nt]);
u(2,:) = rand([1,Nt]);

% Loop over timesteps
for i=1:Nt

    % Plant
    x(:,i+1) = full(fxup(x(:,i),u(:,i), ps));
    y(:,i+1) = x(:,i+1) + mvnrnd(zeros(1,Nx), Rv)';
    
end	

xnn = [y(:,1:end-1);u];
ynn = y(:,2:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Also get the analytically linearlized model %%%%%%%%%%%%%%%%%%%%%%%%%%%
[Alin, Blin, Bplin] = mpc.getLinearizedModel(fxup, {xs, us, ps}, ...
                        {'A', 'B', 'Bp'}, 'deal', true());
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Compute the data required to plot the steady-state curve %%%%%%%%%%
% Computing gains x vs u
uvar = 0:0.005:1;
% Two cases:
% a) Vary u1, keep u2 constant
% b) Vary u2, keep u1 constant
xslin = zeros(Nx, size(uvar,2), 2);
xsnonlin = zeros(Nx, size(uvar,2), 2);

for k = 1:size(uvar,2)
    
    % Linear
    % Computes steady-states in deviation
    xslin(:, k, 1) = ((eye(Nx)-Alin)\Blin)*[uvar(k)-us(1);0] + xs;
    xslin(:, k, 2) = ((eye(Nx)-Alin)\Blin)*[0;uvar(k)-us(2)] + xs;
    
    % Nonlinear -- Initialize it with the linear solution
    xsnonlin(:, k, 1) = xslin(:, k, 1);
    xsnonlin(:, k, 2) = xslin(:, k, 2);
    
    % Iterate a certain number of times so that we find the steady
    % state from the nonlinear equation.
    for t = 1:500
        
    xsnonlin(:, k, 1) = full(fxup(xsnonlin(:, k, 1),[uvar(k);us(2)],ps));
    xsnonlin(:, k, 2) = full(fxup(xsnonlin(:, k, 2),[us(1);uvar(k)],ps));
    
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Set point changes and disturbances for subsequent simulations %%%%%%
Ntr = 350;
CVs = [1, 3];
ysp1 = [295;288];
ysp2 = [310;300];
ysp = NaN(Nx, Ntr + 1);
ysp(CVs, :) = repmat(xs(CVs), 1, Ntr + 1);
ysp(CVs, 101:200) = repmat(ysp1, 1, 100); 
ysp(CVs, 201:end) = repmat(ysp2, 1, 151);

% Disturbance and setpoint.
p = repmat(ps, 1, Ntr);
p(:, 21:end) = ps - 2;

% Constraints
xlb = xs - 10*ones(Nx,1);
xub = xs + 40*ones(Nx,1);
ulb = zeros(Nu,1);
uub = ones(Nu,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save data.
save('-v7', 'hvacnndata.mat','u', 'A', 'B', 'Bp', 'Ahat', 'Bhat', 'time', ...
     'xnn', 'ynn', 'y', 'xs', 'ps', 'Qs', 'us', 'Rv', 'Alin', 'Blin', ...
     'Bplin', 'xslin', 'xsnonlin', 'uvar', 'CVs', ...
     'c1', 'c2', 'Delta', 'ysp', 'p', 'ysp1', 'ysp2', ...
     'xlb', 'xub', 'ulb', 'uub');

save('-v7', 'hvacmodelnonlin.mat', 'A', 'B', 'Bp', 'xs', 'us', 'ps', 'c1', 'c2'); 
% Function required to plot the  steady-state curve
function [A, B, C, D] = myGreyModel(A, B, Ts)

C=eye(size(A,1));
D=zeros(size(A,1),size(B,2));

end