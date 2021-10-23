% The polymerization ODE model.
% x: state, u: control input, p:plant disturbances.
% pars: The parameter struct.
function dzbydt = PolymerizationModel_gxzup(x, z, u, p)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract the Differential States
M1 = x(1); % Monomer concentration, mol/m^3
M2 = x(2); % Comonomer concentration, mol/m^3
In = x(3); % Inert concentration, mol/m^3
H2 = x(4); % Hydrogen concentration, mol/m^3
Co = x(5); % Cocatalyst concentration, mol/m^3
Im = x(6); % Impurity concentration, mol/m^3
T_reactor = x(7); % Reactor temperature, K
Tw_exchanger = x(8); % Cooling water temperature, K
T_recycle = x(9); % Recycle Gas temperature, K

% The moments of living polymers
Y0 = casadi.SX.sym('Y0', 1, 2);
Y1 = casadi.SX.sym('Y1', 1, 2);
Y2 = casadi.SX.sym('Y2', 1, 2);

Y0(1) = x(10); % 0th moment of living polymer at site 1
Y0(2) = x(11); % 0th moment of living polymer at site 2
Y1(1) = x(12); % 1st moment of living polymer at site 1
Y1(2) = x(13); % 1st moment of living polymer at site 2
Y2(1) = x(14); % 2nd moment of living polymer at site 1
Y2(2) = x(15); % 2nd moment of living polymer at site 2

% The moments of dead polymer chains 
X1 = casadi.SX.sym('X1', 1, 2);
X2 = casadi.SX.sym('X2', 1, 2);
X1(1) = x(16); % 1st moment of living polymer at site 1
X1(2) = x(17); % 1st moment of living polymer at site 2
X2(1) = x(18); % 2nd moment of living polymer at site 1
X2(2) = x(19); % 2nd moment of living polymer at site 2

% The moles of bounder monomer/comonomer
B1 = x(20); % moles of bound monomer
B2 = x(21); % moles of bound comonomer
%% Extract the Algebraic States
% All the units are in moles
Nstar = casadi.SX.sym('Nstar', 1, 2);
NdI0 = casadi.SX.sym('NdI0', 1, 2);
NdIH0 = casadi.SX.sym('NdIH0', 1, 2);
N0 = casadi.SX.sym('N0', 1, 2);
NH0 = casadi.SX.sym('NH0', 1, 2);
N11 = casadi.SX.sym('N11', 1, 2);
N21 = casadi.SX.sym('N21', 1, 2);

% Extracting out the algebraic states 
Nstar(1) = z(1); % moles
Nstar(2) = z(2); % moles
NdI0(1) = z(3); % moles
NdI0(2) = z(4); % moles
NdIH0(1) = z(5); % moles
NdIH0(2) = z(6); % moles
N0(1) = z(7); % moles
N0(2) = z(8); % moles
NH0(1) = z(9); % moles
NH0(2) = z(10); % moles
N11(1) = z(11); % moles
N11(2) = z(12); % moles
N21(1) = z(13); % moles
N21(2) = z(14); % moles

NT1 = casadi.SX.sym('NT1', 1, 2);
NT1(1) = N11(1) + N21(1);
NT1(2) = N11(2) + N21(2);

%% Extract the control inputs
FM1 = u(1); % Inlet flow rate of monomer. kg/s
FM2 = u(2); % Inlet flow rate of comonomer. kg/s
FIn = u(3); % Inlet flow rate of inert. kg/s
FH2 = u(4); % Inlet flow rate of hydrogen. kg/s
FCo = u(5); % Inlet flow rate of cocatalyst. kg/s
FIm = u(6); % Inlet flow rate of impurity. kg/s
Fcat = u(7); % Inlet flow rate of Catalyst. kg/s
Frecycle = u(8); % Gas recycle flow rate. kg/s
bleed_valvep = u(9); % Vent valve position. Unitless.
T_cwater = u(10); % Cooling water inlet temperature. K
Rv = u(11); % The polymer outflow. m^3/s.

% Extract the plant disturbances
% The disturbance is the feed temperature
T_feed = p; % K

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get the relevant parameters.
pars = get_parameters(T_reactor, M1, M2); % computes all the necessary parameters for a given k.
fraction_active = pars.fraction_active;

kf = pars.kf;
ki1 = pars.ki1;
ki2 = pars.ki2;
kh1 = pars.kh1;
kh2 = pars.kh2;
kiT = pars.kiT ;
khT = pars.khT;
kp1T = pars.kp1T;
kp2T = pars.kp2T;
kpT1 = pars.kpT1;
kpT2 = pars.kpT2; 
kpTT = pars.kpTT;
kfh1 = pars.kfh1;
kfh2 = pars.kfh2;
kfr1 = pars.kfr1;
kfr2 = pars.kfr2;
kfm1T = pars.kfm1T;
kfm2T = pars.kfm2T;
kfmT1 = pars.kfmT1;
kfmT2 = pars.kfmT2;
kfmTT = pars.kfmTT;
kfhT = pars.kfhT;
kfrT = pars.kfrT;
kfs1 = pars.kfs1;
kfs2 = pars.kfs2;
kfsT = pars.kfsT;

khr = pars.khr;
kds = pars.kds;
kdI = pars.kdI;
ka = pars.ka;

c3 = pars.c3;
c4 = pars.c4;
c5 = pars.c5;
MW1 = pars.MW1;
MW2 = pars.MW2;

%% Start the calculations. 

%%%%%%%%% Compute reactor volumes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Bw = MW1*B1 + MW2*B2;
phop = c3 - c4*(100*(B2/(B1 + B2)))^c5;
Vp = Bw/phop;
MT = M1 + M2 + In + H2 + Co + Im;

%% Now start the computations. 

% Create the arrays. 
dNstarbydt = casadi.SX.sym('dNstarbydt', 1, 2);
dN0bydt = casadi.SX.sym('dN0bydt', 1, 2);
dNH0bydt = casadi.SX.sym('dNH0bydt', 1, 2);
dN11bydt = casadi.SX.sym('dN11bydt', 1, 2);
dN21bydt = casadi.SX.sym('dN21bydt', 1, 2);
dNdIH0bydt = casadi.SX.sym('dNdIH0bydt', 1, 2);
dNdI0bydt = casadi.SX.sym('dNdI0bydt', 1, 2);

% The number of moles of active sites 
Fstarin = Fcat*fraction_active;

% Loop over the sites. 
for j = 1:2

% The balance on the number of moles of active sites. 
dNstarbydt(j) = Fstarin - kf(j)*Nstar(j) - Nstar(j)*(Rv/Vp);

% The balance for the number of moles of initiation sites. 
dN0bydt(j) = kf(j)*Nstar(j) + ka(j)*NdI0(j);
dN0bydt(j) = dN0bydt(j) - N0(j)*(kiT(j)*MT + kds(j) + kdI(j)*Im + (Rv/Vp));

% The balance on the number of moles of hydrogen initiated sites. 
dNH0bydt(j) = Y0(j)*(kfhT(j)*H2 + kfsT(j)) + ka(j)*NdIH0(j);
dNH0bydt(j) = dNH0bydt(j) - NH0(j)*(khT(j)*MT + kds(j) + khr(j)*Co + kdI(j)*Im + (Rv/Vp));

% Mass balance of polymer chain 
% With terminal monomer.
dN11bydt(j) = ki1(j)*N0(j)*M1 + NH0(j)*(kh1(j)*M1 + khr(j)*Co);
dN11bydt(j) = dN11bydt(j) + Y0(j)*(kfmT1(j)*M1 + kfrT(j)*Co);
dN11bydt(j) = dN11bydt(j) - N11(j)*(kp1T(j)*MT + kfm1T(j)*MT + kfh1(j)*H2);
dN11bydt(j) = dN11bydt(j) - N11(j)*(kfr1(j)*Co + kfs1(j) + kds(j) + kdI(j)*Im + (Rv/Vp));

% Mass balance of polymer chain 
% With terminal Comonomer.
dN21bydt(j) = ki2(j)*N0(j)*M2 + NH0(j)*kh2(j)*M2 + Y0(j)*kfmT2(j)*M2;
dN21bydt(j) = dN21bydt(j) - N21(j)*(kp2T(j)*MT + kfm2T(j)*MT + kfh2(j)*H2);
dN21bydt(j) = dN21bydt(j) - N21(j)*(kfr2(j)*Co + kfs2(j) + kds(j) + kdI(j)*Im + (Rv/Vp));

% Number of moles of impurity deactivated sites. 
dNdIH0bydt(j) = kdI(j)*Im*(Y0(j) + NH0(j)) - NdIH0(j)*(ka(j) + (Rv/Vp));
dNdI0bydt(j) = kdI(j)*Im*N0(j) - NdI0(j)*(ka(j) + (Rv/Vp));    
    
     
end

%%%%%%%%%%%%% dzbydt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dzbydt = [dNstarbydt(1); dNstarbydt(2); dNdI0bydt(1); dNdI0bydt(2)
          dNdIH0bydt(1); dNdIH0bydt(2); dN0bydt(1); dN0bydt(2); 
          dNH0bydt(1); dNH0bydt(2); dN11bydt(1); dN11bydt(2); 
          dN21bydt(1); dN21bydt(2)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      
      
return