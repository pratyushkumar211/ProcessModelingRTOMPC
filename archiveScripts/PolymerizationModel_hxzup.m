% The polymerization ODE model.
% x: state, u: control input, p:plant disturbances.
% pars: The parameter struct.
function y = PolymerizationModel_hxzup(x, z, u, p)


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
%% Compute the outputs. 
pars = get_parameters(T_reactor, M1, M2); % computes all the necessary parameters for a given k.

% Extract out necessary parameters. 
c1 = pars.c1;
c2 = pars.c2;
c3 = pars.c3;
c4 = pars.c4;
c5 = pars.c5;


% 13) The polymer density.
phop = c3 - c4*(100*(B2/(B1 + B2)))^c5;


% 1) Compute the total pressure. 
gas_constant = pars.gas_constant;
MT = (M1 + M2 + In + H2 + Co + Im); % mol/m^3
P = MT*gas_constant*T_reactor; % N/m^2

% The gas phase mole fractions
MT = M1 + M2 + In + H2 + Co + Im;
XM1 = M1/MT;
XM2 = M2/MT;
XIn = In/MT;
XH2 = H2/MT;
XCo = Co/MT;

% 2) The bleed flow in mol/s
bleed_coeff = pars.bleed_coeff;
vent_pressure = pars.vent_pressure;
bT = bleed_valvep*bleed_coeff*sqrt(P - vent_pressure); % Total bleed rate.
bleed_flow = bT/MT; % m^3/s

% 3) The production rate
production_rate = Rv*phop; % m^3/sec to kg/m^3 

% 12) The melt index.
MW1 = pars.MW1;
MW2 = pars.MW2;
mbar = (MW1*B1 + MW2*B2)/(B1 + B2);
MWbar = mbar*(X2(1) + Y2(1) + X2(2) + Y2(2))/(X1(1) + Y1(1) + X1(2) + Y1(2));
melt_index = (MWbar/c1)^c2;


% The final measurement vector%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = [P;bleed_flow;production_rate;XM1;XM2;XIn;XH2;XCo;
     T_reactor;Tw_exchanger;T_recycle;melt_index;phop];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
return