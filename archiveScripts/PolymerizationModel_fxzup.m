% The polymerization ODE model
% Pratyush Kumar, pratyushkumar@ucsb.edu
% x: Differential States
% z: Algebraic States
% u: Control Input
% p: Plant disturbances
function dxbydt = PolymerizationModel_fxzup(x, z, u, p)


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
%% Gas phase concentration Balance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract the necessary parameters. 
pars = get_parameters(T_reactor, M1, M2); % computes all the necessary parameters for a given k.
bleed_coeff = pars.bleed_coeff; % Unitless
gas_constant = pars.gas_constant; % (N-m/K-mol)
V_total = pars.V_total; % m^3
vent_pressure = pars.vent_pressure; % N/m^2

MCp_wall = pars.MCp_wall; 
Cp_polymer = pars.Cp_polymer;

M_cwater = pars.M_cwater;
Cp_water = pars.Cp_water;
F_cwater = pars.F_cwater;
UA = pars.UA;
M_holdup_recycle = pars.M_holdup_recycle;

MW1 = pars.MW1;
MW2 = pars.MW2;

T_reference = pars.T_reference;
Hreac = pars.Hreac;

%% The rate constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rate Constants

kf = pars.kf;
ki1 = pars.ki1;
ki2 = pars.ki2;
kh1 = pars.kh1;
kh2 = pars.kh2;
kiT = pars.kiT ;
khT = pars.khT;
kp1T = pars.kp1T;
kp2T = pars.kp2T;
kpT1 = pars.kpT1; % Done
kpT2 = pars.kpT2; % Done
kpTT = pars.kpTT;
kfm1T = pars.kfm1T;
kfm2T = pars.kfm2T;
kfmT1 = pars.kfmT1;
kfmT2 = pars.kfmT2;
kfmTT = pars.kfmTT;
kfhT = pars.kfhT;
kfrT = pars.kfrT;
kfsT = pars.kfsT;

khr = pars.khr;
kds = pars.kds;
kdI = pars.kdI;
ka = pars.ka;

k1star = pars.k1star;
k2star = pars.k2star;
kCostar = pars.kCostar;

c3 = pars.c3;
c4 = pars.c4;
c5 = pars.c5;
phoc = pars.phoc;
phoa = pars.phoa;

Cp_M1 = pars.Cp_M1;
Cp_M2 = pars.Cp_M2;
Cp_In = pars.Cp_In;
Cp_H2 = pars.Cp_H2;

% The volumes of the polymer and the Gas phase.
Vp = pars.Vp;
Vg = pars.Vg; 

%% Start the calculations. 

%%%%%%%%% Compute reactor volumes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Bw = MW1*B1 + MW2*B2;
phop = c3 - c4*(100*(B2/(B1 + B2)))^c5;
alphav = (phoc - phop)/(phoc - phoa);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MT = (M1 + M2 + In + H2 + Co + Im); % mol/m^3
P = MT*gas_constant*T_reactor; % N/m^2
bT = bleed_valvep*bleed_coeff*sqrt(max(P - vent_pressure, 0)); % Total bleed rate.


% Bleed rates
bM1 = (M1/MT)*bT;
bM2 = (M2/MT)*bT;
bIn = (In/MT)*bT;
bH2 = (H2/MT)*bT;
bCo = (Co/MT)*bT;
bIm = (Im/MT)*bT;

% The Reaction Rates.
RM1 = M1*(kpT1(1)*Y0(1) + kpT1(2)*Y0(2));
RM2 = M2*(kpT2(1)*Y0(1) + kpT2(2)*Y0(2));
RCo = Co*(kfrT(1)*Y0(1) + khr(1)*NH0(1) + kfrT(2)*Y0(2) + khr(2)*NH0(2));
RIm = Im*(kdI(1)*(Y0(1) + N0(1) + NH0(1)) - ka(1)*(NdI0(1) + NdIH0(1)));
RIm = RIm + Im*(kdI(2)*(Y0(2) + N0(2) + NH0(2)) - ka(2)*(NdI0(2) + NdIH0(2)));
RH2 = H2*(kfhT(1)*Y0(1) + kfhT(2)*Y0(2));

% Sorption of ethylene, 1-butene, and co-catalyst. 
SM1 = alphav*k1star*M1*Rv;
SM2 = alphav*k2star*M2*Rv;
SCo = alphav*kCostar*Co*Rv;



%% Monomer balances %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Balance on the monomer bounds
dB1bydt = RM1 - B1*(Rv/Vp);
dB2bydt = RM2 - B2*(Rv/Vp);

%% Gas phase balances %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dM1bydt
dM1gbydt = (FM1 - bM1 - RM1 - SM1)/(Vg + (alphav*k1star*Vp*P/MT));

% dM2gbydt
dM2gbydt = (FM2 - bM2 - RM2 - SM2)/(Vg + (alphav*k2star*Vp*P/MT));

% dInbydt
dInbydt = (FIn - bIn)/Vg;

% dH2bydt
dH2bydt = (FH2 - bH2 - RH2)/Vg;

% dCobydt
dCobydt = (FCo - bCo - RCo - SCo)/(Vg + alphav*kCostar);

% dImbydt
dImbydt = (FIm - bIm - RIm)/Vg;


%% The heat balances %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cp_recycle = (M1*Cp_M1 + M2*Cp_M2 + In*Cp_In + H2*Cp_H2)/MT;

% Assume that heat is only carried by monomer, comonomer, inert, and H2. 
H_feed = (FM1*Cp_M1 + FM2*Cp_M2 + FIn*Cp_In + FH2*Cp_H2)*(T_feed - T_reference);

H_recycle_in = Frecycle*Cp_recycle*(T_recycle - T_reference);
H_recycle_out = (Frecycle + bT)*Cp_recycle*(T_reactor - T_reference);

H_reaction = Hreac*(MW1*RM1 + MW2*RM2);
H_polymer = Rv*phop*Cp_polymer;

dTreactorbydt = (H_feed + H_recycle_in - H_recycle_out - H_reaction - H_polymer)/(MCp_wall + Bw*Cp_polymer);
dTwexchangerbydt = (F_cwater/M_cwater)*(T_cwater - Tw_exchanger) + (UA/(M_cwater*Cp_water))*(T_recycle - Tw_exchanger);
dTrecyclebydt = (Frecycle/M_holdup_recycle)*(T_reactor - T_recycle) + (UA/(M_holdup_recycle*Cp_recycle))*(Tw_exchanger - T_recycle);


%% Chain Balances %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Balances on living chains.
dY0bydt = casadi.SX.sym('dY0bydt', 1, 2);
dY1bydt = casadi.SX.sym('dY1bydt', 1, 2);
dY2bydt = casadi.SX.sym('dY2bydt', 1, 2);
dX1bydt = casadi.SX.sym('dX1bydt', 1, 2);
dX2bydt = casadi.SX.sym('dX2bydt', 1, 2);

% Loop over the sites, this is probably a bug free way. 
for j = 1:2
    
    % Zeroth Moment.
    dY0bydt(j) = MT*(kiT(j)*N0(j) + khT(j)*NH0(j)) + khr(j)*NH0(j)*Co; 
    dY0bydt(j) = dY0bydt(j) - Y0(j)*(kfhT(j)*H2 + kfsT(j) + kds(j) + kdI(j)*Im + (Rv/Vp));

    % First Moment. 
    dY1bydt(j) = MT*(kiT(j)*N0(j) + khT(j)*NH0(j)) + khr(j)*NH0(j)*Co + MT*kpTT(j)*Y0(j);
    dY1bydt(j) = dY1bydt(j) + (Y0(j) - Y1(j))*(kfmTT(j)*MT + kfrT(j)*Co);
    dY1bydt(j) = dY1bydt(j) - Y1(j)*(kfhT(j)*H2 + kfsT(j) + kds(j) + kdI(j)*Im + (Rv/Vp));

    % Second Moment. 
    dY2bydt(j) = MT*(kiT(j)*N0(j) + khT(j)*NH0(j)) + khr(j)*NH0(j)*Co; 
    dY2bydt(j) = dY2bydt(j) + MT*kpTT(j)*(2*Y1(j) - Y0(j));
    dY2bydt(j) = dY2bydt(j) + (Y0(j) - Y2(j))*(kfmTT(j)*MT + kfrT(j)*Co);
    dY2bydt(j) = dY2bydt(j) - Y2(j)*(kfhT(j)*H2 + kfsT(j) + kds(j) + kdI(j)*Im + (Rv/Vp));


    % Balances on dead chains
    dX1bydt(j) = (Y1(j)-NT1(j))*(kfmTT(j)*MT + kfrT(j)*Co + kfhT(j)*H2 + kfsT(j) + kds(j) + kdI(j)*Im);
    dX1bydt(j) = dX1bydt(j) - X1(j)*(Rv/Vp);

    dX2bydt(j) = (Y2(j)-NT1(j))*(kfmTT(j)*MT + kfrT(j)*Co + kfhT(j)*H2 + kfsT(j) + kds(j) + kdI(j)*Im);
    dX2bydt(j) = dX2bydt(j) - X2(j)*(Rv/Vp);
    
end



%%%%%%%%% The final derivative vector %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dxbydt = [dM1gbydt; dM2gbydt; dInbydt; dH2bydt; dCobydt; dImbydt;
        dTreactorbydt; dTwexchangerbydt; dTrecyclebydt; dY0bydt(1); dY0bydt(2); 
        dY1bydt(1);dY1bydt(2); dY2bydt(1); dY2bydt(2); dX1bydt(1); dX1bydt(2); 
        dX2bydt(1);dX2bydt(2); dB1bydt; dB2bydt];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

return