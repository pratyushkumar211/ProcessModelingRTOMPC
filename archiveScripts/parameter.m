% Container class for parameters. 
% Stores the value and the unit of the parameter. 
classdef parameter < handle

    %
properties (GetAccess = public, SetAccess = public)
    value = struct(); % Contains A, B, Q, R, M, Pf.
    unit = 'NA'; % MPC problem setup, dense, structured, DP.
end%properties

methods (Access=public)
    
    % Constructor class. 
    function self = parameter(varargin)
        
        
        % Declare a persistent variable
        persistent parser
        if isempty(parser)
            
            parser = ArgumentParser();
            % Dynamic Matrices
            parser.add('value', 'required');
            parser.add('unit', 'required');
            
        end
        
        args = parser.parse(varargin{:});
        
        % Create the 
        self.value = args.value;
        self.unit = args.unit;        
        
    end
    
end

end

