function [x] = NE_multiplier(x,NS)
%% ==================================================================================================================== %
% Normal Equations Operator:
% --------------------------------------------------------------------------------------------------------------------- %
% [x] = NE_multiplier(NS,x) takes as an input the struct containing the Newton blocks, as well as 
% a vector of size m, and returns the matrix-vector product of the normal equations' matrix by this vector.
% _____________________________________________________________________________________________________________________ %
    w = NS.A_tr*x;
    w = (1./(NS.ThetaInv+spdiags(NS.Q,0))).*w;  
    w = NS.A*w;
    x = w + NS.delta.*x;
end

