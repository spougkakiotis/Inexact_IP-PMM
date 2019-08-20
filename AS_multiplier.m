function [w] = AS_multiplier(x,NS)
%% ==================================================================================================================== %
% Augmented System Operator:
% --------------------------------------------------------------------------------------------------------------------- %
% [x] = AS_multiplier(NS,x) takes as an input the struct containing the Newton blocks, as well as 
% a vector of size m, and returns the matrix-vector product of the augmented system's matrix by this vector.
% _____________________________________________________________________________________________________________________ %
    w = zeros(NS.n+NS.m,1);
    w(1:NS.n) = -NS.ThetaInv.*x(1:NS.n) - NS.Q*x(1:NS.n) + NS.A_tr*x(NS.n+1:NS.n+NS.m);
    w(NS.n+1:NS.n+NS.m) = NS.A*x(1:NS.n) + NS.delta .* x(NS.n+1:NS.n+NS.m);
end

