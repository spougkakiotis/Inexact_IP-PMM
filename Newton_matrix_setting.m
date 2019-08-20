function [NS] = Newton_matrix_setting(iter,A,A_tr,Q,x,z,delta,rho,pos_vars,free_vars)
%% ==================================================================================================================== %
% Newton_matrix_setting: Store all relevant information about the Newton matrix.
% --------------------------------------------------------------------------------------------------------------------- %
% NS = Newton_matrix_setting(iter,A,A_tr,Q,x,z,delta,rho,pos_vars,free_vars) returns a MATLAB struct that holds the
%      relevant information of the Newton matrix, required for solving the step equations in
%      the IPM.
% Author: Spyridon Pougkakiotis.
% _____________________________________________________________________________________________________________________ %
[m, n] = size(A);
NS = struct();
%% ==================================================================================================================== %
% Store all the relevant KKT blocks on a struct called NS (Newton System).
% --------------------------------------------------------------------------------------------------------------------- %
    NS.x = x;
    NS.z = z;
    NS.A = A;
    NS.A_tr = A_tr;
    NS.Q = Q;
    NS.m = m;
    NS.n = n;
    R_ThetaInv = zeros(n,1);
    if (size(pos_vars,1) > 0) % Q_bar is the diagonal of the Hessian of the objective function.
        R_ThetaInv(pos_vars) = z(pos_vars)./x(pos_vars) + rho;
        R_ThetaInv(free_vars) = rho;
    else
        R_ThetaInv(:) = rho;
    end
    NS.ThetaInv = R_ThetaInv;
    NS.delta = delta;
    NS.IPiter = iter;
    NS.pos_vars = pos_vars;
    NS.free_vars = free_vars;
% ____________________________________________________________________________________________________________________ %
 
% ******************************************************************************************************************** %
% END OF FILE.
% ******************************************************************************************************************** %
end
