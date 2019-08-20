function w = Precond_Operator(x,PS,solver)
%% ==================================================================================================================== %
% Preconditioner Operator:
% --------------------------------------------------------------------------------------------------------------------- %
% [w] = Precond_Operator(x,PS,solver) takes as an input the struct containing the preconditioner blocks, as well as 
% a vector of size n+m, and returns the matrix-vector product of the inverse preconditioner by this vector.
% _____________________________________________________________________________________________________________________ %
    
    if (solver == "pcg")
        if (PS.switch)
            w = permwithtuning(x,PS.L_M,PS.P,PS.Pinv,PS.V,PS.B_V,PS.TL,PS.TU,PS.TP);
        else
            w = perm(x,PS.L_M,PS.P,PS.Pinv);
        end
    elseif (solver == "minres")
        w = zeros(PS.n+PS.m,1);
        w(1:PS.n) = PS.Q_barInv .* x(1:PS.n);
        if (PS.switch)
            w(PS.n+1:PS.n+PS.m,1) = permwithtuning(x(PS.n+1:PS.n+PS.m,1),PS.L_M,PS.P,PS.Pinv,PS.V,PS.B_V,PS.TL,PS.TU,PS.TP);
        else
            w(PS.n+1:PS.n+PS.m,1) = perm(x(PS.n+1:PS.n+PS.m,1),PS.L_M,PS.P,PS.Pinv);
        end
    else
        error('Incorrect Input argument: solver.')
    end
        

end

