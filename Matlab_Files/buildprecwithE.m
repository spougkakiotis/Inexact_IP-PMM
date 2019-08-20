function [PS] = buildprecwithE(NS,nnz_switch,droptol,mu,nev,solver)
%% ==================================================================================================================== %
% Build Preconditioner with matrix E: An approximation of the Schur complement.
% -------------------------------------------------------------------------------------------------------------------- %
% [M,LAAT,V,B_V,T,P,Pinv,maxpiv] = buildprecwithE(m,nnz_switch,droptol,mu,A,B,Theta,delta,nev)
% ____________________________________________________________________________________________________________________ %

    PS = struct();                                % Struct containing the relevant information for the preconditioner.
    PS.nnz_switch = nnz_switch;
    PS.nev = nev;
    PS.droptol = droptol;
    threshold = min(droptol,mu*droptol);
    n = size(NS.A,2);
    m = size(NS.A,1);
    PS.n = n;
    PS.m = m;
    PS.instability = false;
    %% ================================================================================================================ %
    % building matrix E, which approximates Theta, and with it, build the approximate normal equations' matrix M.
    % ----------------------------------------------------------------------------------------------------------------- %
   
    B = true(n,1);
    E = 1./(NS.ThetaInv + spdiags(NS.Q,0));
    if (solver == "minres")
        PS.Q_barInv = E;
    end
    N = (E<threshold);
    B = xor(B,N);                                 % N = {1,...,n}\B.
  
  
    E(N) = 0;
    if (nnz(B) > 0)
        M = NS.A(:,B)*(spdiags(E(B),0,nnz(B),nnz(B))*NS.A_tr(B,:)) + (NS.delta).*speye(m);
    else
        M = speye(m);
    end
    maxpiv = max(spdiags(M,0));
    if (~isnan(maxpiv) && ~isinf(maxpiv) )
        PS.instability = false;
        PS.maxpiv = maxpiv;
        if (nnz(B) > 0)
            M = M + min(1e-2,maxpiv*eps)*speye(m);                                  % extra ``regularization'' 
            [PS.L_M,chol_flag,PS.P] = chol(M,'lower','vector');                         % Cholesky factorization
        else
            PS.L_M = speye(m);
            chol_flag = 0;
            PS.P = 1:m;
        end
        PS.Pinv(PS.P) = 1:m;
        
        if (chol_flag ~= 0)
            PS.instability = true;
        end
    else
        PS.instability = true;
        return;
    end
    % _________________________________________________________________________________________________________________ %    
    %% ================================================================================================================ %
    % Correct the preconditioner if it is very dense.
    % ----------------------------------------------------------------------------------------------------------------- %
    if nnz(PS.L_M)  > nnz_switch
        PS.switch = true;
        % V contains the approximate eigenvectors of M^{-1} (A Theta A_tr) as columns
        [PS.V,D] = eigs(@(x) perm(NE_multiplier(x,NS),PS.L_M,PS.P,PS.Pinv),m,nev,'lm','Tolerance',1e-2);
        maxeig = spdiags(D,0);
	    cond = max(maxeig)/min(maxeig);           % partial condition number 
        if (cond > 1e8 || cond <= 0)
           PS.switch = false;
           PS.V = [];
           PS.TL = [];
           PS.B_V = [];
           return
        end
	    fprintf('partial condition number %12.2f\n',cond)
        PS.B_V = zeros(m,size(PS.V,2));
        for i = 1:size(PS.V,2)             
            PS.B_V(:,i) = NE_multiplier(PS.V(:,i),NS);    
        end
        T = PS.V'*PS.B_V;
        [PS.TL,PS.TU,PS.TP] = lu(T);
    else
        PS.switch = false;
        PS.V = []; PS.TL = []; PS.B_V = [];
    end
    % ________________________________________________________________________________________________________________ %
