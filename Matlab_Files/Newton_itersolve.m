function [dx,dy,dz,instability,iter,drop_direction] = Newton_itersolve(fid,pred,NS,PS,res_p,res_d,res_mu,maxit,solver)
	%% ==================================================================================================================== %
	% Newton_backsolve    Solve linear system with factorized matrix, by using backward substitution.
	% -------------------------------------------------------------------------------------------------------------------- %
	% OUTPUT:
	%  [dx,dy,dz,instability] = newtonsolve(NS,res_p,res_d,res_mu,A,A_tr,pos_vars,free_vars)
	%  i.e. the Newton direction and a boolean parameter indicating critical ill-conditioning.
	%
	% Author: Spyridon Pougkakiotis.
	% ____________________________________________________________________________________________________________________ %
	m = size(res_p,1);
	n = size(res_d,1);
	instability = false;
    drop_direction = false;
	dx = zeros(n,1);
	dz = zeros(n,1);
	dy = zeros(m,1);
	if (size(NS.pos_vars,1) > 0)
	    temp_res = zeros(n,1);
	end
	%% ==================================================================================================================== %
	% Compute the Newton system's right hand side
	% -------------------------------------------------------------------------------------------------------------------- %
	if (size(NS.pos_vars,1) > 0)
	    temp_res(NS.pos_vars) =  res_mu(NS.pos_vars)./(NS.x(NS.pos_vars));
	    rhs = [res_d-temp_res; res_p];
    else
	    rhs = [res_d; res_p];
    end
    
    if (solver == "pcg")
        Theta = 1./(NS.ThetaInv+spdiags(NS.Q,0));
        rhs_y = NS.A*(Theta.*rhs(1:n)) + rhs(n+1:n+m);
    else
        Theta = 1./(NS.ThetaInv);
    end
    % ____________________________________________________________________________________________________________________ %
    
    %% ==================================================================================================================== %
	% Call the respective solver the problem using the constructed  preconditioner, whose parts are stored in struct PS.
	% -------------------------------------------------------------------------------------------------------------------- %
    warn_stat = warning;
    warning('off','all');
    if (solver == "pcg")
        tol = max(1e-10,NS.IP_tol/max(1,norm(rhs_y,'Inf')));
        [lhs_y, flag, res, iter] = pcg(@(x) NE_multiplier(x,NS), rhs_y, tol, maxit, @(x) Precond_Operator(x,PS,solver));
        lhs_x = (Theta).*(-rhs(1:n) + NS.A_tr*lhs_y);
        lhs = [lhs_x; lhs_y];
        accuracy_bound = 1e-1;
    elseif (solver == "minres")
        tol = max(1e-10,(NS.IP_tol)/max(1,norm(rhs,'Inf')));
        [lhs, flag, res, iter] = minres(@(x) AS_multiplier(x,NS), rhs, tol, maxit, @(x) Precond_Operator(x,PS,solver));
        accuracy_bound = 1e-1;
    end
    warning(warn_stat);
    % ____________________________________________________________________________________________________________________ %
    %% ==================================================================================================================== %
    % Compute the Newton directions and report the relevant statistics.
    % -------------------------------------------------------------------------------------------------------------------- %
    if (flag > 0) % Something went wrong, so we assume that the preconditioner is not good enough -> increase quality.
        iter = maxit;
        if (res > accuracy_bound)
            drop_direction = true;
            return;
        elseif (flag == 2 || flag == 4 || flag == 5)
            instability = true;
            fprintf('Instability detected during the iterative method. flag = %d.\n',flag);
            return;
        end
    end
    
 
    if (flag == 1 && ~pred)
        flag = res;
    end
    if (~pred)
        fprintf('%4d %9.2e %9.2e %9.2d ',iter, flag, PS.droptol, nnz(PS.L_M));
        fprintf(fid,'%8d & %8d & %8d & %s \n',NS.IPiter,iter,nnz(PS.L_M));
    else
        fprintf('%4d %9.2e %9.2e %9d %9.2e \n',iter, flag, PS.droptol, nnz(PS.L_M),PS.maxpiv);
    end
   
    if (nnz(isnan(lhs)) > 0 || nnz(isinf(lhs)) > 0 || (max(lhs) == 0 && min(lhs) == 0)) % Check for ill-conditioning.
        instability = true;
        iter = maxit;
        fprintf('Instability detected during the iterative method.\n');
        return;
    end
    dx = lhs(1:n,1);
    dy = lhs(n+1:n+m,1);
    if (size(NS.pos_vars,1) > 0)
        dz(NS.pos_vars) = (res_mu(NS.pos_vars)-NS.z(NS.pos_vars).*dx(NS.pos_vars))./NS.x(NS.pos_vars);
        dz(NS.free_vars) = 0;
    end
    % ____________________________________________________________________________________________________________________ %
end 