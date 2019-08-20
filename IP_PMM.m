function [x,y,z,opt,iter,totiter,autval] = IP_PMM(file,c,A,Q,b,free_variables,tol,maxit_IPM,pc,printlevel)
% ==================================================================================================================== %
% This function is an Interior Point-Proximal Method of Multipliers, suitable for solving linear and convex quadratic
% programming problems. The method takes as input a problem of the following form:
%
%                                    min   c^T x + (1/2)x^TQx
%                                    s.t.  A x = b,
%                                          x_C >= 0, for i in C \subset {1,...,n},
%                                          x_F free, for i in F = {1,...,n}\C.
% and solves it to optimality, returning the primal and dual optimal solutions (or a message indicating that the
% optimal solution was not found).
%
% INPUT PARAMETERS:
% IP_PMM(c, A, Q, b): find the optimal solution of the problem, with an error tolerance of 10^(-6).
%                     Upon success, the method returns x (primal solution), y (Lagrange multipliers) and
%                     z >= 0 (dual optimal slack variables). If the run was unsuccessful, the method  either returns
%                     a certificate of infeasibility, or terminates after 100 iterations. By default, the method
%                     scales the constraint matrix.
% IP_PMM(c, A, Q, b, free_variables): The last parameter is a matrix of indices, pointing to the free variables of the
%                                     problem. If not given, it is assumed that there are no free variables.
% IP_PMM(c, A, Q, b, free_variables, tol): This way, the user can specify the tolerance to which the problem is solved.
% IP_PMM(c, A, Q, b, free_variables, tol, max_it): This way, the user can also specify the maximum number of iterations.
% IP_PMM(c, A, Q, b, free_variables, tol, maxit, pc): predictor-corrector option.
%                                                     false: no predictor-corrector.
%                                                     true: Mehrotra's predictor-corrector.
% IP_PMM(c, A, Q, b, free_variables, tol, max_it,pc, printlevel): sets the printlevel.
%                                                              0: turn off iteration output
%                                                              1: print primal and dual residual and duality measure
%                                                              2: print centering parameter and step length
% OUTPUT: [x,y,z,opt,iter], where:
%         x: primal solution
%         y: Lagrange multiplier vector
%         z: dual slack variables
%         opt: true if problem was solved to optimality, false if problem not solved or found infeasible.
%         iter: numeber of iterations to termination.
%
% Author: Spyridon Pougkakiotis.
% _____________________________________________________________________________________________________________________ %

%% ==================================================================================================================== %
% Parameter filling and dimensionality testing.
% -------------------------------------------------------------------------------------------------------------------- %
[m, n] = size(A);
% Make sure that b and c are column vectors of dimension m and n.
if (size(b,2) > 1) b = (b)'; end
if (size(c,2) > 1) c = (c)'; end
if (~isequal(size(c),[n,1]) || ~isequal(size(b),[m,1]) )
    error('problem dimension incorrect');
end

% Make sure that A is sparse and b, c are full.
if (~issparse(A)) A = sparse(A); end
if (~issparse(Q)) Q = sparse(Q); end
if (issparse(b))  b = full(b);   end
if (issparse(c))  c = full(c);   end

% Set default values for missing parameters.
if (nargin < 5 || isempty(free_variables)) free_variables = []; end
if (nargin < 6 || isempty(tol))            tol = 1e-4;          end
if (nargin < 7 || isempty(maxit_IPM))      maxit_IPM = 100;     end
if (nargin < 8 || isempty(pc))             pc = false;          end
if (nargin < 9 || isempty(printlevel))     printlevel = 1;      end
pl = printlevel;
% _____________________________________________________________________________________________________________________ %
%% ==================================================================================================================== %
% Initialization - Mehrotra's Initial Point for QP:
% Choose an initial starting point (x,y,z). For that, we ignore the non-negativity constraints, as well as the
% regularization variables and solve the relaxed optimization problem (which has a closed form solution). Then,
% we shift the solution, to respect the non-negativity constraints. The point is expected to be well centered.
% -------------------------------------------------------------------------------------------------------------------- %
fid = fopen(file,'w');

A_tr = A';                                  %Store the transpose for computational efficiency.
P = 1:m;
pos_vars = setdiff((1:n)',free_variables);
num_of_pos_vars = size(pos_vars,1);
e_pos_vars = ones(num_of_pos_vars,1);       %Vector of ones of dimension |C|.

if (num_of_pos_vars == 0 && pc ~= false)    % Turn off Predictor-Corrector when PMM is only running.
    pc = false;
end

% =================================================================================================================== %
% Use PCG to solve two least-squares problems for efficiency (along with the Jacobi preconditioner). 
% ------------------------------------------------------------------------------------------------------------------- %
D = sum(A.^2,2) + 10;
Jacobi_Prec = @(x) (1./D).*x;
NE_fun = @(x) (A*(A_tr*x) + 10.*x);
x = pcg(NE_fun,b,10^(-3),min(300,m),Jacobi_Prec);
x = A_tr*x;
y = pcg(NE_fun,A*(c+Q*x),10^(-3),min(300,m),Jacobi_Prec);
z = c+ Q*x - A_tr*y;
% ____________________________________________________________________________________________________________________ %
if (any(isnan(x)) || any(isnan(y)) || any(isinf(x)) || any(isinf(y)) || norm(x) > 10^(14) || norm(y) > 10^(14))
    disp("Mehrotra starting point failed. Initializing using a relatively centered point.");
    y = zeros(m,1);
    x = 1000.*ones(n,1);
    z(pos_vars) = 1000.*e_pos_vars;
    z(free_variables) = 0;
else
    if (norm(x(pos_vars)) <= 10^(-4)) 
        x(pos_vars) = 0.1.*ones(num_of_pos_vars,1); % 0.1 is chosen arbitrarily
    end

    if (norm(z(pos_vars)) <= 10^(-4))
        z(pos_vars) = 0.1.*ones(num_of_pos_vars,1); % 0.1 is chosen arbitrarily
    end


    delta_x = max(-1.5*min(x(pos_vars)),0);
    delta_z = max(-1.5*min(z(pos_vars)), 0);
    temp_product = (x(pos_vars) + (delta_x.*e_pos_vars))'*(z(pos_vars) + (delta_z.*e_pos_vars));
    delta_x_bar = delta_x + (0.5*temp_product)/(sum(z(pos_vars),1)+num_of_pos_vars*delta_z);
    delta_z_bar = delta_z + (0.5*temp_product)/(sum(x(pos_vars),1)+num_of_pos_vars*delta_x);

    z(pos_vars) = z(pos_vars) + delta_z_bar.*e_pos_vars;
    x(pos_vars) = x(pos_vars) + delta_x_bar.*e_pos_vars;
    z(free_variables) = 0;
end
if (issparse(x))  x = full(x); end
if (issparse(z))  z = full(z); end
if (issparse(y))  y = full(y); end
% _____________________________________________________________________________________________________________________ %
%% ==================================================================================================================== %  
% Initialize parameters
% -------------------------------------------------------------------------------------------------------------------- %
iter = 0;
alpha_x = 0;     % Step-length for primal variables (initialization)
alpha_z = 0;     % Step-length for dual variables (initialization)
sigmamin = 0.05; % Heuristic value.
sigmamax = 0.95; % Heuristic value.
sigma = 0;
opt = 0;

if (num_of_pos_vars > 0)                             % Defined only when non-negativity constraints are present.
    mu = (x(pos_vars)'*z(pos_vars))/num_of_pos_vars; % Initial value of mu.
    res_mu = zeros(n,1);
    mu_prev = mu;
else
    mu = 0;     % Switch to a pure PMM method (no inequality constraints).
    res_mu = [];
end
header(pl);     % Set the printing choice.


if (pc == false)
    retry = 0;  % Num of times a factorization is re-built (for different regularization values)
else
    retry_p = 0;
    retry_c = 0;
end
max_tries = 15; % Maximum number of times before exiting with an ill-conditioning message.

delta = 8;            % Initial dual regularization value.
rho = 8;              % Initial primal regularization value.
lambda = y;           % Initial estimate of the Lagrange multipliers.
zeta = x;             % Initial estimate of the primal optimal solution.
no_dual_update = 0;   % Primal infeasibility detection counter.
no_primal_update = 0; % Dual infeasibility detection counter.
reg_limit = max(5*tol*(1/max(norm(A,'inf')^2,norm(Q,'inf')^2)),1e-13); % Controlled perturbation.
reg_limit = min(reg_limit,10^(-8));
autval = [];
iterlin = 50;
droptol = 1e2;  
roof = 3e7;
nev = 10;             % Number of eigenvalues for tuning 
nnz_switch = 100*m;   % if nnzL > switch then eigenvpairs are computed and tuned is performed
itertot = zeros(2*maxit_IPM,1);
solver = "minres";       % Specifies the Krylov solver for the iterative method (1: "pcg", 2: "minres").
if (solver == "pcg")
    maxit_Krylov = 250;
else
    maxit_Krylov = 750;
end

% _____________________________________________________________________________________________________________________ %
%% While block: IPM outer iterations.
while (iter < maxit_IPM)
% -------------------------------------------------------------------------------------------------------------------- %
% IP-PMM Main Loop structure:
% Until (||Ax_k - b|| < tol && ||c + Qx_k - A^Ty_k - z_k|| < tol && mu < tol) do
%   Choose sigma in [sigma_min, sigma_max] and solve:
%
%      [ -(Q + Theta^{-1} + rho I)   A^T       I] (Delta x)    (c + Qx_k - A^T y_k - [z_C_k; 0] + rho (x-zeta))
%      [           A                delta I    0] (Delta y)  = (b - Ax_k - delta (y-lambda))
%      [         Z_C                  0      X_C] (Delta z_C)    (sigma e_C - X_C Z_C e_C),
%                                                 
%   where mu = x_C^Tz_C/|C|, Theta_i = z_i/x_i for i in C, Theta_i = 0 o/w. Set (z_F)_i = 0, (Delta z_F)i = 0.
%
%   Find two step-lengths a_x, a_z in (0,1] and update:
%              x_{k+1} = x_k + a_x Delta x, y_{k+1} = y_k + a_z Delta y, z_{k+1} = z_k + a_z Delta z
%   k = k + 1
% End
%
%   Instead of the previous system, one can also employ a predictor corrector scheme. We implement Mehrotra's
%   PC scheme here.
%
% -------------------------------------------------------------------------------------------------------------------- %
    %% ================================================================================================================ %
    % Check termination criteria
    % ---------------------------------------------------------------------------------------------------------------- % 
    if (iter > 1)
        nr_res_p = new_nr_res_p;
        nr_res_d = new_nr_res_d;
    else
        nr_res_p = b-A*x;                                % Non-regularized primal residual
        nr_res_d = c-A_tr*y-z + Q*x;                     % Non-regularized dual residual.
    end
    res_p = nr_res_p - delta.*(y-lambda);                % Regularized primal residual.
    res_d = nr_res_d + rho.*(x-zeta);                    % Regularized dual residual.
    
    if (norm(nr_res_p,'Inf')/(max(1,norm(b,'inf'))) < tol && norm(nr_res_d,'inf')/(max(1,norm(c,'inf'))) < tol &&  mu < tol )
        fprintf('optimal solution found\n');
        opt = 1;
        break;
    end
    if ((norm(y-lambda)> 10^10 && norm(res_p) < tol && no_dual_update > 5)) 
        fprintf('The primal-dual problem is infeasible\n');
        opt = 2;
        break;
    end
    if ((norm(x-zeta)> 10^10 && norm(res_d) < tol && no_primal_update > 5))
        fprintf('The primal-dual problem is infeasible\n');
        opt = 3;
        break;
    end
    iter = iter+1;
    % _________________________________________________________________________________________________________________ %
    
    % ================================================================================================================ %
    % Avoid the possibility of converging to a local minimum -> Decrease the minimum regularization value.
    % ---------------------------------------------------------------------------------------------------------------- %
  %  if (no_primal_update > 5 && rho == reg_limit && reg_limit ~= 1e-13)
  %      reg_limit = 1e-13;
  %      no_primal_update = 0;
  %      no_dual_update = 0;
  %  elseif (no_dual_update > 5 && delta == reg_limit && reg_limit ~= 1e-13)
  %      reg_limit = 1e-13;
  %      no_primal_update = 0;
  %      no_dual_update = 0;
  %  end
    % ________________________________________________________________________________________________________________ %
    
    %% ================================================================================================================ %
    % Compute the Newton factorization.
    % ---------------------------------------------------------------------------------------------------------------- %
    NS = Newton_matrix_setting(iter,A,A_tr,Q,x,z,delta,rho,pos_vars,free_variables);
    % _________________________________________________________________________________________________________________ %
    
    %% Predictor-Corrector or Simple iteration if-else block
    if (pc == false) % 
        %% No predictor-corrector. 
        % ============================================================================================================ %
        % Compute the parameter sigma and based on the current solution
        % ------------------------------------------------------------------------------------------------------------ %
        if (iter > 1)
            sigma = max(1-alpha_x,1-alpha_z)^5;
        else
            sigma = 0.5;
        end

        sigma = min(sigma,sigmamax);
        sigma = max(sigma,sigmamin);
        % ____________________________________________________________________________________________________________ %
        if (num_of_pos_vars > 0)
            res_mu(pos_vars) = (sigma*mu).*e_pos_vars - x(pos_vars).*z(pos_vars);
        end
        % ============================================================================================================ %
        % Solve the Newton system and calculate residuals.
        % ------------------------------------------------------------------------------------------------------------ %
        if (iter > 1) % maximum allowed nonzero number in L_M
            nnzL = nnz(PS.L_M);
        else
            nnzL = 0;
        end  
        [droptol,maxit_Krylov] = settol(droptol,iterlin,maxit_Krylov,nnzL,roof);
        PS = buildprecwithE(NS,nnz_switch,droptol,mu,nev,solver); % Create the preconditioner for the predictor system.
        [dx,dy,dz,instability,iterlin] = Newton_itersolve(fid,1,NS,PS,res_p,res_d,res_mu,maxit_Krylov,solver);
        itertot(iter) = iterlin;
        if (instability == true) % Checking if the matrix is too ill-conditioned. Mitigate it.
            if (retry < max_tries)
                fprintf('The system is re-solved, due to bad conditioning.\n')
                delta = delta*100;
                rho = rho*100;
                iter = iter -1;
                retry = retry + 1;
                no_primal_update = 0;
                no_dual_update = 0;
                continue;
            else
                fprintf('The system matrix is too ill-conditioned.\n');
                break;
            end
        end
        retry = 0;
        % ____________________________________________________________________________________________________________ %
    elseif (pc == true) % Mehrotra predictor-corrector. ONLY when num_of_pos_vars > 0!!
    %% ================================================================================================================ %
    % Predictor step: Set sigma = 0. Solve the Newton system and compute a centrality measure.
    % ---------------------------------------------------------------------------------------------------------------- %
        res_mu(pos_vars) = - x(pos_vars).*z(pos_vars);
        % ============================================================================================================ %
        % Solve the Newton system with the predictor right hand side -> Optimistic view, solve as if you wanted to 
        %                                                               solve the original problem in 1 iteration.
        % ------------------------------------------------------------------------------------------------------------ %
        if (iter > 1) % maximum allowed nonzero number in L_M
            if (~PS.instability)
                nnzL = nnz(PS.L_M);
            end
        else
            nnzL = 0;
        end  
        [droptol,maxit_Krylov] = settol(droptol,iterlin,maxit_Krylov,nnzL,roof);
        PS = buildprecwithE(NS,nnz_switch,droptol,mu,nev,solver); % Create the preconditioner for the predictor system.
        if (PS.instability == false)
            [dx,dy,dz,instability,iterlin,drop_direction] = Newton_itersolve(fid,1,NS,PS,res_p,res_d,res_mu,maxit_Krylov,solver);
        else
            instability = true;
        end
        if (instability == true) % Checking if the matrix is too ill-conditioned. Mitigate it.
            if (retry_p < max_tries)
                fprintf('The system is re-solved, due to bad conditioning  of predictor system.\n')
                delta = delta*5;
                rho = rho*5;
                iter = iter -1;
                retry_p = retry_p + 1;
                no_primal_update = 0;
                no_dual_update = 0;
                continue;
            else
                fprintf('The system matrix is too ill-conditioned.\n');
                break;
            end
         elseif (drop_direction == true)
            if (retry_p < max_tries)
                fprintf('Predictor: Dropping the direction, due to inaccuracy.\n');
                iter = iter -1;
                retry_p = retry_p + 1;
                no_primal_update = 0;
                no_dual_update = 0;
                continue;
            else
                fprintf('Not enough accuracy.\n');
                break;
            end
        end
        retry_p = 0;
        itertot(2*iter-1) = iterlin;
        % ____________________________________________________________________________________________________________ %
        
        % ============================================================================================================ %
        % Step in the non-negativity orthant.
        % ------------------------------------------------------------------------------------------------------------ %
        idx = false(n,1);
        idz = false(n,1);
        idx(pos_vars) = dx(pos_vars) < 0; % Select all the negative dx's (dz's respectively)
        idz(pos_vars) = dz(pos_vars) < 0;     
        alphamax_x = min([1;-x(idx)./dx(idx)]);
        alphamax_z = min([1;-z(idz)./dz(idz)]);
        tau = 0.995;
        alpha_x = tau*alphamax_x;
        alpha_z = tau*alphamax_z;
        % ____________________________________________________________________________________________________________ %
        centrality_measure = (x(pos_vars) + alpha_x.*dx(pos_vars))'*(z(pos_vars) + alpha_z.*dz(pos_vars));
        mu = (centrality_measure/(num_of_pos_vars*mu))^2*(centrality_measure/num_of_pos_vars);
    % ________________________________________________________________________________________________________________ %
        
    %% ================================================================================================================ %
    % Corrector step: Solve Newton system with the corrector right hand side. Solve as if you wanted to direct the 
    %                 method in the center of the central path.
    % ________________________________________________________________________________________________________________ %
        res_mu(pos_vars) = mu.*e_pos_vars - dx(pos_vars).*dz(pos_vars);
        % ============================================================================================================ %
        % Solve the Newton system with the predictor right hand side -> Optimistic view, solve as if you wanted to 
        %                                                               solve the original problem in 1 iteration.
        % ------------------------------------------------------------------------------------------------------------ %
        nnzL = nnz(PS.L_M); % maximum allowed nonzeros in L_M
        [droptol,maxit_Krylov] = settol(droptol,iterlin,maxit_Krylov,nnzL,roof);
        [dx_c,dy_c,dz_c,instability,iterlin,drop_direction] = Newton_itersolve(fid,0,NS,PS,zeros(m,1),zeros(n,1),res_mu,maxit_Krylov,solver);
        if (instability == true) % Checking if the matrix is too ill-conditioned. Mitigate it.
            if (retry_c < max_tries)
                fprintf('The system is re-solved, due to bad conditioning of corrector.\n')
                delta = delta*5;
                rho = rho*5;
                iter = iter -1;
                retry_c = retry_c + 1;
                mu = mu_prev;
                no_primal_update = 0;
                no_dual_update = 0;
                continue;
            else
                fprintf('The system matrix is too ill-conditioned.\n');
                break;
            end
        elseif (drop_direction == true)
            if (retry_c < max_tries)
                fprintf('Corrector: Dropping the direction, due to inaccuracy.\n');
                iter = iter -1;
                retry_c = retry_c + 1;
                mu = mu_prev;
                no_primal_update = 0;
                no_dual_update = 0;
                continue;
            else
                fprintf('Not enough accuracy.\n');
                break;
            end
        end
        retry_c = 0;
        itertot(2*iter-1) = iterlin;
        % ____________________________________________________________________________________________________________ %
        dx = dx + dx_c;
        dy = dy + dy_c;
        dz = dz + dz_c;
    % ________________________________________________________________________________________________________________ %
    end
    
    
    %% ================================================================================================================ %
    % Compute the new iterate:
    % Determine primal and dual step length. Calculate "step to the boundary" alphamax_x and alphamax_z. 
    % Then choose 0 < tau < 1 heuristically, and set step length = tau * step to the boundary.
    % ---------------------------------------------------------------------------------------------------------------- %
    if (num_of_pos_vars > 0)
        idx = false(n,1);
        idz = false(n,1);
        idx(pos_vars) = dx(pos_vars) < 0; % Select all the negative dx's (dz's respectively)
        idz(pos_vars) = dz(pos_vars) < 0;

       
        alphamax_x = min([1;-x(idx)./dx(idx)]);
        alphamax_z = min([1;-z(idz)./dz(idz)]);
        tau = 0.995;
        alpha_x = tau*alphamax_x;
        alpha_z = tau*alphamax_z;
    else
        alpha_x = 1;         % If we have no inequality constraints, Newton method is exact -> Take full step.
        alpha_z = 1;
    end
    % ________________________________________________________________________________________________________________ %
    
    %% ================================================================================================================ %
    % Make the step.
    % ---------------------------------------------------------------------------------------------------------------- %
    x = x+alpha_x.*dx; y = y+alpha_z.*dy; z = z+alpha_z.*dz;
    if (num_of_pos_vars > 0) % Only if we have non-negativity constraints.
        mu_prev = mu;
        mu = (x(pos_vars)'*z(pos_vars))/num_of_pos_vars;
        mu_rate = min(abs((mu-mu_prev)/max(mu,mu_prev)),0.9);
        mu_rate = max(mu_rate,0.1);
    end
    % ________________________________________________________________________________________________________________ %
    
    %% ================================================================================================================ %
    % Computing the new non-regularized residuals. If the overall error is decreased, for the primal and dual 
    % residuals, we accept the new estimates for the Lagrange multipliers and primal optimal solution respectively.
    % If not, we keep the estimates constant. However, 
    % we continue decreasing the penalty parameters, limiting the decrease to the value of the minimum pivot
    % of the LDL^T decomposition (to ensure single pivots).
    % ---------------------------------------------------------------------------------------------------------------- %
    new_nr_res_p = b-A*x;
    new_nr_res_d = c + Q*x - A_tr*y - z;
    if (0.95*norm(nr_res_p) > norm(new_nr_res_p))
        lambda = y;
        if (num_of_pos_vars > 0)
            delta = max(reg_limit,delta*(1-mu_rate));  
        else
            delta = max(reg_limit,delta*0.1);               % In this case, IPM not active -> Standard PMM (heuristic)      
        end
        no_dual_update = 0;
    else
        if (num_of_pos_vars > 0)
            delta = max(reg_limit,delta*(1-0.666*mu_rate)); % Slower rate of decrease, to avoid losing centrality.       
        else
            delta = max(reg_limit,delta*0.5);               % In this case, IPM not active -> Standard PMM (heuristic)      
        end
        no_dual_update = no_dual_update + 1;
    end
    if (0.95*norm(nr_res_d) > norm(new_nr_res_d))
        zeta = x;
        if (num_of_pos_vars > 0)
            rho = max(reg_limit,rho*(1-mu_rate));     
        else
            rho = max(reg_limit,rho*0.1);                   % In this case, IPM not active -> Standard PMM (heuristic)        
        end
        no_primal_update = 0;
    else
        if (num_of_pos_vars > 0)
            rho = max(reg_limit,rho*(1-0.666*mu_rate));     % Slower rate of decrease, to avoid losing centrality.      
        else
            rho = max(reg_limit,rho*0.5);                   % In this case, IPM not active -> Standard PMM (heuristic)    
        end
        no_primal_update = no_primal_update + 1;
    end
    % ________________________________________________________________________________________________________________ %
    %% ================================================================================================================ %
    % Print iteration output.  
    % ---------------------------------------------------------------------------------------------------------------- %
    pres_inf = norm(new_nr_res_p,'inf');
    dres_inf = norm(new_nr_res_d,'inf');  
    output(fid,pl,iter,pres_inf,dres_inf,mu,sigma,alpha_x,alpha_z,delta);
    fprintf('\n');
    % ________________________________________________________________________________________________________________ %
end % while (iter < maxit)

% The IPM has terminated because the solution accuracy is reached or the maximum number 
% of iterations is exceeded, or the problem under consideration is infeasible. Print result.  
totiter = sum(itertot);
fprintf('iterations: %4d\n', iter);
fprintf('CG iterations: %4d\n', totiter);
fprintf('primal feasibility: %8.2e\n', norm(A*x-b));
fprintf('dual feasibility: %8.2e\n', norm(A'*y+z-c - Q*x));
fprintf('complementarity: %8.2e\n', full(dot(x,z)/n));  
end


%% ==================================================================================================================== %
% header + output printing functions: 
% pl = 1: primal-dual infeasibility and mu is printed at each iteration k
% pl = 2: primal-dual infeasibility, mu, sigma, and step-lengths are printed at each iteration k
% -------------------------------------------------------------------------------------------------------------------- %
function header(pl)
    if (pl >= 1)
        fprintf('%9s\n', 'Krylov    ');
        fprintf(' ');
        fprintf('%4s  ', 'its');
        fprintf('%8s  ', 'flag');
        fprintf('%9s  ', 'droptol');
        fprintf('%7s  ', 'nnz(L)');
        fprintf('%7s  ', 'IPiter');
        fprintf('%9s  ', 'pr feas');
        fprintf('%9s  ', 'dl feas');
        fprintf('%9s  ', 'mu');
        fprintf('%9s  ', 'delta');
    end
    if (pl >= 2)
        fprintf('  ');
        fprintf('%8s  ', 'sigma');
        fprintf('%8s  ', 'alpha_x');
        fprintf('%8s  ', 'alpha_z');
    end
    if (pl >= 1)
        fprintf('\n ==== ========= =========  ========  ========  =========  =========  =========  =========');
    end
    if (pl >= 2)
        fprintf('   ========  ========  ========');
    end
    if (pl >= 1) fprintf('\n'); end
end



function output(fid,pl,it,xinf,sinf,mu,sigma,alpha_x,alpha_z,delta)
    if (pl >= 1)
        fprintf(' ');
        fprintf('%8d    ', it);
        fprintf('%8.2e  ', xinf);
        fprintf('%8.2e  ', sinf);
        fprintf('%9.2e  ', mu);
        fprintf('%9.2e  ', delta);
        fprintf(fid,' ');
        fprintf(fid,'$ %8.1e $ & $ ', xinf);
        fprintf(fid,'%8.1e  $ & $', sinf);
        fprintf(fid,'%8.1e $ \\\\ \n', mu);
    end
    if (pl >= 2)
        fprintf('  ');
        fprintf('%8.2e  ', sigma);
        fprintf('%8.2e  ', alpha_x);
        fprintf('%8.2e  ', alpha_z);
    end
    if (pl >= 1) fprintf('\n'); end
end

% ____________________________________________________________________________________________________________________ %
% ******************************************************************************************************************** %
% END OF FILE
% ******************************************************************************************************************** %