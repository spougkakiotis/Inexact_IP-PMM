function [model, b, free_variables, objective_const_term] = QP_Convert_to_Standard_Form_ContQP(model)
% ==================================================================================================================== %
% QP_Convert_to_Standard_Form(model):
% This function takes as input the data of an LP in the following form:
%                       min    c^T x + (1/2)x^T Q x
%                       s.t.   rl <= Ax <= ru,
%                              lb <= x <= ub
% and transforms it in a semi-standard form, that is:
%                       min    c_bar^T x + (1/2)x^T H_bar x
%                       s.t.   A_bar x = b
%                              (x_C)_i >= 0, for i in C, (x_F)_i free for i in F,
% where C in {1,...,n} is a set the indices of constrained variables and F = {1,...,n}\C, is the set of indices
% of free variables, and n is the number of variables of the final model.
%
% Author: Spyridon Pougkakiotis
% ==================================================================================================================== %

% ==================================================================================================================== %
% Test input data, dimensions, e.t.c.
% -------------------------------------------------------------------------------------------------------------------- %
n = size(model.A,2);
m = size(model.A,1);
if (size(model.lb,2) > 1)
    model.lb = model.lb';
elseif (size(model.ub,2) > 1)
    model.ub = model.ub';
elseif (size(model.ru,2) > 1)
    model.ru = model.ru';
elseif (size(model.rl,2) > 1)
    model.rl = model.rl';
elseif (size(model.c,2) > 1)
    model.c = model.c';
elseif (~issparse(model.A))
    model.A = sparse(model.A);
end

if (size(model.c,1) ~= n || size(model.rl,1) ~= m || size(model.ru,1) ~= m || size(model.lb,1) ~= n         ...
                         || size(model.ub,1) ~= n || size(model.Q,1) ~= size(model.Q,2) || size(model.Q,1) ~= n)
    error("Incorrect input dimensions")
end
% ==================================================================================================================== %

% ==================================================================================================================== %
% Initialization.
% -------------------------------------------------------------------------------------------------------------------- %
num_of_slacks = 0;        % Counter for the slack variables to be added in the inequality constraints.
free_variables = [];      % To store the indices of the free variables.
objective_const_term = 0; % To keep constants that need to be added in the objective.
extra_constraints = 0;    % Counter for the extra constraints to be added in case of double bounds.
[rows,cols,v] = find(model.A);
if (size(rows,2) > 1)
    rows = rows';
end
if (size(cols,2) > 1)
    cols = cols';
end
if (size(v,2) > 1)
    v = v';
end
b = zeros(m,1);
% ==================================================================================================================== %

% ==================================================================================================================== %
% Make all the constraints to be of equality type (add slack variables)
% -------------------------------------------------------------------------------------------------------------------- %
for i = 1:m   
    if (model.ru(i) ~= model.rl(i))
        if (model.ru(i) == Inf)
            % we only have a lower bound, and hence we should add a slack of the form -x_slack
            num_of_slacks = num_of_slacks + 1;
            rows = [rows; i];
            cols = [cols; n + num_of_slacks];
            v = [v; -1];        % assign -1 in the element A(i,n+num_of_slacks) 
            b(i) = model.rl(i); % Fix the RHS        
        elseif (model.rl(i) == -Inf)
            % we only have an upper bound, and hence we should add a slack of the form +x_slack
            num_of_slacks = num_of_slacks + 1;
            rows = [rows; i];
            cols = [cols; n + num_of_slacks];
            v = [v; 1];         % assign 1 in the element A(i,n+num_of_slacks)
            b(i) = model.ru(i); % Fix the RHS     
        else
            % transform rl <=Axi <=ru to Axi' = aui, Axi' = ali
            extra_constraints = extra_constraints + 1;
            k_max = size(cols,1);
            for k = 1:k_max
                if (rows(k) == i)
                    cols = [cols; cols(k)];
                    rows = [rows; m + extra_constraints];
                    v = [v; v(k)];
                end
            end
            
            % treat the case of the upper bound
            num_of_slacks = num_of_slacks + 1;
            rows = [rows; i];
            cols = [cols; n + num_of_slacks];
            v = [v; 1];         % assign 1 in the element A(i,n+num_of_slacks)
            b(i) = model.ru(i); % Fix the RHS
            
            % Now add a new constraint that will treat the case of the LB
            num_of_slacks = num_of_slacks + 1;
            rows = [rows; m + extra_constraints];
            cols = [cols; n + num_of_slacks];
            v = [v; -1];
            b = [b; model.rl(i)]; % The RHS of the extra constraint     
        end      
    else
        b(i) = model.rl(i); % Already an equality constraint.
    end
    
end
% ==================================================================================================================== %
model.A = sparse(rows,cols,v,m + extra_constraints, n + num_of_slacks); % Renew the matrix to incude new constraints.
b_new = [];
% ==================================================================================================================== %
% Add extra constraints to treat the upper and lower bounds on the variables.
% -------------------------------------------------------------------------------------------------------------------- %
for i = 1:n % I want the initial n, since only those variables have bounds  
    if ((model.ub(i) == Inf) && (model.lb(i)> -Inf)) % We only have a lower bound 
        % In this case we implicitly substitute x_i = w_i + model.lb(i), w_i >=0
        if (model.lb(i) ~= 0)
            b(:) = b(:) - model.A(:,i).*model.lb(i); 
            objective_const_term = objective_const_term + model.c(i)*model.lb(i);
        end      
    elseif ((model.lb(i) == -Inf) && (model.ub(i) == Inf)) % The variable is free.     
        free_variables = [free_variables; i]; % Simply keep track of them.   
    elseif ((model.lb(i) == -Inf) && (model.ub(i) < Inf)) % We only have an upper bound. 
        % In this case we implicitly substitute x_i = ub(i) - w_i, w_i >=0
        k_max = size(cols,1);
        for k = 1:k_max
            if (cols(k) == i)
                v(k) = -v(k);
            end
        end  
        objective_const_term = objective_const_term + model.c(i)*model.ub(i);
        model.c(i) = -model.c(i); 
        if (model.ub(i) ~= 0)
            b(:) = b(:) - model.A(:,i).*model.ub(i); 
        end
    else % We have both upper and lower bound.
        % In this case we implicitly substitute x_i = w_i + lb(i)
        if (model.lb(i) ~= 0)
            b(:) = b(:) - model.A(:,i).*model.lb(i);
            objective_const_term = objective_const_term + model.c(i)*model.lb(i);
        end
        
        extra_constraints = extra_constraints + 1; %add one constraint, one variable
        num_of_slacks = num_of_slacks + 1;
        b_new = [b_new; model.ub(i) - model.lb(i)]; %The RHS of extra constraint w_i + w_i_2 = ub_i - lb_i
        rows = [rows; m + extra_constraints ; m + extra_constraints];
        cols = [cols; i ; n + num_of_slacks];
        v = [v; 1; 1]; % assigns ones in the element A(m+extra_constr,i) and A(m+extra_constr,n+num_of_slacks)  
    end
end
% ==================================================================================================================== %
b = [b; b_new];
model.c = [model.c; zeros(num_of_slacks,1)];
model.A = sparse(rows,cols,v,size(b,1),size(model.c,1));
end

