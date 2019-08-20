%This script loads various NETLIB problems and solves them using IP_PMM
clear all;
clc;

suitesparse = false;

if (~suitesparse)
    %The path on which all the netlib problems lie
    lib_path = '../../../NETLIB_PROBLEMS_IN_MATLAB_FORM/netlib';
else
    lib_path = '../../SuiteSparse';
end
%Finds all the Netlib problems and stores their names in a struct
d = dir(fullfile(lib_path,'*.mat')); 

whos d

%Open the file to write the results
%fileID = fopen('Netlib_tabular_format_final_results.txt','a+');
fileID = fopen('results.txt','a+');
fields = {'A','obj','sense','rhs','lb','ub','vtype','modelname','varnames','constrnames'};
total_iters = 0;
total_time = 0;
total_in_iters = 0;
scaling_direction = 'r';
scaling_mode = 3;
pc_mode = true;
tol = 1e-6;
print_mode = 2;
maxit = 200;
%Each indice k=1..num_of_netlib_files gives the name of each netlib problem through d(i).name
w = [1:96];
problems_converged = 0;
%for k =w 
 %  load(fullfile(lib_path,d(k).name))
  %      disp(d(k).name)
%end   
for k = w
        load(fullfile(lib_path,d(k).name))
        disp(d(k).name)

    %file = strcat('output_',d(k).name)
    file = 'output';
    if suitesparse
        c = Problem.aux.c;
        A = Problem.A;
        b = Problem.b;


        ub = Problem.aux.hi;
        lb = Problem.aux.lo;

        sense = zeros(size(A,1),1);
        sense(:) = '=';
        sense = char(sense);
        [c,A,b,free_variables,objective_const_term] = LP_Convert_to_Standard_Form(c, A, b, lb, ub, sense);
    else
    
        c = model.obj;
        A = model.A;
        b = model.rhs;
        [c,A,b,free_variables,objective_const_term] = LP_Convert_to_Standard_Form(c, A, b, model.lb, model.ub, model.sense);
    end

    n = size(A,2);
    m = size(A,1);
    Q = sparse(n,n);
     if (scaling_direction == 'r')
        [D,~] = Scale_the_problem(A,scaling_mode,scaling_direction);
        A = A*spdiags(D,0,n,n); % Apply the right scaling.
        c = c.*D;
    elseif (scaling_direction == 'l')
        [D,~] = Scale_the_problem(A,scaling_mode,scaling_direction);
        A = spdiags(D,0,m,m)*A;  % Apply the left scaling.
        b = b.*D;
    elseif (scaling_direction == 'b')
        [D_R,D_L] = Scale_the_problem(A,scaling_mode,scaling_direction);
        if (size(D_L,1) ~= 0)
            A = (spdiags(D_L.^(1/2),0,m,m)*A)*spdiags(D_R.^(1/2),0,n,n);
            b = b.*D_L;
        else
            A = A*spdiags(D_R,0,n,n); % Apply the right scaling.        
        end
        c = c.*D_R;
    end
    time = 0;
    tic;
	    [x,y,z,opt,iter,totiter,autval] = IP_PMM(file,c,A,Q,b,free_variables,tol,maxit,pc_mode,print_mode); 
	    total_iters = total_iters + iter;
    time = time + toc;
    total_time = total_time + time;
    total_in_iters = total_in_iters + totiter;
    obj_val = c'*x + objective_const_term;
    if suitesparse
        if (opt == 1)
           problems_converged = problems_converged + 1;
           fprintf(fileID,'%s & %d & %d & %9.2f & opt  \n',d(k).name, iter, totiter, time); 
           fprintf(fileID,'The optimal solution objective is %d.\n',obj_val);
        else
           fprintf(fileID,'%s & %d & %d & non-opt \n',d(k).name, iter, time); 
        end
    else
        if (opt == 1)
           problems_converged = problems_converged + 1;
           fprintf(fileID,'%s & %d & %d & %9.2f & opt  \n',model.modelname, iter, totiter, time); 
           fprintf(fileID,'The optimal solution objective is %d.\n',obj_val);
        else
           fprintf(fileID,'%s & %d & %d & non-opt \n',model.modelname, iter, time); 
        end
    end
end
fprintf(fileID,'The total iterates were: %d and the total time was %d\n Problems converged: %d. Total inner iters %d.\n',total_iters,total_time,problems_converged,total_in_iters);
fclose(fileID);

fprintf('The total iterates were: %d and the total time was %d\n Problems converged: %d, Total inner iters %d.\n',total_iters,total_time,problems_converged,total_in_iters);

