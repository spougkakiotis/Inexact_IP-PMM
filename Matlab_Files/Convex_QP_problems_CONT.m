%This script loads various NETLIB problems and solves them using
%Dual_Regularized IPM
clear all;
clc;
%The path on which all the netlib problems lie
QP_problems_path = '../../QP_PROBLEMS/QPset/maros_CONT'; 

%Finds all the Netlib problems and stores their names in a struct
d = dir(fullfile(QP_problems_path,'*.mat')); 

%Each indice i=1..num_of_netlib_files gives the name of each netlib problem
%though d(i).name

%Open the file to write the results
fileID = fopen('QP_results.txt','a+');
%fileID1 = fopen('QP_problems_performance_prof_time.txt','a+');
%fileID2 = fopen('QP_problems_performance_prof_iter.txt','a+');


model = struct();
fields = {'c','A','Q','rl','ru','lb','ub'};
total_iters = 0;
total_time = 0;
scaling_option = 1;
scaling_direction = 'r';
tol = 1e-8; 
max_iters = 200;
total_in_iters = 0;

pc = true;
print_mode = 2;
problems_converged = 0;
w = [1:5];
for k = w
        file = 'output';
   % if (w >= 9 || w <= 13)
        model = load(fullfile(QP_problems_path,d(k).name));
        n = size(model.A,2);
        m = size(model.A,1);
        [model, b, free_variables, objective_const_term] = QP_Convert_to_Standard_Form_ContQP(model);
  %  else       
    
   
     n_new = size(model.A,2);
    m_new = size(model.A,1);
    model.Q = [model.Q sparse(n,n_new -n)]; 
    model.Q = [model.Q ;sparse(n_new-n,n_new)];
   
    D = Scale_the_problem(model.A,scaling_option,scaling_direction);
    if (scaling_direction == 'l')
        model.A = spdiags(D,0,m_new,m_new)*model.A;  % Apply the left scaling.
        b = b.*D;
    elseif (scaling_direction == 'r')
        model.A = model.A*spdiags(D,0,n_new,n_new);
        model.c = model.c.*D;
        model.Q = spdiags(D,0,n_new,n_new)*model.Q*spdiags(D,0,n_new,n_new);
    end
    n = n_new;
    m = m_new;
    
    time = 0;
    tic;
    [x,y,z,opt,iter,totiter,autval] = IP_PMM(file,model.c,model.A,model.Q,b,free_variables,tol,max_iters,pc,print_mode);
    total_iters = total_iters + iter;
    time = time +toc;
    total_time = total_time + time;
        total_in_iters = total_in_iters + totiter;

    %obj_val = model.g'*x + objective_const_term + model.g0 + (1/2)*(x'*(model.Q*x));
    if (opt == 1)
        problems_converged = problems_converged + 1;
        fprintf(fileID,'%s & %d & %d &%d & opt  \n',d(k).name, iter, totiter, time); 
        %fprintf(fileID,'The optimal solution objective is %d.\n',obj_val);
    else
        fprintf(fileID,'%s & %d & %d & non-opt \n',d(k).name, iter, time); 
    end
end
fprintf(fileID,'The total iterates were: %d and the total time spent was: %d and %d problems converged. Total in iters %d\n',total_iters,total_time,problems_converged,total_in_iters);
fclose(fileID);

