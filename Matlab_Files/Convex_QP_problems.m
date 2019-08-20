%This script loads various NETLIB problems and solves them using
%Dual_Regularized IPM
clear all;
clc;
%The path on which all the netlib problems lie
QP_problems_path = '../../../QP_PROBLEMS/QPset/maros'; 

%Finds all the Netlib problems and stores their names in a struct
d = dir(fullfile(QP_problems_path,'*.mat')); 

%Each indice i=1..num_of_netlib_files gives the name of each netlib problem
%though d(i).name

%Open the file to write the results
fileID = fopen('results.txt','a+');
%fileID1 = fopen('QP_problems_performance_prof_time.txt','a+');
%fileID2 = fopen('QP_problems_performance_prof_iter.txt','a+');


model = struct();
fields = {'H','name','xl','xu','al','au','g','g0','A'};
total_iters = 0;
total_time = 0;
scaling_option = 3;
scaling_direction = 'r';
tol = 1e-8; 
max_iters = 200;
total_in_iters = 0;

pc = true;
print_mode = 2;
problems_converged = 0;
w = [1:122];
for k = w
        file = 'output';

    if (isfield(model,fields)) %If any of the fields is missing, dont remove anything
        model = rmfield(model,fields); %Remove all fields before loading new ones
    end
    model = load(fullfile(QP_problems_path,d(k).name));
    model.name
    n = size(model.A,2);
    m = size(model.A,1);
  
    [model,b,free_variables,objective_const_term] = QP_Convert_to_Standard_Form(model);
    
    n_new = size(model.A,2);
    m_new = size(model.A,1);
    model.H = [model.H sparse(n,n_new -n)]; 
    model.H = [model.H ;sparse(n_new-n,n_new)];
    D = Scale_the_problem(model.A,scaling_option,scaling_direction);
    if (scaling_direction == 'l')
        model.A = spdiags(D,0,m_new,m_new)*model.A;  % Apply the left scaling.
        b = b.*D;
    elseif (scaling_direction == 'r')
        model.A = model.A*spdiags(D,0,n_new,n_new);
        model.g = model.g.*D;
        model.H = spdiags(D,0,n_new,n_new)*model.H*spdiags(D,0,n_new,n_new);
    end
    n = n_new;
    m = m_new;
    
    time = 0;
    tic;
    [x,y,z,opt,iter,totiter,autval] = IP_PMM(file,model.g,model.A,model.H,b,free_variables,tol,max_iters,pc,print_mode);
    total_iters = total_iters + iter;
    time = time +toc;
    total_time = total_time + time;
        total_in_iters = total_in_iters + totiter;

    obj_val = model.g'*x + objective_const_term + model.g0 + (1/2)*(x'*(model.H*x));
    if (opt == 1)
        problems_converged = problems_converged + 1;
        fprintf(fileID,'%s & %d & %d &%d & opt  \n',model.name, iter, totiter, time); 
        fprintf(fileID,'The optimal solution objective is %d.\n',obj_val);
        %fprintf(fileID1,'%d \n', time);
        %fprintf(fileID2,'%d \n', iter);
    else
        fprintf(fileID,'%s & %d & %d & non-opt \n',model.name, iter, time); 
        %fprintf(fileID1,'inf \n');
        %fprintf(fileID2,'inf \n');
    end
end
fprintf(fileID,'The total iterates were: %d and the total time spent was: %d and %d problems converged. Total in iters %d\n',total_iters,total_time,problems_converged,total_in_iters);
fclose(fileID);
%fclose(fileID1);
%fclose(fileID2);
