function [droptol,itmax] = settol(droptol,iter,itmax,nnzL,roof)
% ==================================================================================================================== %
% settol: This function dynamically sets the droptolerance used construct a preconditioner for the Schur complement.
% -------------------------------------------------------------------------------------------------------------------- %
% droptol = settol(droptol,iter,itmax,nnzL,roof):  If the number of CG/MINRES iterations is large, then 
%                                                  we have to increase the effectiveness of the preconditioner,
%                                                  by decreasing the droptolerance. Similarly, if the iterative method
%                                                  converges fast, the next approximation will be less good.
% ____________________________________________________________________________________________________________________ %   
    droptol_thr = 1e12;
    if (nnzL > roof && iter < itmax/2)
        droptol = min(droptol*3,droptol_thr);
        return
    elseif (nnzL > roof && iter < 3*itmax/4)
        droptol = min(droptol*2,droptol_thr);
        return
    elseif (nnzL > roof && iter < itmax)
        droptol = min(droptol/2,droptol_thr);
        return
    elseif (nnzL > roof)
        droptol = min(droptol/3,droptol_thr);
       % itmax = itmax + 100;
        return
    end
    
  
    if iter > itmax-1
        droptol = max(droptol/10,1/droptol_thr);
    elseif iter > 3*itmax/4
        droptol = max(droptol/5,1/droptol_thr);
    elseif iter > itmax/2
        droptol = max(droptol/2,1/droptol_thr);
    elseif iter < itmax/5
        droptol = min(droptol*2,droptol_thr);
    end
    
  %  if (iter == itmax && droptol == 1/droptol_thr)
  %      itmax = itmax + 100;
  %  end
end


