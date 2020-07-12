function [alpha,beta,Q]=LanczosPrecRes(r,fun,k)
% LANCZOS Symmetric Lanczos process
%   [alpha,beta,Q,r]=Lanczos(A,r,k) applies k<=n steps of the
%   symmetric Lanczos process to the symmetric matrix A starting with
%   the vector r. Computes Q orthogonal and a symmetric tridiagonal
%   matrix given by the diagonal in the vector alpha, and the super-
%   and subdiagonal in the vector beta.

    Q(:,1)=r/norm(r); 
    if k==1, beta=[]; end;
    for j=1:k
      v=fun(Q(:,j));
      alpha(j)=Q(:,j)'*v;
      v=v-alpha(j)*Q(:,j);
      if j>1
        v=v-beta(j-1)*Q(:,j-1);
      end
      if j<k
        beta(j)=norm(v);
        Q(:,j+1)=v/beta(j) ;
      end
    end
end