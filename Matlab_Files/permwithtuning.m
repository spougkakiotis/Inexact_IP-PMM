function u = permwithtuning(x,L,P,r,V,A_V,TL,TU,TP)

% spectral
if 0
xP = x(P,1);
z = T*(V'*xP);
z = V*z;
z = z + L'\(L\xP);
u = z(r,1);


% Melina
xP = x(P,1);
z = T*(V'*xP);
z = V*z;
z = z - L'\(L\xP);
u = z(r,1);
end
    % ==================================================================================== %
    %BFGS type low-rank update to the preconditioner.
    % ------------------------------------------------------------------------------------ %
    beta = 10;
    z = V'*x;
   % z = TL'\(TL\z); % Solve using the Cholesky factors of T for numerical stability.
    tmp = TL\(TP*z);
    z = TU\tmp;
    s = A_V*z;
    w = x - s;
    w = perm(w,L,P,r);     % solve using the Cholesky factors of the initial preconditioner.
    s = A_V'*w;
  %  u = TL'\(TL\s);
    tmp = TL\(TP*s);
    u = TU\tmp;
    u = V*(beta.*z-u) + w;    
    % _____________________________________________________________________________________ %
end