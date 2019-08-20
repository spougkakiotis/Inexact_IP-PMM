function u = perm(x,L,P,r)

y = x(P,:);
z = L'\(L\y);
u = z(r,:);

end
