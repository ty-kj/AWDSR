function [Z,S,obj] = AWLSR1(X,Z_ini,lambda1,lambda2,max_iter,miu,rho)
% The code is written by Kun Jiang, 
% for the following reference:
% remove the weighted sparse term in N. Bouhlel's paper:
% Adaptive weighted least squares regression for subspace clustering
%min_{Z,E,W} |S^0.5.*E|_F^2+lambda1/2|S|_F^2|+lambda2*|Z|_F^2 s.t. E=X-XZ

max_miu = 1e8;
tol  = 1e-6;
tol2 = 1e-2;
C1 = zeros(size(X));
C2 = zeros(size(Z_ini));
S = ones(size(X));
Ctg = inv(X'*X+eye(size(X,2)));
for iter = 1:max_iter
    if iter == 1
        Z = Z_ini;
        U = Z_ini;
        E = X-X*Z;
    end
    Z_old = Z;
    U_old = U; 
    E_old = E;

    % ------------ S ------------- %
    S_temp = -(E.^2)/lambda1;
    S = zeros(size(S_temp));
    for ii = 1:size(E,2)
        S(:,ii) = EProjSimplex(S_temp(:,ii));
    end
    % --------- E -------- %
    G = X-X*Z+C1/miu;
    E = (miu*G)./(miu+2*S);
    
    % -------- Z ------------ %
    M1 = X-E+C1/miu;
    M2 = U-C2/miu;
    Z = Ctg*(X'*M1+M2);
    Z = Z - diag(diag(Z));
    for ii = 1:size(Z,2)
        idx = 1:size(Z,2);
        idx(ii) = [];
        Z(ii,idx) = EProjSimplex_new(Z(ii,idx));
    end
    % ------------ U ------------ %
    tempU = Z+C2/miu;
    U = miu*tempU/(miu+2*lambda2);    
 
    % ------ C1 C2 miu ---------- %
    L1 = X-X*Z-E;
    L2 = Z-U; 
    C1 = C1+miu*L1;
    C2 = C2+miu*L2;
    
    LL1 = norm(Z-Z_old,'fro');
    LL2 = norm(U-U_old,'fro');
    LL3 = norm(E-E_old,'fro');
%     LL4 = norm(S-S_old,'fro');
    SLSL = max(max(LL1,LL2),LL3)/norm(X,'fro');
    if miu*SLSL < tol2
        miu = min(rho*miu,max_miu); 
    end
    stopC = (norm(L1,'fro')+norm(L2,'fro'))/norm(X,'fro');
    if stopC < tol
%         iter
        break;
    end
    obj(iter) = stopC;   
end
end




    