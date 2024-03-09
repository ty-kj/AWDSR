function [Z,S,obj] = AWSLSR(X,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho)
% The code is written by Kun Jiang, 
% for the following reference:
% N. Bouhlel Adaptive weighted least squares regression for subspace clustering

max_miu = 1e8;
tol  = 1e-6;
tol2 = 1e-2;
C1 = zeros(size(X));
C2 = zeros(size(Z_ini));
S = ones(size(X));
distX = L2_distance_1(X,X);
D = lambda3*distX;
for iter = 1:max_iter
    if iter == 1
        Z = Z_ini;
        U = Z_ini;
        E = X-X*Z;
    end
    Z_old = Z;
    U_old = U; 
    E_old = E;
%     S_old = S;

    % ------------ S -------weight------ %
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
    Z = Ctg*(X'*M1+M2-D/miu);
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




    