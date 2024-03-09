function [Z,S,obj] = DSR(X,Z_ini,lambda2,lambda3,max_iter,miu,rho)
% The code is written by Kun Jiang, 
% min_{Z,E,S} ||E||_F^2+lambda2*|Z-Z.*S||_F^2+lambda3*Tr(Z'L_SZ)
% s.t. E = X-X*Z

max_miu = 1e8;
tol  = 1e-6;
tol2 = 1e-2;
C1 = zeros(size(X));
[dim,n] = size(X);

for iter = 1:max_iter
    if iter == 1
        Z = Z_ini;
        E = X-X*Z;
        S = ones(n);
        clear Z_ini
    end
    Z_old = Z;
    E_old = E;
%     S_old = S;

    % ------------ S ----graph--------- %
    B1 = 2*Z-ones(n);
    Q= 2*Z-B1.*S;
    D = L2_distance_1(Z,Z);
    S_temp = Q-0.5*(D.^2)/lambda3;
    S = zeros(size(S_temp));
    for ii = 1:n
        S(:,ii) = EProjSimplex(S_temp(:,ii));
%          S(:,ii) = S_temp(:,ii);
    end  
    DS = diag(sum(S));
    LS = DS - S; 

%      --------- E -------- %
    N = X-X*Z+C1/miu;
    E = N*miu/(miu+2*lambda2);

    % -------- Z ------------ %
    M1 = X-E+C1/miu;
    Ctg = inv(miu*X'*X+2*eye(size(X,2))+2*lambda3*LS);
    G=Z.*S;
    Z = Ctg*(miu*X'*M1+2*G);
    Z = Z - diag(diag(Z));
    for ii = 1:size(Z,2)
        idx = 1:size(Z,2);
        idx(ii) = [];
        Z(ii,idx) = EProjSimplex_new(Z(ii,idx));
    end
    
%     % ------ C1 miu ---------- %
    L1 = X-X*Z-E;
    C1 = C1+miu*L1;
    
    LL1 = norm(Z-Z_old,'fro');
    LL3 = norm(E-E_old,'fro');
    SLSL= max(LL1,LL3)/norm(X,'fro');
    if miu*SLSL < tol2
        miu = min(rho*miu,max_miu); 
    end
    stopC = (norm(LL1,'fro')+norm(LL3,'fro'))/norm(X,'fro');
    if stopC < tol
    %         iter
        break;
    end
    obj(iter) = stopC;   
end
end




    