% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me: wenjie@hrbeu.edu.cn 
% If you find the code is useful, please cite the following reference:
% J. Wen, B. Zhang, Y. Xu, J. Yang, and N. Han, 
% Adaptive Weighted Nonnegative Low-Rank Representation, 
% Pattern Recognition, 2018.

clear all
clc
clear memory;
addpath(genpath('data4sc'));
name = 'AR10P';
% name = 'YaleB';
% name = 'ORL';
% name = 'COIL100';
name = 'MNIST_6996';
% name = 'COIL20';
% name = 'usps_random_1000';
% name = '3ring_data';
load (name);
% fea=X;
% gnd=y;

fea = fea';
fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);%column-wise 
n = length(gnd);
nnClass = length(unique(gnd));  

options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'HeatKernel';
Z = constructW(fea',options);
Z_ini = full(Z);
clear LZ Z Z1 options


lambda1 = 0;
lambda2 = 10;
lambda3 = 10;

%lambda1=0.000010; lambda2=0.000010; lambda3=1.000000;
miu = 1e-2;
rho = 1.1;
max_iter = 30;
Ctg = inv(fea'*fea+eye(size(fea,2)));
for k=1:1
% [Z,W,obj] = AWLSR1(fea,Z_ini,lambda1,lambda2,max_iter,miu,rho);
[Z,W,S,obj] = AWDSR(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,miu,rho);
% [Z,S,obj] = DSR(fea,Z_ini,lambda2,lambda3,max_iter,miu,rho);
% [Z,W,obj] = AWLSRG(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,miu,rho);
% [Z,W,obj] = AWSLSR(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
addpath('Ncut_9');
Z_out = Z;
A = Z_out;
A = A - diag(diag(A));
A = abs(A);
A = (A+A')/2;  

[NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,nnClass);
result_label = zeros(size(fea,2),1);%vec2ind
for j = 1:nnClass
    id = find(NcutDiscrete(:,j));
    result_label(id) = j;
end
result = ClusteringMeasure(gnd, result_label);
% acc  = result(1)
% nmi  = result(2) 

acc(k)  = result(1);
nmi(k)  = result(2);  
end % 10 experiments
fprintf('all the acc values are :%f\n',acc);
fprintf('mean acc is: %f and std is: %f\n\n', mean(acc), std(acc));
fprintf('all the nmi values are :%f\n',nmi);
fprintf('mean nmi is: %f and std is: %f\n\n', mean(nmi), std(nmi));

                                                      