function S0 = init_weight (X,r)
% X input data, dim*num
%S weight matrix
%min sij^r||x_i-x_j||^2, s.t.S'1=1, 0=<s_ij<=1
% randam initiaze s_ij
% use Lagrange multipliers to solve this problem

num = size(X,2);
% initialize S
eta=zeros(num,num);
S0 = zeros(num,num);
for i=1 : num
    for j=1 : num
        if j ~= i
            eta(i,j)=power(r*(X(:,i)-X(:,j))'*(X(:,i)-X(:,j))+eps,1/(1-r));
        end
    end
end
eta=power(sum(eta,2),1-r);

for i=1 : num
    for j=1 : num
        if j ~= i   
            S0(i,j)=power(eta(i)/(r*(X(:,i)-X(:,j))'*(X(:,i)-X(:,j))+eps),1/(r-1));
        end
    end
end
disp('Initialization of S0 has been completed!')