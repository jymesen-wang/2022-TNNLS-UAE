%object function min_{S>=0,S*1=1,W'*St*W=I,F} 2Tr(F'*L_s*F)+lambda||X'W-F||^2
function [W, iter,S,Sr,B, ob] = UAE(X, d, lambda, r)
%% input and output
%X: dim*num data matrix, each column is a data point
%d: the reduced dimension
% W: dim*d projection matrix
% S: num*num learned symmetric similarity matrix/weight matrix
% F: purified data matrix
%The online version is: https://ieeexplore.ieee.org/abstract/document/9448340
%% initialization S
[~,num] = size(X);
% time 
S0 = init_weight(X,r);

ob0=0;
iter = 0;
error = 1;

while iter<30
    iter = iter + 1;
% while 1
    Sr=(power(S0,r)+(power(S0,r))')/2;
    Lr = diag(sum(Sr)) - Sr;
    L = inv(2*Lr/lambda + eye(num));
    B = eye(num) - L;
    B = (B+B')/2;
%     calculate W,F
    H = eye(num)-1/num*ones(num);
    St = X*H*X';
    St = (St+St')/2;
    invSt = inv(St);
    invSt = (invSt + invSt')/2;
    M1=X*B*X';
    M1=(M1+M1')/2;
    M = invSt*(M1);
    [W, ~] = eig1(M, d, 0, 0);
     %     normlization
    W = W*diag(1./sqrt(diag(W'*St*W)));
    %     calculate F dim*num
    F = (L*X'*W)';
    
    %      calculate object function
    ob(iter) = trace(2*F*Lr*F'+ lambda*W'*X*X'*W -2*lambda*F*X'*W +lambda*F*F');
    
    error = abs(ob(iter) - ob0)/ob0;
%     error = abs(ob(iter) - ob0);
         %     update weight matrix S
        eta=zeros(num,num);
        S = zeros(num,num);
        for i=1 : num
            for j=1 : num
                if j ~= i
                    eta(i,j)=power((r*((F(:,i)-F(:,j))'*(F(:,i)-F(:,j))+eps)),1/(1-r));
                end
            end
        end
        eta=power(sum(eta,2),1-r);

        for i=1 : num
            for j=1 : num
                if j ~= i
                    S(i,j)=power(eta(i)/(r*((F(:,i)-F(:,j))'*(F(:,i)-F(:,j))+eps)),1/(r-1));
                end
            end
        end
        S0 = S;
        ob0=ob(iter);
       fprintf('complete %d iteration\n', iter);
end





