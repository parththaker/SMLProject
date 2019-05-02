%% new model decentral
clear all;
close all;
clc;

L = 10;
N = 20;
K = 3; % rank of matrix X
%R = 40; % radius of nuclear norm
X_true = zeros(N,L);
lambda=2;

%% X_true:
for i=1:K
    p_i = transpose(sin((1:N)*pi/(N+1)));
    q_i = rand(L,1);
    
    X_true = X_true + abs(p_i*q_i');
end

X_true = X_true./K;
[U,RR,V] = svd(X_true,'econ');
R = 0.9*sum(diag(RR));

%% alpha generation:
%X_true(X_true>1)=1;
sparse_control=2;
alpha_indices = randi([1,N],sparse_control,L);
alpha_true  = zeros(N,L); % just for now and assumed to be known.
for ell=1:L
    alpha_true(alpha_indices(:,ell),ell) = rand(sparse_control,1);
    alpha_true(:,ell)=alpha_true(:,ell)./(sum(alpha_true(:,ell))+0.4);
end
Lower_triang = eye(N)+tril(ones(N),-1);
alpha_y = ones(N,L)-Lower_triang*(alpha_true);
%some_indices = randperm(L,L-5);
%alpha_true(some_indices)=0;
%% true data
x_p = repmat(rand(N,1),1,L);%for now
Y = zeros(size(X_true));
for ell=1:L
    Y(:,ell) = diag(alpha_y(:,ell))*X_true(:,ell);%+0.001.*randn(size(X_true)); % no missing measurements
    %Y(:,ell) = diag(alpha_y(:,ell))*x_p(:,ell);%+0.001.*randn(size(X_true));
end

% % %% describe the network/graph here with weight matrix: use erdos renyi graph: p=0.1:
% G = rand(L,L)<0.5;
% G = triu(G,1);
% G = G+G';
% % Adjacency matrix
% Adjacency=G;
% 
% % Metropolis Hastings matrix:
% degree_L = sum(Adjacency,2);
% M_H = zeros(L);
% for m_h_i =1:L
%     for m_h_j=1:L
%         M_H(m_h_i,m_h_j) = Adjacency(m_h_i,m_h_j)./(max(degree_L(m_h_i),degree_L(m_h_j)));
%     end
% end

%Weight = diag( ones(L,1) - M_H*ones(L,1) ) + M_H;
%T = 150; %number of iterations
x_i = (rand(N,L)); %start with random initialization %one matrix for each node.

[N,L]=size(Y);
x_i_old=x_i;
Obj_val=[];
pp_cost=[10];
pl=1; %to plot or not
%% run for loop for consensus
%% init alpha:
alpha=0.01*rand(N,L);
for iter=1:20
    
    %% solve for x
    big_A  = ones(N,L)-Lower_triang*alpha;
    cvx_begin
    variable x_i(N,L);
    minimize(norm(Y-x_i.*big_A,'fro'));
    subject to
    x_i >= 0;
    x_i <= 1;
    norm_nuc(x_i) <= R;
    cvx_end
    
    Obj_val = [Obj_val; cvx_optval];
    %% solve for alpha:
    for al=1:L
        %% part 2:
        %% update alpha_ell:
        cvx_begin quiet
        variable alpha_ell(N);
        z_ell = Y(:,al)-x_i(:,al);
        X_ell = diag(x_i(:,al))*Lower_triang;
        %          z_ell = Y(:,al)-X_true(:,al);
        %          X_ell = diag(X_true(:,al))*Lower_triang;
        
        minimize(norm(z_ell+X_ell*alpha_ell)+0.005*lambda*sum(alpha_ell));
        subject to
        sum(alpha_ell) <= 1;
        alpha_ell >= 0;
        alpha_ell <= 1;
        cvx_end
        
        %alpha_ell(alpha_ell<1e-5)=0;
        alpha(:,al)=alpha_ell;
    end
    if(pl)
        figure;
        axes('ylim',[0,1]);
        for i=1:L
            subplot(L/2,2,i);
            stem(alpha_true(:,i),'-o');
            hold on
            stem(alpha(:,i),'-*');
            
        end
    
    
        %%
        figure;
        plot(X_true,'color','blue');
        hold on
        MSE_all=zeros(L,1);
        for i=1:L
%             if(mod(i,2))
%                 color_string = 'red';
%             else
%                 color_string = 'black';
%             end
            plot(x_i(:,i),'color','black');
%             hold on;
%             MSE_all(i) = norm(x_i(:,i)-X_true(:,i))^2./(N*L);
        end
        
        figure;
        plot(MSE_all,'-*');
        
        %% Y:
        figure;
        for i=1:L
            subplot(L/2,2,i);
            plot(Y(:,i),'-*');
            hold on
            L_tilde = diag(ones(N,1)-Lower_triang*alpha(:,i));
            y_ell =   L_tilde*x_i(:,i);
            plot(y_ell,'-o');
        end
    end
    if(iter<20)
    close all;
    end
    
end
