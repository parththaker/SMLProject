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
    p_i = rand(N,1);
    q_i = rand(L,1);
    
    X_true = X_true + abs(p_i*q_i');
end

X_true = X_true./K;
[U,RR,V] = svd(X_true,'econ');
R = 0.9*sum(diag(RR));

%% alpha generation:
%X_true(X_true>1)=1;
sparse_control=4;
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

%% %% describe the network/graph here with weight matrix: use erdos renyi graph: p=0.1:
G = rand(L,L)<0.5;
G = triu(G,1);
G = G+G';
%% Adjacency matrix
Adjacency=G;

%% Metropolis Hastings matrix:
degree_L = sum(Adjacency,2);
M_H = zeros(L);
for m_h_i =1:L
    for m_h_j=1:L
        M_H(m_h_i,m_h_j) = Adjacency(m_h_i,m_h_j)./(max(degree_L(m_h_i),degree_L(m_h_j)));
    end
end

Weight = diag( ones(L,1) - M_H*ones(L,1) ) + M_H;
T = 150; %number of iterations
x_i = (rand(N,L,L)); %start with random initialization %one matrix for each node.

[N,L]=size(Y);
x_i_old=x_i;
F_i = zeros(N*L,L); % one for each loc
F_i_old = zeros(N*L,L); % one for each loc
Obj_val=[];

x_i_diff = zeros(T,1);
norm_grad = zeros(T,1);
pp_cost=[10];
pl=0; %to plot or not
%% run for loop for consensus
for iter=1:10
    
    %%
        %% solve for alpha:
    for al=1:L
        %% part 2:
        %% update alpha_ell:
        cvx_begin quiet
        variable alpha_ell(N);
        z_ell = Y(:,al)-x_i(:,al,al);
        X_ell = diag(x_i(:,al,al))*Lower_triang;
        %          z_ell = Y(:,al)-X_true(:,al);
        %          X_ell = diag(X_true(:,al))*Lower_triang;
        
        minimize(norm(z_ell+X_ell*alpha_ell));%+0.05*lambda*sum(alpha_ell));
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
    end
    for t=1:T
        % for t=1:5
        obj_t=0; obj_tt = 0;
        %% gamma:
        gamma_t = 1/(t+1);
        beta_t = 0.03;
        %% gamma:
        %gamma_t = 1./t^(0.75);
        %% \hat{x}_{i} using AC:
        vec_x_i=zeros(N*L,L);
        for pp=1:L
            vec_x_i(:,pp)=vec(x_i(:,:,pp));
            %         for qq=1:L
            %             x_i(:,qq,pp)=x_i(:,qq,pp)./norm(x_i(:,qq,pp));
            %         end
        end
        % x_i_diff(t) = norm((1/N)*x_i*ones(L,1)-x_real).^2;
        vec_x_i = vec_x_i*Weight;
        x_i = reshape(vec_x_i,N,L,L);
        %% compute gradient
        for i_grad=1:L
            
            f_grad_old = zeros(N*L,1);
            f_grad = zeros(size(f_grad_old));
            
            % make this lower triangular matrix:
            L_tilde = diag(ones(N,1)-Lower_triang*alpha(:,i_grad));
            f_grad_x_old  = L_tilde'*L_tilde*x_i_old(:,i_grad,i_grad)-L_tilde'*Y(:,i_grad);
            f_grad_x  = L_tilde'*L_tilde*x_i(:,i_grad,i_grad)-L_tilde'*Y(:,i_grad);
            
            f_grad_old((i_grad-1)*N+1:i_grad*N) = f_grad_x_old ;
            f_grad((i_grad-1)*N+1:i_grad*N) = f_grad_x ;
            obj_tt = obj_tt + norm(Y(:,i_grad)- L_tilde*x_i(:,i_grad,i_grad)).^2;
            
            if(t>1)
                F_i(:,i_grad)= F_i(:,i_grad) + f_grad-f_grad_old;
            else
                F_i(:,i_grad)= f_grad;
            end
        end
        
        x_i_old = x_i;
        % if(t>1)
        F_i = F_i*Weight;
        %  end
        
        norm_grad(t) = norm((1/N)*F_i*ones(L,1));
        
        %% constrained optimization even for x_i: all values must be between zero and one:
        
        for al=1:L
            %first update x_i:
            A = reshape(F_i(:,al),N,L);
            [u1,sigma1,v1]=svds(A,1);
            
            x_i(:,:,al) = (1-gamma_t)*x_i(:,:,al) -(R)* gamma_t*u1*v1';
            
        end 
        
        Obj_val=[Obj_val;obj_tt];
    end
    
    if(obj_tt>pp_cost(end) && iter>1)
        break;
    end
    pp_cost=[pp_cost; obj_tt];
    %%
    if(pl)
    figure;
    plot(Obj_val,'-*');
    figure;
    plot(norm_grad);
    %%
    figure;
    plot(X_true,'color','blue');
    hold on
    MSE_all=zeros(L,1);
    for i=1:L
        if(mod(i,2))
            color_string = 'red';
        else
            color_string = 'black';
        end
        plot(x_i(:,:,i),'color',color_string);
        hold on;
        MSE_all(i) = norm(x_i(:,:,i)-X_true)^2./(N*L);
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
    y_ell =   L_tilde*x_i(:,i,i);    
    plot(y_ell,'-o');
end
    end
   close all; 
    
end
