%
%
%	Zahra Zamanzadeh Darban, M. Hadi Valipour
%	IT2LL (Interval Type-2 Locally Linear Neuro Fuzzy Model Based on Locally Linear Model Tree)
%
%
mctr = 1;
msear = [];
data=load('first');
input=data.z;
%Lolimot Algorithm

%step 1
[sampnum dim1] = size(input);
dim = dim1 - 1;  %dimensions of X
Down(1:dim)=0;
Up(1:dim)=0;
cNum=1;
center(1,1:dim)=0;
sigma1(1,1:dim)=0;
ks1=1/2;
LF(1)=0;
phi_r(1:cNum, 1:sampnum)=1;
phi_l(1:cNum, 1:sampnum)=1;
err(1:sampnum)=1;
W_l(cNum, 1:dim+1)=1;   %initialize Weight!?
W_r(cNum, 1:dim+1)=1;
mse=Inf;

%Initial Algorithm (first center)
for i=1:dim
    Down(i) = min(input(:,i));
    Up(i) = max(input(:,i));
    cNum=1;
    center(1,i) = ( Up(i) + Down(i) )/2;
    sigma1(1,i) = abs( Up(i) - Down(i) ) * ks1;
end


% --------------- Main Algorithm -----------------------------------
while (mse>0.001)
    
    
    %step 2
    
    X = [ones(size(input,1),1) input(:,1:dim)];
    
    % find the worst LF after inserting new rules
    for j=1:cNum
        tmp=0;
        for k=1:sampnum
            tmp = tmp + (err(k)^2) * (phi_r(j,k) + phi_l(j,k))/2;
        end
        LF(j)=tmp;
    end
    [worst iworst]=max(LF(:));
    
    
    %step 3
    Down = center(iworst, :) - ((sigma1(iworst, :) / ks1)/2);
    Up = center(iworst, :) + ((sigma1(iworst, :) / ks1)/2);
    
    %check all possible division
    for i=1:dim
        cdiv1(i,:)=center(iworst, :);
        cdiv2(i,:)=center(iworst, :);
        sdiv1(i,:)=sigma1(iworst, :);
        sdiv2(i,:)=sigma1(iworst, :);
        
        cdiv1(i,i)=Down(i)+ ((sigma1(iworst,i) /ks1)/4);
        cdiv2(i,i)=Up(i)- ((sigma1(iworst,i) / ks1)/4);
        sdiv1(i,i)= (sigma1(iworst,i) / 2);
        sdiv2(i,i)= sdiv1(i,i);
    end
    
    %find best devision
    tcenter=center;
    tsigma1=sigma1;
    tmse(1:dim)=0;
    mmse=Inf;
    terr(1:sampnum)=0;
    
    for d=1:dim
        
        %step 3.a
        tcenter(iworst,:)=cdiv1(d,:);
        tsigma1(iworst, :)=sdiv1(d,:);
        tcNum=cNum+1;
        tcenter(tcNum,:)=cdiv2(d,:);
        tsigma1(tcNum,:)=sdiv2(d,:);
        
        %step 3.b
        tmiu_r(1:tcNum, 1:sampnum)=1;
        tmiu_l(1:tcNum, 1:sampnum)=1;
        tmp_phi_r(1:tcNum, 1:sampnum)=0;
        tmp_phi_l(1:tcNum, 1:sampnum)=0;
        for k=1:sampnum
            for j=1:tcNum
                for i=1:dim
                    tLMF = exp( -0.5*((input(k,i)-tcenter(j,i))^2)/tsigma1(j,i) ) - 0.005;
                    tUMF = tLMF + 0.01;
                    if tLMF < 0
                        tLMF=0;
                    end
                    tmiu_l(j,k) = tmiu_l(j,k) * tLMF;
                    if tUMF>1
                        tUMF=1;
                    end
                    tmiu_r(j,k) = tmiu_r(j,k) * tUMF;
                end
            end
            
%             if (sum(tmiu_r(:,k)) ~= 0)
                tmp_phi_r(:,k)=tmiu_r(:,k)/sum(tmiu_r(:,k));
%             end
%             if (sum(tmiu_l(:,k)) ~= 0)
                tmp_phi_l(:,k)=tmiu_l(:,k)/sum(tmiu_l(:,k));
%             end
        end
        
        %step 3.c
        landa=0.00001;  %for singularity
        for j=1:tcNum
            Q_r = diag( tmp_phi_r(j,:) );
            tmp_W_r(j,:)= inv((X'*Q_r*X)+landa*eye(dim1)) * X'*Q_r*input(:,dim1);
            Q_l = diag( tmp_phi_l(j,:) );
            tmp_W_l(j,:)= inv((X'*Q_l*X)+landa*eye(dim1)) * X'*Q_l*input(:,dim1);
        end
        
        %step 3.d
        ty_rule_r(1:tcNum, 1:sampnum)=0;
        ty_rule_l(1:tcNum, 1:sampnum)=0;
        for k=1:sampnum
            for j=1:tcNum
                for i=1:dim1
                    ty_rule_r(j,k) = sum(tmp_W_r(j,:) .* X(k,:));
                    ty_rule_l(j,k) = sum(tmp_W_l(j,:) .* X(k,:));
                end
            end
            [ty_trn(k),yl(k),yr(k)]  = KM(ty_rule_l(:,k)', ty_rule_r(:,k)', tmp_phi_l(:,k)', tmp_phi_r(:,k)', 1);
%             ty_trn(k)=ty_trn(k)./0.73;
            terr(k)=input(k,dim1) - ty_trn(k);
        end
        
        tmse(d)=sum(terr.^2)/sampnum;
        if (tmse(d)<mmse)  %select best new LLM (division)
            mmse=tmse(d);
            mse=tmse(d);
%             ibest=d;
            % add new LLMs to existing LLMs
            phi_r = tmp_phi_r;
            phi_l = tmp_phi_l;
            W_r = tmp_W_r;
            W_l = tmp_W_l;
            center=tcenter;
            sigma1=tsigma1;
            err = terr;
        end
    end
    cNum=cNum+1;
    
    msear (mctr) = mse;
    mctr = mctr + 1;
    
end

scatter(getcolumn(input(1:80,1:2),1),getcolumn(input(1:80,1:2),2),'DisplayName','F2D(1:80,1:2)(:,2) vs. F2D(1:80,1:2)(:,1)','YDataSource','F2D(1:80,1:2)(:,2)');figure(gcf)
%plot(input(:,1),'bo');
hold on;
plot(center, 'r*');


