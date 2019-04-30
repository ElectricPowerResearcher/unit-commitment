clear
clc
yalmip;
Cplex;
%%ϵͳ����
%���в�����������ֵ��ʾ
paragen=xlsread('excel2017','�������');
loadcurve=xlsread('excel2017','��������');
netpara=xlsread('excel2017','�������');
branch_num=size(netpara);%�����е�֧·
branch_num=branch_num(1,1);
PL_max=netpara(:,6);%��·��󸺺�
PL_min=netpara(:,7);%��·��С����
limit=paragen(:,3:4);%�������������//limit(:,1)��ʾ���ޣ�limit(:,2)��ʾ����
para=paragen(:,5:7);%�ɱ�ϵ��//para(:,1)��ʾϵ��a,para(:,2)��ʾϵ��b,para(:,3)��ʾϵ��c��
price=100;
para=price*para;%�۸���
lasttime=paragen(:,9);%����ʱ��
Rud=paragen(:,8);%������������//�����м����������ٶ���ͬ
H=paragen(:,10);%�����ɱ�
J=paragen(:,11);%��ͣ�ɱ�
u0=[1 1 1 1 1 1];%��ʼ״̬
%% ��ģ����
%������
gennum=size(paragen);
gennum=gennum(1,1);
%�ڵ���
numnodes=size(loadcurve);
numnodes=numnodes(1,1)-1;
%ʱ�䷶Χ
T=size(loadcurve);
T=T(1,2)-1;
%���Ի��ֶ���(����Ҫ����)
m=4;
%��ʱ�̽ڵ��ܸ���
PL=loadcurve(numnodes+1,2:T+1);
%%
%���߱���
u=binvar(gennum,T,'full');%״̬����
p=sdpvar(gennum,T,'full');%��������ʵʱ����p(i,t)
Ps=sdpvar(gennum,T,m,'full');%�ֶγ���
costH=sdpvar(gennum,T,'full');%�����ɱ�
costJ=sdpvar(gennum,T,'full');%��ͣ�ɱ�
sum_PowerGSDF=sdpvar(T,branch_num,numnodes,'full');%��������������ת���ܺ�
%% Ŀ�꺯�����Ի�
MaxPs=zeros(gennum,T,m);%�����ʾ�ֶγ���������
st=[];%stԼ����ʼ��
for i=1:gennum   %Ŀ�꺯�����Ի���ֶγ����Ĳ���ʽԼ��
   for t=1:T
     for s=1:m
	MaxPs(i,t,s)=(limit(i,1)-limit(i,2))/m;
    st=st+[Ps(i,t,s)>=0,Ps(i,t,s)<=MaxPs(i,t,s)];
     end
   end
end
K=zeros(gennum,m);%ú�ĺ�����б��ֵ
for i=1:gennum
for s=1:m
K(i,s)=2*para(i,1)*(2*s-1)*MaxPs(i,1,1)+para(i,2);%�Ƶ��򻯺��ú��б��
end
end
 %Ŀ�꺯�����Ի���ֶγ����ĵ�ʽԼ��
for i=1:gennum 
    for t=1:T
st=st+[p(i,t)==(sum(Ps(i,t,:),3)+u(i,t)*limit(i,2))];
    end
end
%% Ŀ�꺯��
totalcost=0;%������óɱ���С
%���Ի������ųɱ�Ŀ��
for i=1:gennum
for t=1:T
for s=1:m
    totalcost=totalcost+K(i,s)*Ps(i,t,s);%���Ի�ú�ĳɱ�
end
    totalcost=totalcost+u(i,t)*(para(i,2)*limit(i,2)+para(i,1)*limit(i,2)^2+para(i,3));%���ϱ�ʾ���鿪��������С���� ���в�����ú��
    totalcost=totalcost+costH(i,t)+costJ(i,t);%���ϻ�����ͣ�����Ŀ�ͣ���ɱ�
end
end
%ԭ���κ���ʽ�����ųɱ�Ŀ��
% for i=1:gennum
%     for t=1:T
%     totalcost=totalcost+para(i,1)*p(i,t).^2+para(i,2)*p(i,t)+para(i,3)*u(i,t);  %ú�ĳɱ�
%     totalcost=totalcost+costH(i,t);                                %�����ɱ�
%     totalcost=totalcost+costJ(i,t);                                %��ͣ�ɱ�
%     end
% end
%%
for t=1:T
st=st+[sum(p(:,t))==PL(1,t)];%����ƽ��Լ��;
end
%%
for t=1:T
    for i=1:gennum
  st=st+[u(i,t)*limit(i,2)<=p(i,t)<=u(i,t)*limit(i,1)];%�������������Լ��
    end
end
%% ��������Լ��
%����ʽ�����Ƶ����
% %�������������
% Su=(Pmax+Pmin)/2;
% %ͣ���������
% Sd=(Pmax+Pmin)/2;
%Ru=Rud;Rd=Rud;
% %������Լ��
% for t=2:T
% st=st+[p(:,t)-p(:,t-1)<=u(:,t-1).*(Ru-Su)+Su];
% end
% %������Լ��
% for t=2:T
%st=st+[p(:,t-1)-p(:,t)<=u(:,t).*(Rd-Sd)+Sd];
% end
%չ�����ʽ��
for t=2:T
    for i=1:gennum
    % st=st+[-Rud(i,1)*u(i,t)+(u(i,t)-u(i,t-1))*limit(i,2)-limit(i,1)*(1-u(i,t))<=p(i,t)-p(i,t-1)];
    % st=st+[p(i,t)-p(i,t-1)<=Rud(i,1)*u(i,t-1)+(u(i,t)-u(i,t-1))*limit(i,2)+limit(i,1)*(1-u(i,t))];
    %����ԭʽ���ܹػ��Ժ���޷��ٿ����ˣ�������ʽ
    st=st+[p(i,t-1)-p(i,t)<=Rud(i,1)*u(i,t)+(1-u(i,t))*(limit(i,2)+limit(i,1))/2];%����
    st=st+[p(i,t)-p(i,t-1)<=Rud(i,1)*u(i,t-1)+(1-u(i,t-1))*(limit(i,2)+limit(i,1))/2];%����
    end
end
%% �ȱ���Լ��
hp=0.05;%�ȱ���ϵ��
for t=1:T
st=st+[sum(u(:,t).*limit(:,1)-p(:,t))>=hp*PL(1,t)];
end
%% ��ͣʱ��Լ��
%����Լ��
for t=2:T
    for i=1:gennum
        indicator=u(i,t)-u(i,t-1);%��ͣʱ��Լ���ļ򻯱��ʽ���Լ��Ƶ��ģ�,indicatorΪ1��ʾ������Ϊ0��ʾֹͣ
        range=t:min(T,t+lasttime(i)-1);
        st=st+[u(i,range)>=indicator];
    end
end
%ͣ��Լ��
for t=2:T
    for i=1:gennum
        indicator=u(i,t-1)-u(i,t);%��ͣʱ��Լ��
        range=t:min(T,t+lasttime(i)-1);%�ر�����ʱ������
        st=st+[u(i,range)<=1-indicator];
    end
end
%% ��ͣ�ɱ�Լ��
for t=1:T   %��ͣ�ɱ�����Լ��
    for i=1:gennum
        st=st+[costH(i,t)>=0]; 
        st=st+[costJ(i,t)>=0];
    end
end
for i=1:gennum  %��ͣ�ɱ�����Լ��
   for t=2:T
         st=st+[costH(i,t)>=H(i,1)*(u(i,t)-u(i,t-1))];
         st=st+[costJ(i,t)>=J(i,1)*(u(i,t-1)-u(i,t))];
   end
    st=st+[costH(i,1)>=H(i,1)*(u(i,1)-u0(1,i))];%��ʼ״̬�µ���ͣ�ɱ�
    st=st+[costJ(i,1)>=J(i,1)*(u0(1,i)-u(i,1))];
end
%% ֱ������Լ��
%% ֱ�������µĵ��ɾ���ڵ������ʼ��
netpara(:,4)=1./netpara(:,4);%�翹�����ɵ���
slack_bus=26;%����ͬ��ƽ��ڵ�Ÿ���
Y=zeros(numnodes,numnodes);
%% ֱ�������ĵ��ɾ������
for k=1:branch_num 
    i=netpara(k,2);%�׽ڵ�
    j=netpara(k,3);%β�ڵ�
    Y(i,j)=-netpara(k,4);%���ɾ����зǶԽ�Ԫ��
    Y(j,i)= Y(i,j);
end
for k=1:numnodes
       Y(k,k)=-sum(Y(k,:)); %���ɾ����еĶԽ�Ԫ�� 
end
%��ɾ����ƽ��ڵ����ڵ�������
Y(slack_bus,:)=[];
Y(:,slack_bus)=[];
xlswrite('ֱ�������µĽڵ㵼�ɾ���',Y,'ƽ��ڵ�ȡ�ڵ�26�Žڵ�');
%% �������ת�Ʒֲ�����(GSDF)
X=inv(Y);%XΪֱ�������½ڵ㵼�ɾ���������
xlswrite('�ڵ㵼�ɵ������',X,'ƽ��ڵ�δ����');
row=zeros(1,numnodes-1);%numnodes-1����Ϊ�ڵ㵼�ɾ���ȥ����ƽ��ڵ�
%�ٴ�����ƽ��ڵ�ľ���ֵ������ֱ���������妤��=����P,ƽ����Ƕ�ʼ��Ϊ0�����������漰ƽ��ڵ��X��Ϊ0
X=[X(1:slack_bus-1,:);row;X(slack_bus:numnodes-1,:)];%����ȫ0��
column=zeros(numnodes,1);
X=[X(:,1:slack_bus-1) column X(:,slack_bus:numnodes-1)];%����ȫ0��
xlswrite('�ڵ㵼�ɵ������',X,'ƽ��ڵ㲹��');
G=zeros(branch_num,numnodes);%GSDF����ת�ƾ����ʼ��
for k=1:branch_num
    m=netpara(k,2);%�׶˽ڵ�
    n=netpara(k,3);%ĩ�˽ڵ�
    xk=netpara(k,4);%֧·k���迹ֵ
for i=1:numnodes
    G(k,i)=(X(m,i)-X(n,i))*xk;%�������ת�Ʒֲ�����
end
end
power_gen=paragen(:,2);%�������Ӧ�ڵ�
sum_nodeGSDF=zeros(T,branch_num);%���ɽڵ���������ת��
for t=1:T
    for k=1:branch_num
        for i=1:gennum
            sum_PowerGSDF(t,k,i)=G(k,power_gen(i,1))*p(i,t);%���Ｔ���������·���������ת��ʽ
        end
        for i=1:numnodes
            sum_nodeGSDF(t,k)=sum_nodeGSDF(t,k)+G(k,i)*loadcurve(i,t+1);%���������и��ɽڵ����·���������ת��ʽ
        end
        st=st+[PL_min(k,1)<=(sum(sum_PowerGSDF(t,k,:))-sum_nodeGSDF(t,k))];
        st=st+[(sum(sum_PowerGSDF(t,k,:))-sum_nodeGSDF(t,k))<=PL_max(k,1)];
    end
end
%% ���
    ops=sdpsettings('solver', 'cplex');
result=solvesdp(st,totalcost);
double(totalcost) 
subplot(1,2,1)
bar(value(p)','stack')%����ͼ
legend('Unit 1','Unit 2','Unit 3','Unit 4','Unit 5','Unit 6');	%�������������ͼ��
subplot(1,2,2)
stairs(value(p)')
legend('Unit 1','Unit 2','Unit 3','Unit 4','Unit 5','Unit 6');	%�������������ͼ��
xlswrite('����������������',double(u),'�����ʱ����ͣ�ƻ�');
P=(sum(sum_PowerGSDF(:,:,:),3)-sum_nodeGSDF(:,:))';%����֧·��ʵʱ����
P_sp=zeros(numnodes,T);%�����ڵ��ֱ����������
for i=1:numnodes
for k=1:branch_num
    m=netpara(k,2);%�׶˽ڵ�
    n=netpara(k,3);%ĩ�˽ڵ�
    if m==i
        P_sp(i,:)=P_sp(i,:)+P(k,:);
    end
    if n==i
        P_sp(i,:)=P_sp(i,:)-P(k,:);
    end
end
end
dot_theta=zeros(numnodes,T);
dot_theta=X*P_sp;       
xlswrite('����������������',double(P),'֧·��ʱ�ε�ֱ������');
xlswrite('����������������',double(P_sp),'�ڵ��ʱ�εĳ�������');
xlswrite('����������������',double(dot_theta),'�ڵ��ʱ�εĳ������');


set(0,'ShowHiddenHandles','On')
set(gcf,'menubar','figure')
