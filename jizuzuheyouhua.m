clear
clc
yalmip;
Cplex;
%%系统参数
%所有参数均用有名值表示
paragen=xlsread('excel2017','机组参数');
loadcurve=xlsread('excel2017','负荷曲线');
netpara=xlsread('excel2017','网络参数');
branch_num=size(netpara);%网络中的支路
branch_num=branch_num(1,1);
PL_max=netpara(:,6);%线路最大负荷
PL_min=netpara(:,7);%线路最小负荷
limit=paragen(:,3:4);%机组出力上下限//limit(:,1)表示上限，limit(:,2)表示下限
para=paragen(:,5:7);%成本系数//para(:,1)表示系数a,para(:,2)表示系数b,para(:,3)表示系数c。
price=100;
para=price*para;%价格换算
lasttime=paragen(:,9);%持续时间
Rud=paragen(:,8);%上下爬坡速率//因题中简化上坡下坡速度相同
H=paragen(:,10);%启动成本
J=paragen(:,11);%关停成本
u0=[1 1 1 1 1 1];%初始状态
%% 规模变量
%机组数
gennum=size(paragen);
gennum=gennum(1,1);
%节点数
numnodes=size(loadcurve);
numnodes=numnodes(1,1)-1;
%时间范围
T=size(loadcurve);
T=T(1,2)-1;
%线性化分段数(按需要更改)
m=4;
%各时刻节点总负荷
PL=loadcurve(numnodes+1,2:T+1);
%%
%决策变量
u=binvar(gennum,T,'full');%状态变量
p=sdpvar(gennum,T,'full');%即各机组实时功率p(i,t)
Ps=sdpvar(gennum,T,m,'full');%分段出力
costH=sdpvar(gennum,T,'full');%启动成本
costJ=sdpvar(gennum,T,'full');%关停成本
sum_PowerGSDF=sdpvar(T,branch_num,numnodes,'full');%发电机的输出功率转移总和
%% 目标函数线性化
MaxPs=zeros(gennum,T,m);%这里表示分段出力的上限
st=[];%st约束初始化
for i=1:gennum   %目标函数线性化后分段出力的不等式约束
   for t=1:T
     for s=1:m
	MaxPs(i,t,s)=(limit(i,1)-limit(i,2))/m;
    st=st+[Ps(i,t,s)>=0,Ps(i,t,s)<=MaxPs(i,t,s)];
     end
   end
end
K=zeros(gennum,m);%煤耗函数的斜率值
for i=1:gennum
for s=1:m
K(i,s)=2*para(i,1)*(2*s-1)*MaxPs(i,1,1)+para(i,2);%推导简化后的煤耗斜率
end
end
 %目标函数线性化后分段出力的等式约束
for i=1:gennum 
    for t=1:T
st=st+[p(i,t)==(sum(Ps(i,t,:),3)+u(i,t)*limit(i,2))];
    end
end
%% 目标函数
totalcost=0;%机组费用成本最小
%线性化的最优成本目标
for i=1:gennum
for t=1:T
for s=1:m
    totalcost=totalcost+K(i,s)*Ps(i,t,s);%线性化煤耗成本
end
    totalcost=totalcost+u(i,t)*(para(i,2)*limit(i,2)+para(i,1)*limit(i,2)^2+para(i,3));%加上表示机组开机并以最小出力 运行产生的煤耗
    totalcost=totalcost+costH(i,t)+costJ(i,t);%加上机组启停产生的开停机成本
end
end
%原二次函数式的最优成本目标
% for i=1:gennum
%     for t=1:T
%     totalcost=totalcost+para(i,1)*p(i,t).^2+para(i,2)*p(i,t)+para(i,3)*u(i,t);  %煤耗成本
%     totalcost=totalcost+costH(i,t);                                %启动成本
%     totalcost=totalcost+costJ(i,t);                                %关停成本
%     end
% end
%%
for t=1:T
st=st+[sum(p(:,t))==PL(1,t)];%负荷平衡约束;
end
%%
for t=1:T
    for i=1:gennum
  st=st+[u(i,t)*limit(i,2)<=p(i,t)<=u(i,t)*limit(i,1)];%机组出力上下限约束
    end
end
%% 机组爬坡约束
%按下式进行推导编程
% %启动最大升速率
% Su=(Pmax+Pmin)/2;
% %停机最大降速率
% Sd=(Pmax+Pmin)/2;
%Ru=Rud;Rd=Rud;
% %上爬坡约束
% for t=2:T
% st=st+[p(:,t)-p(:,t-1)<=u(:,t-1).*(Ru-Su)+Su];
% end
% %下爬坡约束
% for t=2:T
%st=st+[p(:,t-1)-p(:,t)<=u(:,t).*(Rd-Sd)+Sd];
% end
%展开表达式：
for t=2:T
    for i=1:gennum
    % st=st+[-Rud(i,1)*u(i,t)+(u(i,t)-u(i,t-1))*limit(i,2)-limit(i,1)*(1-u(i,t))<=p(i,t)-p(i,t-1)];
    % st=st+[p(i,t)-p(i,t-1)<=Rud(i,1)*u(i,t-1)+(u(i,t)-u(i,t-1))*limit(i,2)+limit(i,1)*(1-u(i,t))];
    %由于原式可能关机以后就无法再开动了，改用下式
    st=st+[p(i,t-1)-p(i,t)<=Rud(i,1)*u(i,t)+(1-u(i,t))*(limit(i,2)+limit(i,1))/2];%下坡
    st=st+[p(i,t)-p(i,t-1)<=Rud(i,1)*u(i,t-1)+(1-u(i,t-1))*(limit(i,2)+limit(i,1))/2];%上坡
    end
end
%% 热备用约束
hp=0.05;%热备用系数
for t=1:T
st=st+[sum(u(:,t).*limit(:,1)-p(:,t))>=hp*PL(1,t)];
end
%% 启停时间约束
%启动约束
for t=2:T
    for i=1:gennum
        indicator=u(i,t)-u(i,t-1);%启停时间约束的简化表达式（自己推导的）,indicator为1表示启动，为0表示停止
        range=t:min(T,t+lasttime(i)-1);
        st=st+[u(i,range)>=indicator];
    end
end
%停机约束
for t=2:T
    for i=1:gennum
        indicator=u(i,t-1)-u(i,t);%启停时间约束
        range=t:min(T,t+lasttime(i)-1);%特别限制时间上限
        st=st+[u(i,range)<=1-indicator];
    end
end
%% 启停成本约束
for t=1:T   %启停成本零限约束
    for i=1:gennum
        st=st+[costH(i,t)>=0]; 
        st=st+[costJ(i,t)>=0];
    end
end
for i=1:gennum  %启停成本条件约束
   for t=2:T
         st=st+[costH(i,t)>=H(i,1)*(u(i,t)-u(i,t-1))];
         st=st+[costJ(i,t)>=J(i,1)*(u(i,t-1)-u(i,t))];
   end
    st=st+[costH(i,1)>=H(i,1)*(u(i,1)-u0(1,i))];%初始状态下的启停成本
    st=st+[costJ(i,1)>=J(i,1)*(u0(1,i)-u(i,1))];
end
%% 直流潮流约束
%% 直流潮流下的导纳矩阵节点参数初始化
netpara(:,4)=1./netpara(:,4);%电抗求倒数成电纳
slack_bus=26;%按不同的平衡节点号更改
Y=zeros(numnodes,numnodes);
%% 直流潮流的导纳矩阵计算
for k=1:branch_num 
    i=netpara(k,2);%首节点
    j=netpara(k,3);%尾节点
    Y(i,j)=-netpara(k,4);%导纳矩阵中非对角元素
    Y(j,i)= Y(i,j);
end
for k=1:numnodes
       Y(k,k)=-sum(Y(k,:)); %导纳矩阵中的对角元素 
end
%再删除掉平衡节点所在的行与列
Y(slack_bus,:)=[];
Y(:,slack_bus)=[];
xlswrite('直流潮流下的节点导纳矩阵',Y,'平衡节点取在第26号节点');
%% 输出功率转移分布因子(GSDF)
X=inv(Y);%X为直流潮流下节点导纳矩阵的逆矩阵
xlswrite('节点导纳的逆矩阵',X,'平衡节点未补充');
row=zeros(1,numnodes-1);%numnodes-1是因为节点导纳矩阵去掉了平衡节点
%再次引入平衡节点的矩阵值，根据直流潮流定义ΔΘ=ΧΔP,平衡机角度始终为0，所以所有涉及平衡节点的X均为0
X=[X(1:slack_bus-1,:);row;X(slack_bus:numnodes-1,:)];%插入全0行
column=zeros(numnodes,1);
X=[X(:,1:slack_bus-1) column X(:,slack_bus:numnodes-1)];%插入全0列
xlswrite('节点导纳的逆矩阵',X,'平衡节点补充');
G=zeros(branch_num,numnodes);%GSDF功率转移矩阵初始化
for k=1:branch_num
    m=netpara(k,2);%首端节点
    n=netpara(k,3);%末端节点
    xk=netpara(k,4);%支路k的阻抗值
for i=1:numnodes
    G(k,i)=(X(m,i)-X(n,i))*xk;%输出功率转移分布因子
end
end
power_gen=paragen(:,2);%发电机对应节点
sum_nodeGSDF=zeros(T,branch_num);%负荷节点的输出功率转移
for t=1:T
    for k=1:branch_num
        for i=1:gennum
            sum_PowerGSDF(t,k,i)=G(k,power_gen(i,1))*p(i,t);%这里即发电机对线路的输出功率转移式
        end
        for i=1:numnodes
            sum_nodeGSDF(t,k)=sum_nodeGSDF(t,k)+G(k,i)*loadcurve(i,t+1);%这里是所有负荷节点对线路的输出功率转移式
        end
        st=st+[PL_min(k,1)<=(sum(sum_PowerGSDF(t,k,:))-sum_nodeGSDF(t,k))];
        st=st+[(sum(sum_PowerGSDF(t,k,:))-sum_nodeGSDF(t,k))<=PL_max(k,1)];
    end
end
%% 求解
    ops=sdpsettings('solver', 'cplex');
result=solvesdp(st,totalcost);
double(totalcost) 
subplot(1,2,1)
bar(value(p)','stack')%阶梯图
legend('Unit 1','Unit 2','Unit 3','Unit 4','Unit 5','Unit 6');	%在坐标轴上添加图例
subplot(1,2,2)
stairs(value(p)')
legend('Unit 1','Unit 2','Unit 3','Unit 4','Unit 5','Unit 6');	%在坐标轴上添加图例
xlswrite('机组组合问题求解结果',double(u),'机组各时段启停计划');
P=(sum(sum_PowerGSDF(:,:,:),3)-sum_nodeGSDF(:,:))';%各段支路的实时潮流
P_sp=zeros(numnodes,T);%各个节点的直流潮流功率
for i=1:numnodes
for k=1:branch_num
    m=netpara(k,2);%首端节点
    n=netpara(k,3);%末端节点
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
xlswrite('机组组合问题求解结果',double(P),'支路各时段的直流潮流');
xlswrite('机组组合问题求解结果',double(P_sp),'节点各时段的潮流功率');
xlswrite('机组组合问题求解结果',double(dot_theta),'节点各时段的潮流相角');


set(0,'ShowHiddenHandles','On')
set(gcf,'menubar','figure')
