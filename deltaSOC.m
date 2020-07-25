
% German Perea
% This space is for making predictions for 
% change of SOC

M = datmatlab;


% Comments on data
% Columns 

% 1st = timestamp
% 2nd = battery
% 3rd = StateoCharge
% 4th = BuildingkW
% 5th = AdjustedLoadkW
% 6th = Ampere 
% 7th = Voltage 
% 8th = deltaSOC
% 9th = Weeks 

% Extra Values from table
N = M{:,:};

% Transpose the values
ts =N(:,1)';
dat = datetime(ts, 'InputFormat','yyyy-MM-dd HH:mm:ss','Format', 'yyyy-MM-dd HH:mm:ss');

% Initially it was  {str2double(N(:,2)')}

bat = str2double(N(:,2)');
SOC =  str2double(N(:,3)');
building = str2double(N(:,4)');
adjusted = str2double(N(:,5)');
amps= str2double(N(:,6)');
volts = str2double(N(:,7)');
deSOC = con2seq(str2double(N(:,8)'));
week = str2double(N(:,9)');
hour = str2double(N(:,10)');
day = str2double(N(:,11)');

%input = [bat;amps;volts];

%input = [bat;amps;volts;building;SOC;adjusted];
%input = bat;
%input  = [bat;building];
input = [bat;week];
% Works best at 3 hidden neurons

narx_net = narxnet(1:2,1:2,5);
narx_net.divideFcn = '';

% Set the number of minimmum gradient moving step 

% We have different methods if we want speficy different requirements
% Regularization


narx_net.trainParam.min_grad = 1e-10;

%
[p,Pi,Ai,t] = preparets(narx_net,con2seq(input),{},deSOC);

% Input the testing data here 

% Replace t 

[narx_net, tr] = train(narxnet,p,t,Pi, Ai);
yp = sim(narx_net,p,Pi);
e = cell2mat(yp) - cell2mat(t);
TS1 = size(t,2);
figure(2)
plot(1:TS1,cell2mat(t),'b',1:TS1,cell2mat(yp),'r')
legend('True','Predicted')
title('Inputs: Battery and Week')
xlabel('Row in data')
ylabel('Normalized deltaSOC')





% Closed_Loop 
% 
% figure(3)
% narx_net_closed = closeloop(narx_net);
% y1 = deSOC(1700:2600);
% u1 = con2seq(input(1700:2600));
% [p1,Pi1,Ai1,t1] = preparets(narx_net_closed,u1,{},y1);
% yp1 = sim(narx_net_closed ,Pi1,Ai1);
% TS = size(t1, 2);
% 
% plot(1:TS,cell2mat(t1),'b',1:TS,cell2mat(yp1),'r')

% yp1 = sim(narx_net_closed,Pi1,Ai1);
% e2 = cell2mat(yp1) - cell2mat(t1);
% TS = size(t1,2);


% plot(1:TS,cell2mat(t1),'b',1:TS,cell2mat(yp1),'r')
% legend('True','Predicted')
% title('deltaSOC: with Battery(kW) as input')
% xlabel('Row in data')
% ylabel('Normalized deltaSOC')
% 

% For testing purposes
Y = narx_net(p, Pi, Ai);
RMSE = sqrt(mse(narx_net, t, Y));










