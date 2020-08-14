
% German Perea
% This space is for making predictions for 
% change of SOC
M = trainingnormalized;
test_data = testnormalized1;

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

input = [bat];

% Creating our NARX NN

narx_net = narxnet(1:2,1:2,15);
narx_net.divideFcn = '';

% Set the number of minimmum gradient moving step 

% We have different methods if we have specific preferences
% Setting the gradient 
narx_net.trainParam.min_grad = 1e-10;


[p,Pi,Ai,t] = preparets(narx_net,con2seq(input),{},deSOC);


% Input the testing data here 

% Plugging the testing here

% The following is the cleaning for our TESTING data 
test_da = test_data{:,:}; % correcting format
test_bat = test_da(:,2)'; % Battery 
test_SOC =  test_da(:,3)'; % State of charge
test_building = test_da(:,4)'; % Building 
test_adjusted = test_da(:,5)'; % Adjusted (Building + Battery)
test_amps = test_da(:,6)'; % Ampere
test_volts = test_da(:,7)'; % Voltage
test_deSOC = con2seq(test_da(:,8)'); % Change in State of Charge 
test_week = test_da(:,9)'; % Week of the year
test_hour = test_da(:,10)'; % Hour of the day
test_day = test_da(:,11)'; % Day of the week

% Features we would like to select 
in2 = [test_bat];

[narx_net, tr] = train(narxnet,p,t,Pi);
yp = sim(narx_net,p,Pi);
e = cell2mat(yp) - cell2mat(t);
TS1 = size(t,2);
train_mse = mse(e);
train_rmse = sqrt(train_mse);
train_mae = mae(e);

figure(2)

plot(1:TS1,cell2mat(t),'b',1:TS1,cell2mat(yp),'r')
legend('True','Predicted')
title('Inputs: Battery and Week')
xlabel('Row in data')
ylabel('Normalized deltaSOC')



%% This section covers how to test our model

% Closed_Loop


figure(3)


narx_net_closed = closeloop(narx_net);
y1 = test_deSOC;
u1 = con2seq(in2);
[p1,Pi1,Ai1,t1] = preparets(narx_net_closed,u1,{},y1);
yp1 = narx_net_closed(p1, Pi1,Ai1);
TS = size(t1, 2);

plot(1:TS,cell2mat(t1),'b',1:TS,cell2mat(yp1),'r')
e2 = cell2mat(yp1) - cell2mat(t1);

plot(1:TS,cell2mat(t1),'b',1:TS,cell2mat(yp1),'r')
legend('True','Predicted')
title('deltaSOC: with Battery(kW) as input')
xlabel('Row in data')
ylabel('Normalized deltaSOC')
% 

% For testing purposes
Y = narx_net_closed(p1, Pi1, Ai1);

% Metrics
MSE = mse(narx_net_closed, t1, Y);
RMSE = sqrt(MSE);
% MAE 
MAE = mae(e2);






