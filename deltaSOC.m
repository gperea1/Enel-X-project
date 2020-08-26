
% German Perea
% This space is for making predictions for 
% change of SOC


%You will need to import the following
%They are provided as CSV files 

%% Uncomment for Data training and testing with 10 outliers in 

%M = trainingnormalized1;
%test_data = testnormalized1;


%% Data is for training and testing without outliers 
% This data gave lower metrics 
M = trainnormalized2;
test_data = testnormalized2;

%% TRAINING OUR MODEL in series-parallel architecture 

% The index is in date format: 'yyyy-MM-dd HH:mm:ss'
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
% Converting the values into a correct format for training 

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

% Trying different combination of features 
%input = [bat;amps;volts];

%input = [bat;amps;volts;building;SOC;adjusted];
%input = bat;
%input  = [bat;building];
input = [bat;amps;volts];
 
% narxnet(inputDelays,feedbackDelays,hiddenSizes,feedbackMode,trainFcn)

% inputDelays = 1:2
% feedbackDelays = 1:2
% hiddenSizes = 10 (neurons in the hidden layer)
% feedbackMode = default "open"
% trainFcn = default "Levenberg-Marquardt backpropagation"

narx_net = narxnet(1:2,1:2,10);
narx_net.divideFcn = '';

% Set the number of minimmum gradient moving step 

narx_net.trainParam.min_grad = 1e-15;


[p,Pi,Ai,t] = preparets(narx_net,con2seq(input),{},deSOC);


% Input the testing data here 
% Plugging the testing here
% The following is for the testing data 

test_da = test_data{:,:};
test_da = str2double(test_da);
test_bat = test_da(:,2)';
test_SOC =  test_da(:,3)';
test_building = test_da(:,4)';
test_adjusted = test_da(:,5)';
test_amps = test_da(:,6)';
test_volts = test_da(:,7)';
test_deSOC = con2seq(test_da(:,8)');

% The following are times
test_week = test_da(:,9)';
test_hour = test_da(:,10)';
test_day = test_da(:,11)';

[narx_net, tr] = train(narxnet,p,t,Pi);
yp = sim(narx_net,p,Pi);
e = cell2mat(yp) - cell2mat(t);
TS1 = size(t,2);

% Metrics for training

train_mse = mse(e);
train_rmse = sqrt(train_mse);
train_mae = mae(e);
figure(2)

plot(1:TS1,cell2mat(t),'b',1:TS1,cell2mat(yp),'r')
legend('True','Predicted')
title('Training data: Battery(kW), Ampere(amps), Voltage(Volts)')
xlabel('Row in data')
ylabel('Normalized deltaSOC')



%% TESTING OUR MODEL in Parallel Architecture 

% Closed_Loop
in2 = [test_bat;test_amps;test_volts];
figure(3)
[narx_net_closed] = closeloop(narx_net);
y1 = test_deSOC;
u1 = con2seq(in2);
[p1,Pi1,Ai1,t1] = preparets(narx_net_closed,u1,{},y1);
yp1 = narx_net_closed(p1, Pi1,Ai1);
TS = size(t1, 2);

%yp1 = sim(narx_net_closed,Pi1,Ai1);
e2 = cell2mat(yp1) - cell2mat(t1);

plot(1:TS,cell2mat(t1),'b',1:TS,cell2mat(yp1),'r')
legend('True','Predicted')
title('Testing data: Battery(kW), Ampere(amps), Voltage(Volts)')
xlabel('Row in data')
ylabel('Normalized deltaSOC')

% For testing purposes

% Metrics for testing 
Y = narx_net_closed(p1, Pi1, Ai1);
MSE = mse(narx_net_closed, t1, Y);
RMSE = sqrt(MSE);
MAE = mae(e2);












