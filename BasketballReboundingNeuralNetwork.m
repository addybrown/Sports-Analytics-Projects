%% Attempt at Developing (ANN) for categorizing players

%Goal is to feed into the network a set of 4 inputs 
%Usage, Rebounding Rate, Minutes Played, Offensive Rebounds Per Game,
%Salary
%Categorize player as elite rebounder for value 
%Position


%% Setup Data Set

n=4; %number of elements
m=5000; %number of data


data(1,:)=USG(19691:24691);
data(2,:)=ORB(19691:24691);
data(3,:)=DRB(19691:24691);
data(4,:)=DBPM(19691:24691);

A=zeros(n,m);
A=str2double(data);

%% Collect Training Data 

%Generate data for training data 

%% Define Constants 

K=0.5; %Training Constant for Data

%% Define Input nodes
AvgUsage=18.1; %(%) Usage Rate;
AvgReboundingRate=[23 21 24 25 30]; %Defined by Position 1=PG, 2=SG, 3=SF.. etc
AvgSalary=6.2; %million USD
AvgORB=[3.2 4.1 5.3 6.2 7.3]; %done by position 
AvgMinutes=15; %MPG 

%% Define Initial Weights
%Start off with one hidden layer in the network
WiUsage=0.35;
WiReboundingRate=0.65;
WiSalary=0.45;
WiORB=0.21;
WiMinutes=0.10;

w1=[0.35 0.65 0.45 0.21 0.10]';
%%inputs generic 
inputs=[17*0.35 24*0.65 5.5*0.45 4.1*0.21 13*0.10];

%% Define Expected 

w2=[0.25 0.15 0.15;
    0.25 0.15 0.15;
    0.25 0.15 0.15;
    0.25 0.15 0.15;];



w3=[0.12 0.34 0.56]';



x=sigmf(x,[1 1]);

diffout=(x-1)*(1-x)*x;

