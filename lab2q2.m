clear all;
clc;

load lab2_1.mat;

x = transpose(a);

pd = fitdist(x,'Normal')

x1 = [0:.1:10];
y1 = normpdf(x1,5,1);
y2 = normpdf(x1,5.07628,1.15027);

figure;
plot(x1,y1,'-',x1,y2,'r-.','LineWidth',1)
xlabel('Observation, x')
ylabel('Density, p(x)')
legend('True Gaussian Density','MLE-Gaussian')

clear all;
clc;

load lab2_1.mat;

x = transpose(b);

pd = fitdist(x,'Normal')

x1 = [0:.1:10];
y1 = exppdf(x1,1);
y2 = normpdf(x1,0.963301,0.88183);

figure;
plot(x1,y1,'-',x1,y2,'r-.','LineWidth',1)
xlabel('Observation, x')
ylabel('Density, p(x)')
legend('True Exponential Density','MLE-Gaussian')

clear all;
clc;

load lab2_1.mat;

x = transpose(a);

pd = fitdist(x,'Exponential')

x1 = [0:.1:10];
y1 = normpdf(x1,5,1);
y2 = exppdf(x1,1/0.19699);

figure;
plot(x1,y1,'-',x1,y2,'-.','LineWidth',1)
xlabel('Observation, x')
ylabel('Density, p(x)')
legend('True Gaussian Density','MLE-Exponential')

clear all;
clc;

load lab2_1.mat;

x = transpose(b);

pd = fitdist(x,'Exponential')

x1 = [0:.1:10];
y1 = exppdf(x1,1);
y2 = exppdf(x1,1.03810);

figure;
plot(x1,y1,'-',x1,y2,'r-.', 'LineWidth',1)
xlabel('Observation, x')
ylabel('Density, p(x)')
legend('True Exponential Density','MLE-Exponential')

clear all;
clc;

load lab2_1.mat;

A = transpose(a);
B = transpose(b);

a1 = min(A)
a2 = max(A)
b1 = min(B)
b2 = max(B)

%pd1 = makedist('Uniform');                      % Standard uniform distribution
pd2 = makedist('Uniform','lower',a1,'upper',a2); % A
pd3 = makedist('Uniform','lower',b1,'upper',b2); % B


x = 0:.01:10;
pdf1  = normpdf(x,5,1);
pdf11 = exppdf(x,1);
pdf2 = pdf(pd2,x);
pdf3 = pdf(pd3,x);

figure;
plot(x,pdf1,'LineWidth',1);
%plot(x,pdf11, 'LineWidth',1);
hold on;
plot(x,pdf2,'r-.','LineWidth',1);
%plot(x,pdf3,'r-.','LineWidth',1);
legend({'True Gaussian Density','MLE-Uniform'},'Location','northeast');
%legend({'True Exponential Density','MLE-Uniform'},'Location','northeast');
xlabel('Observation, x')
ylabel('Probability Density, p(x)')
hold off;

clear all;
clc;

load lab2_1.mat;

A = transpose(a);
B = transpose(b);

a1 = min(A)
a2 = max(A)
b1 = min(B)
b2 = max(B)

%pd1 = makedist('Uniform');                      % Standard uniform distribution
pd2 = makedist('Uniform','lower',a1,'upper',a2); % A
pd3 = makedist('Uniform','lower',b1,'upper',b2); % B


x = 0:.01:10;
pdf1  = normpdf(x,5,1);
pdf11 = exppdf(x,1);
pdf2 = pdf(pd2,x);
pdf3 = pdf(pd3,x);

figure;
%plot(x,pdf1);
plot(x,pdf11, 'LineWidth',1);
hold on;
%plot(x,pdf2,'r:','LineWidth',1);
plot(x,pdf3,'r-.','LineWidth',1);
%legend({'True Gaussian Density','MLE-Uniform'},'Location','northeast');
legend({'True Exponential Density','MLE-Uniform'},'Location','northeast');
xlabel('Observation, x')
ylabel('Probability Density, p(x)')
hold off;

clear all;
clc;

load lab2_1.mat
x = transpose(a);

% pd1 = normpdf(x1,5,1);
pd2 = fitdist(x,'kernel','Kernel','Normal','BandWidth',0.1);
pd3 = fitdist(x,'kernel','Kernel','Normal','BandWidth',0.4);

% Compute each pdf
x1 = 0:0.01:10;
y1 = normpdf(x1,5,1);
y2 = pdf(pd2,x1);
y3 = pdf(pd3,x1);

% Plot each pdf
figure;
plot(x1,y1,'Color','k','LineStyle','-','LineWidth',0.5)
hold on
plot(x1,y2,'Color','r','LineStyle',':','LineWidth',1)
plot(x1,y3,'Color','b','LineStyle','-.','LineWidth',1.5)
xlabel('Observation, x')
ylabel('Density, p(x)')
legend({'True Gaussian Density','PWE, BandWidth: h = 0.1','PWE, BandWidth: h = 0.4'})
hold off

clear all;
clc;

load lab2_1.mat
x = transpose(b);

% pd1 = normpdf(x1,5,1);
pd2 = fitdist(x,'kernel','Kernel','Normal','BandWidth',0.1);
pd3 = fitdist(x,'kernel','Kernel','Normal','BandWidth',0.4);

% Compute each pdf
x1 = 0:0.01:10;
y1 = exppdf(x1,1);
y2 = pdf(pd2,x1);
y3 = pdf(pd3,x1);

% Plot each pdf
figure;
plot(x1,y1,'Color','k','LineStyle','-','LineWidth',0.5)
hold on
plot(x1,y2,'Color','r','LineStyle',':','LineWidth',1)
plot(x1,y3,'Color','b','LineStyle','-.','LineWidth',1.5)
xlabel('Observation, x')
ylabel('Density, p(x)')
legend({'True Exponential Density','PWE, BandWidth: h = 0.1','PWE, BandWidth: h = 0.4'})
hold off