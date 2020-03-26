% Parametric Model Estimation (Q3.1)
close all
clear all

% Load Data
load('lab2_2.mat');

% Compute sample mean and covariance
m_a = [mean(al(:,1)), mean(al(:,2))];
m_b = [mean(bl(:,1)), mean(bl(:,2))];
m_c = [mean(cl(:,1)), mean(cl(:,2))];

s_a = cov(al);
s_b = cov(bl);
s_c = cov(cl);

% Test points
n_points = 902;

x1_grid = linspace(0, 450, n_points);
x2_grid = linspace(0, 450, n_points);
[x1,x2] = meshgrid(x1_grid, x2_grid);
x = [x1(:) x2(:)];

decision_grid = zeros(n_points, n_points);
%% Other Stuff
mean = [0,0]; 
sigma = [400 0; 0 400];
dat_x1 = -450:1:450; 
dat_x2 = -450:1:450; 
[dat_X1, dat_X2] = meshgrid(dat_x1,dat_x2);
F = mvnpdf([dat_X1(:) dat_X2(:)], mean, sigma);
F = reshape (F, length(dat_x2), length(dat_x1)); 

[p_a, x1_a, x2_a] = parzen(al, [0.5 0 0 450 450], F);

[p_b, x1_b, x2_b] = parzen(bl, [0.5 0 0 450 450], F);

[p_c, x1_c, x2_c] = parzen(cl, [0.5 0 0 450 450], F);

% ML classification
for i = 1:length(x1_grid)
    a_i = find(x1_a == i);
    b_i = find(x1_b == i);
    c_i = find(x1_c == i);
    for j = 1:length(x2_grid)
        a_j = find(x2_a == j);
        b_j = find(x2_b == j);
        c_j = find(x2_c == j);
        lh_A = 0;
        lh_B = 0;
        lh_C = 0;
        if (~isempty(a_i) && ~isempty(a_j))
            lh_A = p_a(a_j,a_i);
        end
        if (~isempty(b_i) && ~isempty(b_j))
            lh_B = p_b(b_j,b_i);
        end
        if (~isempty(c_i) && ~isempty(c_j))
            lh_C = p_c(c_j,c_i);
        end
        
        if (lh_A > lh_B && lh_A > lh_C) % if A
            decision_grid(j,i) = 7;
        elseif (lh_B > lh_A && lh_B > lh_C) % if B
            decision_grid(j,i) = 8; 
        else % if C
            decision_grid(j,i) = 9;
        end
    end
end

% Plot data points and clasification boundaries
figure(1);
imshow(decision_grid, [1,10]);
hold on;
scatter(at(:,1),at(:,2),'r');
scatter(bt(:,1),bt(:,2),'g');
scatter(ct(:,1),ct(:,2),'b');
hold off;
axis on;
xlim([0,450]);
ylim([0,450]);
xlabel('x_1');
ylabel('x_2');
set(gca,'YDir','normal');
title('ML Classification using Parzen Window Estimation');
legend('Cluster A', 'Cluster B', 'Cluster C');
% Parzen - compute 2-D density estimates
%
% [p,x,y] = parzen( data, res, win )    
%
%  data - two-column matrix of (x,y) points
%         (third row/col optional point frequency)
%  res  - resolution (step size)
%         optionally [res lowx lowy highx highy]
%  win  - optional, gives form of window 
%          if a scalar - radius of square window
%          if a vector - radially symmetric window
%          if a matrix - actual 2D window shape
%
%  x    - locations along x-axis
%  y    - locations along y-axis
%  p    - estimated 2D PDF (at y,x)
function [p,x,y] = parzen( data, res, win )
    if (size(data,2)>size(data,1)), data = data'; end;
    if (size(data,2)==2), data = [data ones(size(data))]; end;
    numpts = sum(data(:,3));

    dl = min(data(:,1:2));
    dh = max(data(:,1:2));
    if length(res)>1, dl = [res(2) res(3)]; dh = [res(4) res(5)]; res = res(1); end;

    if (nargin == 2), win = 10; end;
    if (max(dh-dl)/res>1000) 
      error('Excessive data range relative to resolution.');
    end;

    if length(win)==1, win = ones(1,win); end;
    if min(size(win))==1, win = win(:) * win(:)'; end;
    win = win / (res*res*sum(sum(win)));

    p = zeros(2+(dh(2)-dl(2))/res,2+(dh(1)-dl(1))/res);
    fdl1 = find(data(:,1)>dl(1));
    fdh1 = find(data(fdl1,1)<dh(1));
    fdl2 = find(data(fdl1(fdh1),2)>dl(2));
    fdh2 = find(data(fdl1(fdh1(fdl2)),2)<dh(2));

    for i=fdl1(fdh1(fdl2(fdh2)))'
      j1 = round(1+(data(i,1)-dl(1))/res);
      j2 = round(1+(data(i,2)-dl(2))/res);
      p(j2,j1) = p(j2,j1) + data(i,3);
    end;

    p = conv2(p,win,'same')/numpts;
    x = [dl(1):res:(dh(1)+res)]; x = x(1:size(p,2));
    y = [dl(2):res:(dh(2)+res)]; y = y(1:size(p,1));
end