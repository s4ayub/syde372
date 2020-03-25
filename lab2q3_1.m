% Parametric Model Estimation (Q3.1)
load('lab2_2.mat');

% Compute sample mean and covariance
m_a = [mean(al(:,1)), mean(al(:,2))];
m_b = [mean(bl(:,1)), mean(bl(:,2))];
m_c = [mean(cl(:,1)), mean(cl(:,2))];

s_a = cov(al);
s_b = cov(bl);
s_c = cov(cl);

% Test points
n_points = 500;

x1_grid = linspace(0, 450, n_points);
x2_grid = linspace(0, 450, n_points);
[x1,x2] = meshgrid(x1_grid, x2_grid);
x = [x1(:) x2(:)];

decision_grid = zeros(n_points, n_points);

y_a = mvnpdf(x,m_a,s_a);
y_a = reshape(y_a, length(x2_grid), length(x1_grid));
y_b = mvnpdf(x,m_b,s_b);
y_b = reshape(y_b, length(x2_grid), length(x1_grid));
y_c = mvnpdf(x,m_c,s_c);
y_c = reshape(y_c, length(x2_grid), length(x1_grid));

% ML classification
for i = 1:length(x1_grid)
    for j = 1:length(x2_grid)
        lh_A = y_a(j,i);
        lh_B = y_b(j,i);
        lh_C = y_c(j,i);
        
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
scatter(al(:,1),al(:,2),'r');
scatter(bl(:,1),bl(:,2),'g');
scatter(cl(:,1),cl(:,2),'b');
hold off;
axis on;
xlim([0,450]);
ylim([0,450]);
xlabel('x_1');
ylabel('x_2');
set(gca,'YDir','normal');
title('ML Classification using Parametric Estimation');
legend('Cluster A', 'Cluster B', 'Cluster C');