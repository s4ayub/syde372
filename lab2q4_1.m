clear all;
close all;
% Sequential Discriminants (Q4)
load('lab2_3.mat');

A = a;
B = b;

disc_a = {};
disc_b = {};
num_wrong_a = {};
num_wrong_b = {};

% 1. Let a and b represent the data points in classes A and B. Let j = 1.
j = 1;
limit_num_classifiers = 50;

while (size(a,1) > 0 && size(b,1) > 0 && j <= limit_num_classifiers)
    misclass_a = 1;
    misclass_b = 1;
    while (sum(misclass_a>0) > 0 && sum(misclass_b>0) > 0)
        % 2. Randomly select one point from a and one point from b
        a_point = get_random_point(a);
        b_point = get_random_point(b);

        % 3. Create a discriminant G using MED with the two points as prototypes
        % 4. Using all of the data in a and b, work out the confusion matrix entries
        misclass_a = get_misclassified_at_index(a, a_point, b_point, 1);
        misclass_b = get_misclassified_at_index(b, a_point, b_point, 0);

        %5. If num_wrong_a != 0 and num_wrong_b != 0 then no good, go back to step 2.
    end

    curr_num_wrong_a = sum(misclass_a>0);
    curr_num_wrong_b = sum(misclass_b>0);
    % 6. This discriminant is good; save it
    disc_a = [disc_a, a_point];
    disc_b = [disc_b, b_point];
    num_wrong_a = [num_wrong_a, curr_num_wrong_a];
    num_wrong_b = [num_wrong_b, curr_num_wrong_b];  

    % 7. If num_wrong_a = 0 then remove those points from b that G classifies as B.
    if (curr_num_wrong_a == 0)
        b = remove_points(b, misclass_b);
    end

    % 8. If num_wrong_b = 0 then remove those points from a that G classifies as A.
    if (curr_num_wrong_b == 0)
        a = remove_points(a, misclass_a);
    end
    
    % 9. If a and b still contain points, go back to step 2.
    j = j + 1;
end

num_wrong_a = cell2mat(num_wrong_a);
num_wrong_b = cell2mat(num_wrong_b);

[xgrid, ygrid] = get_meshgrid(A, B);
length = size(xgrid, 1) * size(xgrid,2);
predicted = zeros(1, size(xgrid, 1) * size(xgrid,2));
points = {};
index = 1;
for x = 1:size(xgrid, 2)
    for y = 1:size(ygrid, 1)
        xval = xgrid(1,x);
        yval = ygrid(y,1);
        predicted(1,index) = classify_point([xval, yval], disc_a, disc_b, num_wrong_a, num_wrong_b);
        index = index +1;
    end
end
myScatter = gscatter(xgrid(:), ygrid(:), predicted);
editGScatter(myScatter);
hold on;
a_plot = plot(A(:,1), A(:, 2), 'o', 'color', 'red', 'MarkerSize', 3, 'DisplayName', 'Class A Points');
hold on
b_plot = plot(B(:,1), B(:, 2), 'o', 'color', 'blue', 'MarkerSize', 3, 'DisplayName', 'Class B Points');
title("Sequential Discriminants Classifier",'FontSize',15);
ylabel("x2");
xlabel("x1");
% plot_disc(disc_a, disc_b);
myLegend = legend('Location', 'northeast');
limit_axes(A,B);
error = calculate_error(A, B, disc_a, disc_b, num_wrong_a, num_wrong_b)

function limit_axes(A, B)
    maxXY = max(max(A),max(B));
    minXY = min(min(A), min(B));
    xlim([minXY(1) maxXY(1)]);
    ylim([minXY(2) maxXY(2)]);
end

function error_percentage = calculate_error(A, B, disc_a, disc_b, num_wrong_a, num_wrong_b)
A_size = size(A, 1);
B_size = size(B, 1);

A_class = zeros(1, A_size);
B_class = zeros(1, B_size);

for i=1:A_size
   point = A(i,:);
   A_class(1,i) = classify_point(point, disc_a, disc_b, num_wrong_a, num_wrong_b);
end

for i=1:B_size
   point = B(i,:);
   B_class(1,i) = classify_point(point, disc_a, disc_b, num_wrong_a, num_wrong_b);
end

errors = sum(A_class ~= 1) + sum(B_class ~= 2);
error_percentage = errors/(A_size + B_size);
end

function plot_disc(disc_a, disc_b)
    colors = {'k','g','y','c','m',[.5 .6 .7],[.8 .2 .6]}; % Cell array of colros.
    for i=1:size(disc_a, 2)
        A = disc_a{i}; %[x,y]
        B = disc_b{i}; %[x,y]
        Clen = 500; % distance off the line AB that C will lie
        % Call AB the vector that points in the direction
        % from A to B.
        AB = B - A;
        % Normalize AB to have unit length
        AB = AB/norm(AB);
        % compute the perpendicular vector to the line
        % because AB had unit norm, so will ABperp
        ABperp = AB*[0 -1;1 0];
        % midpoint between A and B
        ABmid = (A + B)/2;
        % Compute new points C and D, each at a ditance
        % Clen off the line. Note that since ABperp is
        % a vector with unit eEuclidean norm, if I
        % multiply it by Clen, then it has length Clen.
        C = ABmid + Clen*ABperp;
        D = ABmid - Clen*ABperp;
        % plot them all
        hold on
        color = colors{i};
        plot([A(1);B(1)],[A(2);B(2)],'color', color, 'marker', 's');
        plot([C(1);D(1)],[C(2);D(2)], 'color', color)
        % axis equal is important because it ensures the lines appear
        % mutually perpendicular. If the axes had different units
        % along the axes, then the lines would look skewed.
        axis equal
    end
end

function point = get_random_point(values)
    length = size(values,1);
    index = randi(length);
    point = values(index,:);
end

function isClassA = med_classify(point,a,b)
    dist_a = sqrt((a(1)-point(1))^2 + (a(2)-point(2))^2);
    dist_b = sqrt((b(1)-point(1))^2 + (b(2)-point(2))^2);
    
    if (dist_a <= dist_b)
        isClassA = 1;
    else
        isClassA = 0;
    end

end

function misclassified = get_misclassified_at_index(values, a, b, isA)
    length = size(values, 1);
    misclassified = zeros(length,1);
    for i=1:length
        isClassA = med_classify(values(i,:), a, b);
        if (isA == 1&& isClassA == 1 || isA == 0 && isClassA == 0)
            misclassified(i) = 0;
        else
            misclassified(i) = 1;
        end
            
    end
end

function new_values = remove_points(values, wrong_at_index)
    length = sum(wrong_at_index>0);
    if (length == 0)
        new_values = [];
    else
        new_values = zeros(length,2);
        index = 1;
        for i=1:size(wrong_at_index)
            if (wrong_at_index(i) == 1)
                new_values(index,:) = values(i, :);
                index = index + 1;
            end
        end
    end
end

 function [xgrid, ygrid] = get_meshgrid(A ,B)
    step = 1;
    maxXY = max(max(A),max(B));
    minXY = min(min(A), min(B));
    [xgrid, ygrid] = meshgrid(minXY(1):step:maxXY(1), minXY(2):step:maxXY(2));
 end
 
 function group = classify_point(point, disc_a, disc_b, num_wrong_a, num_wrong_b)
    group = 3;
    % 1. Let j = 1
    for j=1:size(num_wrong_a, 2)
        isClassA = med_classify(point, disc_a{j}, disc_b{j});
        % 2. If Gj classifies x as class B and num_wrong_a,j = 0 then �Say Class B�
        if (isClassA == 0 && num_wrong_a(j) == 0 )
            group = 2;
            break;
        % 3. If Gj classifies x as class A and num_wrong_b,j = 0 then �Say Class A�
        elseif (isClassA == 1 && num_wrong_b(j) == 0)
            group = 1;
            break;
        end
        % 4. Otherwise j = j + 1 and go back to step 2.
    end
    
%     if (group == 3)
%         disp("Something is wrong")
%     end
 end

 % Get Colors for ScatterPlot depending on which values show up first
 function editGScatter(myScatter)
    x = 0.8;
    legend = ["Class A Region", "Class B Region", "Unclassified Region"];
    colors = [1 x x; x x 1; x 1 x];
    for i=1:size(myScatter,1)
        index = str2double(myScatter(i).DisplayName);
        myScatter(i).Color = colors(index,:);
        myScatter(i).DisplayName = legend(index);
    end
end