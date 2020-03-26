clear all;
% Sequential Discriminants (Q4)
load('lab2_3.mat');

A = a;
B = b;

discriminants = {};
all_num_wrong_a = {};
all_num_wrong_b = {};

% 1. Let a and b represent the data points in classes A and B. Let j = 1.
j = 1;
while (size(a,1) > 0 && size(b,1) > 0)
    misclassified_a = 1;
    misclassified_b = 1;
    while (sum(misclassified_a>0) > 0 && sum(misclassified_b>0) > 0)
        % 2. Randomly select one point from a and one point from b
        [a_index, a_point] = get_random_point(a);
        [b_index, b_point] = get_random_point(b);

        % 3. Create a discriminant G using MED with the two points as prototypes
        % 4. Using all of the data in a and b, work out the confusion matrix entries
        misclassified_a = get_misclassified_at_index(a, a_point, b_point, 1);
        misclassified_b = get_misclassified_at_index(b, a_point, b_point, 0);

        %5. If num_wrong_a != 0 and num_wrong_b != 0 then no good, go back to step 2.
    end

    num_wrong_a = sum(misclassified_a>0);
    num_wrong_b = sum(misclassified_b>0);
    % 6. This discriminant is good; save it
    discriminants = [discriminants, [a_point; b_point]];
    all_num_wrong_a = [all_num_wrong_a, num_wrong_a];
    all_num_wrong_b = [all_num_wrong_b, num_wrong_b];  

    % 7. If num_wrong_a = 0 then remove those points from b that G classifies as B.
    if (num_wrong_a == 0)
        b = remove_points(b, misclassified_b);
    end

    % 8. If num_wrong_b = 0 then remove those points from a that G classifies as A.
    if (num_wrong_b == 0)
        a = remove_points(a, misclassified_a);
    end
    
    % 9. If a and b still contain points, go back to step 2.
    j = j + 1;
end

% 1. Let j = 1
j = 1;
% 2. If Gj classifies x as class B and num_wrong_a,j = 0 then “Say Class B”
% 3. If Gj classifies x as class A and num_wrong_b,j = 0 then “Say Class A”
% 4. Otherwise j = j + 1 and go back to step 2.

function [index, point] = get_random_point(values)
    length = size(values,1);
    index = randi(length);
    point = values(index,:);
end

% If diff < 0 then class A, else class B
function [diff] = med_classify(point,a,b)
    dist_a = sqrt((a(1)-point(1))^2 + (a(2)-point(2))^2);
    dist_b = sqrt((b(1)-point(1))^2 + (b(2)-point(2))^2);

    diff = dist_a - dist_b;
end

function [misclassified] = get_misclassified_at_index(values, a, b, isA)
    length = size(values, 1);
    misclassified = zeros(length,1);
    for i=1:length
        dist = med_classify(values(i,:), a, b);
        if (isA)
            misclassified(i) = dist >= 0;
        else 
            misclassified(i) = dist < 0;
        end
    end
end

function [new_values] = remove_points(values, wrong_at_index)
    length = sum(wrong_at_index>0);
    if (length == 0)
        new_values = [];
    else
        new_values = zeros(length,2);
        index = 1;
        for i=1:size(wrong_at_index)
            if (wrong_at_index(i) == 0)
                new_values(index,:) = values(i, :);
                index = index + 1;
            end
        end
    end
end