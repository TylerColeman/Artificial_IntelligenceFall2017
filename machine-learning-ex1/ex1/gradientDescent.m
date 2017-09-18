function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %need a vector that represents only the secon column vector
    second_col = X(:,2);
    %Calculate Hypothesis function
    htheta = X * theta;
    
    %calculate error
    error_theta0 = sum(htheta - y);
    %calculate theta
    theta0 = theta(1) - ((alpha / m) * error_theta0);
    
    %repeat from above except our error is now multipled with 
    %the second column of the X matrix.
    error_theta1 = sum((htheta - y) .* second_col);
    theta1 = theta(2) - ((alpha / m) * error_theta1);
    
    %Update theta
    theta = [theta0; theta1];





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end