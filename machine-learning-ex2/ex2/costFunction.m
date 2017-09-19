function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%Make small x the transpose of X
x = X';
%This creates a vector of Hypotheses.
hypo = sigmoid(X * theta);

%loop for every y value
for i = 1:m
%Logistic Regression cost function
J = J + (-y(i) .* log(hypo(i)) - (1 - y(i)) .* log(1 - hypo(i)));
end 
J = J / m;

%Vectorized form of Gradient Descent
grad = (1 / m) .* ((hypo - y)' * X)';




% =============================================================

end
