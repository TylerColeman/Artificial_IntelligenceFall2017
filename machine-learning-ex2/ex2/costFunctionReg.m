function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
Jnot = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Make small x the transpose of X
x = X';
%Vector of guesses
hypo = sigmoid(X * theta);

%Cost function with the regularization
for i = 1:m
J = J + (-y(i) .* log(hypo(i)) - (1 - y(i)) .* log(1 - hypo(i)));
endfor
J = J / m;
for k = 2:n
  Jnot = Jnot + theta(k)^2;
endfor
Jnot = Jnot * (lambda / (2 * m));
J = J + Jnot;
  
%Gradient Descent with Regularization
grad = (1 / m) * ((hypo - y)' * X)';
%Don't regularize Theta(1)!!!! 
for l = 2:n
  grad(l) = grad(l)  + ((lambda / m) * theta(l));
endfor



% =============================================================

end
