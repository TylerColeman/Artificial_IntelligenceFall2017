function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
%Make small x the transpose of X
x = X';
%Make a vector of guesses
guess = sigmoid(theta'*x);
%loop for the size of y
for i = 1:m
  %if the guess is greater than or equal to 0.5
  %We will predict a True outcome
  if (guess(i) >= 0.5)
    p(i) = 1;
  %Otherwise a false outcome
  else
    p(i) = 0;
  endif
endfor






% =========================================================================


end
