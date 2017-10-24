function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%just some possible C and sigma values
%We will try them all, which will give us 64 possible combos
possible_C = [0.01 0.03 0.1 0.3 1 3 10 30];
possible_Sigma =[0.01 0.03 0.1 0.3 1 3 10 30];
% initialize the "best Choice" of C and sigma to 0 and 0
%the last element is the error when using some c and sigma
best_choice = [0 0 999999];

%loop through all possible C's and Sigmas
for i = 1:8
  for j = 1:8
    %get each possible C val from the vector above
    C = possible_C(i);
    %get each possible sigma val from the vector above
    sigma = possible_Sigma(j);
    %taken from ex6.m: this creates the model for SVM using C and Sigma
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    %get predictions
    predictions = svmPredict(model, Xval);
    %check their error
    pred_error = mean(double(predictions ~= yval));
    %if the error of this combination of C and Sigma
    %is lower than the current C and Sigma's error
    %replace them
    if pred_error < best_choice(3)
      best_choice = [C sigma, pred_error]
    end
  endfor
 endfor

%set C and Sigma to the best values we found earlier.
C = best_choice(1);
sigma = best_choice(2);





% =========================================================================

end
