% CTRL R to comment out lines
% CTRL T to uncomment out lines

% X = [1 1 3; 1 2 2.5]
% W = [-1 0 1; 0 -1 0; 1 0 1]
% F1 = relu(relu(X*W)*W)

X = [1 2 1; 1 5 1]
W1 = [-1 0 1; 0 -1 0; 1 0 -1]
W2 = [-1 0 1; 0 -1 0; 1 0 1; 1 -1 1]
W3 = W2

% F2 = onesToLeft(W3)
% F2 = onesToRight(W3)
% F2 = onesToTop(W3)
% F2 = onesToBot(W3)

F2 = sigmoid(onesToLeft(sigmoid(onesToLeft(sigmoid(X*W1, 1))*W2, 1))*W3, 1)

function Y = relu(X)
    Y = max(0, X);
end

% The ./ operator performs element-wise division
function Y = sigmoid(X, beta)
    Y = 1.0 ./ (1.0 + exp(-X*beta));
end

function Y = onesToLeft(X)
    % Create a column of ones
    ones_column = ones(size(X, 1), 1); 

    % Add the column of ones to the left side of X
    Y = horzcat(ones_column, X);
end

function Y = onesToRight(X)
    % Create a column of ones
    ones_column = ones(size(X, 1), 1); 

    % Add the column of ones to the right side of X
    Y = horzcat(X, ones_column);
end

function Y = onesToTop(X)
    % Create a row of ones
    ones_row = ones(1, size(X, 2));

    % Add the column of ones to the top side of X
    Y = vertcat(ones_row, X);
end

function Y = onesToBot(X)
    % Create a row of ones
    ones_row = ones(1, size(X, 2));

    % Add the column of ones to the bottom side of X
    Y = vertcat(X, ones_row);
end

