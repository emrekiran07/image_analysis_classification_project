load('TrainingData.mat');

% Reshape Labels: [1200 × 3] → [3600 × 1]
Y = reshape(Labels', [], 1);  % 3 columns become rows

% Reshape Features: [1200 × 45] (15×3 per image) → [3600 × 15]
X = reshape(Features', 15, [])';  % Transpose first!

% Define weak learner
t = templateTree('MaxNumSplits', 10);

% Train AdaBoost model
AdaBoostModel = fitcecoc(X, Y, 'Learners', t);

% Save
save('AdaBoostModelv2.mat', 'AdaBoostModel');
fprintf('Training completed. Model saved.\n');
