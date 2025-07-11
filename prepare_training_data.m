% Load labels.txt and prepare the feature extraction
data = importdata('Train/labels.txt'); % Import labels
img_nrs = data(:, 1);                  % Image numbers
true_labels = data(:, 2:4);            % True labels for digits
num_images = size(img_nrs, 1);         % Total number of images

% Initialize variables for features and labels
Features = [];
Labels = [];

% Loop through all images to extract features
for i = 1:num_images
    % Load the image
    img_name = sprintf('Train/captcha_%04d.png', img_nrs(i));
    im = imread(img_name);
    
    % Extract features using FeatureExtraction.m
    FeatMat = FeatureExtraction(im);
    
    % Reshape and append features and labels
    Features = [Features; reshape(FeatMat, 3, [])]; % Flattened features for 3 digits
    Labels = [Labels; true_labels(i, :)];          % Corresponding labels
end

% Save the extracted features and labels
save('TrainingData.mat', 'Features', 'Labels');
disp('Feature extraction completed and saved.');
