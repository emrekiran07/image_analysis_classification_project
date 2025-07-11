function prepare_data(data_folder)
    % Load labels file based on selected folder
    labels_path = fullfile(data_folder, 'labels.txt');
    data = importdata(labels_path);  % Load label file  
    img_nrs = data(:, 1);            % Image numbers
    true_labels = data(:, 2:4);      % Labels for each image
    num_images = size(img_nrs, 1);   % How many images total

    Features = [];  % Empty array to store all features
    Labels = [];    % Empty array to store all labels

    for i = 1:num_images
        % Build image filename
        img_name = fullfile(data_folder, sprintf('captcha_%04d.png', img_nrs(i)));
        im = imread(img_name);

        % Extract features for the image
        FeatMat = FeatureExtraction(im);

        % Save features and labels
        Features = [Features; reshape(FeatMat, 3, [])];  % Flattened 3 digits
        Labels = [Labels; true_labels(i, :)];            % Matching labels
    end

    % Save results
    save('TrainingData.mat', 'Features', 'Labels');
    fprintf('Feature extraction completed for %s. Saved to TrainingData.mat.\n', data_folder);
end
