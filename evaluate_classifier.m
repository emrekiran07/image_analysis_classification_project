function evaluate_classifier(data_folder)
    % Load labels from specified folder
    data = importdata(fullfile(data_folder, 'labels.txt'));
    img_nrs = data(:, 1);
    true_labels = data(:, 2:4);

    % Reshape labels to a single column vector (since we have 3 digits per image)
    Y_test = reshape(true_labels', [], 1);

    % Allocate predicted labels array
    predicted_labels = zeros(size(Y_test));
    N = length(img_nrs);

    fprintf('Processing images in folder: %s\n', data_folder);
    for i = 1:N
        img_index = img_nrs(i);
        img_path = fullfile(data_folder, sprintf('captcha_%04d.png', img_index));
        im = imread(img_path);

        predicted_labels((i-1)*3+1:i*3) = myclassifier(im);
    end

    % Calculate and print accuracy
    accuracy = mean(predicted_labels == Y_test);
    fprintf('\nAccuracy: %.2f%%\n', accuracy * 100);
end
