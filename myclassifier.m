function predicted_label = myclassifier(im)
    % Load the trained SVM model
    load('AdaBoostModelv2.mat'); 

    % Extract features for the input image
    Features = FeatureExtraction(im); % Extract features for all 3 digits

    % Predict labels in batch
    predicted_label = predict(AdaBoostModel, Features);
end
