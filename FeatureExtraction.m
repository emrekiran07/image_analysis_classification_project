function FeatMat = FeatureExtraction(I)
    FeatMat = zeros(3, 15); % Preallocate with 3 rows and 15 columns

    % + FFT
    I = spec_filter_image(I,0.75,0.9);

    % Step 1: Preprocess the image
    I_filt = uint8(zeros(size(I)));
    for i = 1:3
        I_filt(:,:,i) = medfilt2(I(:,:,i), [3 3]); % Noise removal
    end
    I_gray = rgb2gray(I_filt);
    I_bw = ~imbinarize(I_gray, graythresh(I_gray));
    J = imdilate(bwareaopen(imerode(I_bw, strel('disk',4)), 300), strel('disk',3));

    % Step 2: Find connected components
    cc = bwconncomp(J, 4);
    props = regionprops(cc, 'Image');

    % Step 3: Handle different numbers of components
    if cc.NumObjects == 1
        [m, n] = size(props(1).Image);
        split = round(n / 3);
        d1 = imcrop(props(1).Image, [0 0 split m]);
        d2 = imcrop(props(1).Image, [split 0 split m]);
        d3 = imcrop(props(1).Image, [2*split 0 split m]);
    elseif cc.NumObjects == 2
        % Split the wider component into two parts
        [m1, n1] = size(props(1).Image);
        [m2, n2] = size(props(2).Image);
        if n1 > n2
            d1 = imcrop(props(1).Image, [0 0 round(n1/2) m1]);
            d2 = imcrop(props(1).Image, [round(n1/2) 0 round(n1/2) m1]);
            d3 = props(2).Image;
        else
            d1 = props(1).Image;
            d2 = imcrop(props(2).Image, [0 0 round(n2/2) m2]);
            d3 = imcrop(props(2).Image, [round(n2/2) 0 round(n2/2) m2]);
        end
    elseif cc.NumObjects >= 3
        d1 = props(1).Image;
        d2 = props(2).Image;
        d3 = props(3).Image;
    else
        return; % If no components are detected, FeatMat remains zeros(3,15)
    end

    % Step 4: Extract features
    FeatMat(1,:) = [ShapeFeats(d1), hu_moments(d1)];
    FeatMat(2,:) = [ShapeFeats(d2), hu_moments(d2)];
    FeatMat(3,:) = [ShapeFeats(d3), hu_moments(d3)];
end

function F = ShapeFeats(S)
    fts = {'Circularity', 'Area', 'EulerNumber', 'Centroid', 'Solidity', 'Eccentricity'}; 
    Ft = regionprops('Table', S, fts{:});
    if ~isempty(Ft)
        [~, idx] = max(Ft.Area);
        F = [Ft{idx,:}];
    else
        F = zeros(1, 6); % Return zeros if no shape features are detected
    end
end

% utility functions for fft of image
function result_fft = forward_fft(src_image)
% forward fft transform
    I = src_image;
    result_fft = fftshift(fft2(I));
end
function result = disp_fft(src_fft)
% rescaled log magnitude in freq domain to [0,1]
    F = src_fft;
    result = mat2gray(log(abs(F)));
end
function result_image = backward_fft(src_fft)
% backward fft transform
    F = src_fft;
    result_image = ifft2(ifftshift(F));
end
function result_fft = spec_filter_fft(src_fft, LOW, HIGH)
% mixed low-high-bandpass filter,
% interesting results when exponent (ex) > 1 
% suppresses mean frequency and reveals outliers
    ex = 1; 
    d_fft = disp_fft(src_fft).^(ex);
    d_mask = ~((d_fft>LOW)&(d_fft<HIGH));
    result_fft = src_fft.*d_mask;
end
function result_image = spec_filter_image(src_image, LOW, HIGH)
    % filter frequency bands of an image; 
    % good defaults are LOW=0.75, HIGH=0.90
    result_image = real(backward_fft(spec_filter_fft(forward_fft(src_image),LOW,HIGH)));
end