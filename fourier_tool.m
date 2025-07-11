function [fft_image_data] = fourier_tool(image_data,channel)
    % View an image in the fourier domain

    % images are 325 x 435 x (RGB) uint8
    % convert image to double
    I = im2double(image_data); % expect (Y * X * C) matrix
    C = channel; % expect int8 <= C

    fft_image_data = zeros(size(I)); % return (Y * X * 1) matrix

    if size(I,3) < C
       throw(MException('Image:noSuchChannel', ...
        'Image has no channel %d',C));
    end

    % r, g, b, or grayscale
    test_image_channels = imsplit(I);
    
    % fourier domains, recenter with fftshift
    test_imd_gs_fft = fftshift(fft2(test_image_channels(C)));
    test_imd_gs_lafft = log(abs(test_imd_gs_fft));
    % normalize to [0,1]
    test_sel = test_imd_gs_lafft;
    test_sel = test_sel - min(test_sel(:));
    test_sel = test_sel ./ max(test_sel(:)); % image range is [0,1]

    fft_image_data = test_sel;
    fft_image_data
end