function downscale_matlab(input_path, output_path, scale)
% Generate downscaled images
%
% input_path - Base path of the input images.
% output_path - Base path of the output (downscaled) images.
% scale - Downscaling factor.

if (~exist(output_path, 'dir'))
  mkdir(output_path);
end

image_list = dir([input_path, '/*.png']);
num_images = length(image_list);

for i = 1:num_images
  image_name = image_list(i).name;
  image_input_path = fullfile([input_path, '/', image_name]);
  image_output_path = fullfile([output_path, '/', image_name]);

  fprintf([num2str(i), '/', num2str(num_images), ', ', image_name, '\n']);

  image = imread(image_input_path);
  image = imresize(image, 1.0/scale);
  imwrite(image, image_output_path);
end

