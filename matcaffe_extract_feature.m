function [scores, maxlabel] = matcaffe_extract_feature(model_name, im_path, layer_name, varargin)
%% Author: Sang Phan
%% Jan 27, 2015
%% layer_name: fc6 (4096d), fc7 (4096d), full (1000d)
%% model_name: default by DeepCaffe, Places205, PlacesHybrid, VeryDeep
%% varargin: 'numlayer' for verydeep model
% scores = matcaffe_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 

use_gpu = 0;


for k=1:2:length(varargin),

	opt = lower(varargin{k});
	arg = varargin{k+1};
	
	switch opt
		case 'numlayer'
			numlayer = arg;
		otherwise
			error(sprintf('Option ''%s'' unknown.', opt)) ;
	end  
end

switch model_name,
	case 'caffe'
		model_def_dir = '/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc/models/bvlc_reference_caffenet/';
		model_def_file = sprintf('%s/__deploy_matlab.%s.prototxt', model_def_dir, layer_name);
		model_file = '/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
	case 'places205'
		model_def_dir = '/net/per920a/export/das14a/satoh-lab/plsang/places205/placesCNN_upgraded';
		model_def_file = sprintf('%s/__places205CNN_deploy_upgraded.%s.prototxt', model_def_dir, layer_name);
		model_file = sprintf('%s/%s', model_def_dir, 'places205CNN_iter_300000_upgraded.caffemodel');
	case 'placeshybrid'
		model_def_dir = '/net/per920a/export/das14a/satoh-lab/plsang/places205/hybridCNN_upgraded';
		model_def_file = sprintf('%s/__hybridCNN_deploy_upgraded.%s.prototxt', model_def_dir, layer_name);
		model_file = sprintf('%s/%s', model_def_dir, 'hybridCNN_iter_700000_upgraded.caffemodel');
	case 'verydeep'
		model_def_dir = '/net/per920a/export/das14a/satoh-lab/plsang/very_deep/caffe';
		model_def_file = sprintf('%s/VGG_ILSVRC_%d_layers_deploy.%s.prototxt', model_def_dir, numlayer, layer_name);
		model_file = sprintf('%s/VGG_ILSVRC_%d_layers.caffemodel', model_def_dir, numlayer);
	otherwise
		error('unknown model name <%s> \n', model_name);
end

matcaffe_init(use_gpu, model_def_file, model_file);

im = imread(im_path);

% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {prepare_image(im, model_name)};
toc;

% do forward pass to get scores
% scores are now Width x Height x Channels x Num
tic;
scores = caffe('forward', input_data);
toc;

scores = scores{1};
size(scores);
scores = squeeze(scores);
scores = mean(scores,2);

[~,maxlabel] = max(scores);

% ------------------------------------------------------------------------
function images = prepare_image(im, model_name)
% ------------------------------------------------------------------------
d = load('/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc/matlab/caffe/ilsvrc_2012_mean.mat');
IMAGE_MEAN = d.image_mean;
IMAGE_DIM = 256;

switch model_name,
	case 'verydeep'
		CROPPED_DIM = 224;
	otherwise
		CROPPED_DIM = 227;
end

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]) - IMAGE_MEAN;

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
curr = 1;
for i = indices
  for j = indices
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
center = floor(indices(2) / 2)+1;
images(:,:,:,5) = ...
    permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);
