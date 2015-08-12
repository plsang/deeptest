function deepcaffe_extract_feature_vsd2015( model_name, partition, layer_name, numlayer, start_seg, end_seg)
	
	addpath('/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc/matlab/caffe');
    addpath('/net/per610a/export/das11f/plsang/codes/common');
	
	if nargin < 4,
        fprintf('Usage: deepcaffe_extract_feature( model_name, partition, layer_name, start_seg, end_seg ) \n');
        fprintf(' @param: model_name (caffe, places205, placeshybrid, verydeep) \n');
		fprintf(' @param: partition (devset, testset) \n');
		fprintf(' @varargin: numlayer (16, 19) (for verydeep network) \n');
		fprintf(' @varargin: start - start_seg \n');
		fprintf(' @varargin: end - end_seg \n');
        return;
    end
    
    root_meta = '/net/per920a/export/das14a/satoh-lab/plsang/vsd2015/metadata';
    
    meta_file = sprintf('%s/%s.txt', root_meta, partition);
    fh = fopen(meta_file, 'r');
    clips = textscan(fh, '%s');
    clips = clips{1};
    fclose(fh);
    
    %%%% feature dim
    if strcmp(layer_name, 'full'),
		switch model_name,
			case 'caffe'
				feat_dim = 1000;
			case 'places205'
				feat_dim = 205;
			case 'placeshybrid'
				feat_dim = 1183;
			case 'verydeep'
				feat_dim = 1000;
			case 'googlenet'
				feat_dim = 1000;	
			otherwise
				error('unknown model name <%s> \n', model_name);
		end
    else
        feat_dim = 4096;
    end
    
	%%%%% Load shot info
    if ~exist('numlayer', 'var'), numlayer = 0; end;
	
    if ~exist('start_seg', 'var') || start_seg < 1, start_seg = 1; end;
    
    if ~exist('end_seg', 'var') || end_seg > length(clips), end_seg = length(clips); end;
    
    %tic
	
	
	proj_dir = '/net/per920a/export/das14a/satoh-lab/plsang/vsd2015';
    kf_dir = sprintf('%s/keyframes/%s', proj_dir, partition);
    
	feature_ext = sprintf('%s.%s', model_name, layer_name);
	if strcmp(model_name, 'verydeep'),
		feature_ext = sprintf('%s.%s.l%d', model_name, layer_name, numlayer);
	else
		feature_ext = sprintf('%s.%s', model_name, layer_name);
	end
	
	output_dir = sprintf('/net/per920a/export/das14a/satoh-lab/plsang/vsd2015/feature/%s/%s', feature_ext, partition);
    if ~exist(output_dir, 'file'),
		mkdir(output_dir);
        change_perm(output_dir, 777, 0);
    end
    
    for ii = start_seg:end_seg,
        video_id = clips{ii}(1:end-4);                 
		     
		video_kf_dir = fullfile(kf_dir, video_id);
		
		kfs = dir([video_kf_dir, '/*.jpg']);
		
		fprintf(' [%d --> %d --> %d] Extracting & encoding for [%s - %d kfs]...\n', start_seg, ii, end_seg, video_id, length(kfs));
		
        output_file = sprintf('%s/%s.mat', output_dir, video_id);
        if exist(output_file, 'file'),
            continue;
        end
        
        %output_kf_dir = fileparts(output_file);
        %if ~exist(output_kf_dir), mkdir(output_kf_dir); end;
        
        code = zeros(feat_dim, length(kfs), 'single');
		for jj = 1:length(kfs),
			img_name = kfs(jj).name;
			img_path = fullfile(video_kf_dir, img_name);
            try
				if strcmp(model_name, 'verydeep'),
					code_ = matcaffe_extract_feature(model_name, img_path, layer_name, 'numlayer', numlayer);
				else
					code_ = matcaffe_extract_feature(model_name, img_path, layer_name);
				end
            catch
				fprintf('Error extracting %s \n', img_path);
                continue;
            end
            code(:, jj) = code_;
		end 
        
        sge_save(output_file, code);
        %save(output_file, 'code', '-v7.3');
    end
    	
    %toc
    quit;

end