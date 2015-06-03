function placehybridCNN_extract_feature_alpha( dataset, layer_name, start_seg, end_seg )

    addpath('/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc/matlab/caffe');
	
    % encoding type
    enc_type = 'placehybridCNN';
	
    feature_ext = sprintf('%s.%s', enc_type, layer_name);
	
    fprintf('Loading metadata for dataset <%s> \n', dataset);
	%%%%% Load shot info
    if strcmp(dataset, 'med2012'),
        meta_file = '/net/per610a/export/das11f/plsang/trecvidmed/metadata/med12/medmd_2012.mat';
        load(meta_file);
        clips = MEDMD.clips;
    elseif strcmp(dataset, 'med2014'),
        meta_file = '/net/per610a/export/das11f/plsang/trecvidmed14/metadata/medmd_2014_devel_ps.mat';
        load(meta_file);
        clips = MEDMD.videos;
    else
        error('unknown dataset <%s> \n', dataset);
    end
    
    %%%% feature dim
    if strcmp(layer_name, 'full'),
        feat_dim = 1183;
    else
        feat_dim = 4096;
    end
    
	%%%%% Load shot info
    
    if ~exist('start_seg', 'var') || start_seg < 1,
        start_seg = 1;
    end
    
    if ~exist('end_seg', 'var') || end_seg > length(clips),
        end_seg = length(clips);
    end
    
    %tic
	
	proj_dir = '/net/per610a/export/das11f/plsang/trecvidmed';
    kf_dir = sprintf('%s/keyframes/', proj_dir);
    
	output_dir = sprintf('/net/per920a/export/das14a/satoh-lab/plsang/trecvidmed/feature/keyframes/%s', feature_ext);
    if ~exist(output_dir, 'file'),
		mkdir(output_dir);
    end
    
    for ii = start_seg:end_seg,
        video_id = clips{ii};                 
		
        if ~isfield(MEDMD.info, video_id),
            fprintf('could not look up for video <%s> \n', video_id);
            continue;
        end
            
		video_kf_dir = fullfile(kf_dir, fullfile(fileparts(MEDMD.info.(video_id).loc), video_id));
		
		kfs = dir([video_kf_dir, '/*.jpg']);
		
		fprintf(' [%d --> %d --> %d] Extracting & encoding for [%s - %d kfs]...\n', start_seg, ii, end_seg, video_id, length(kfs));
		
        output_file = sprintf('%s/%s/%s.mat', output_dir, fileparts(MEDMD.info.(video_id).loc), video_id);
        if exist(output_file, 'file'),
            continue;
        end
        
        output_kf_dir = fileparts(output_file);
        if ~exist(output_kf_dir), mkdir(output_kf_dir); end;
        
        code = zeros(feat_dim, length(kfs), 'single');
		for jj = 1:length(kfs),
			img_name = kfs(jj).name;
			img_path = fullfile(video_kf_dir, img_name);
            %try
                code_ = matcaffe_extract_place205_hybrid(img_path, layer_name);
            %catch
            %    continue;
            %end
            code(:, jj) = code_;
		end 
        
        save(output_file, 'code', '-v7.3');
    end
    	
    %toc
    quit;

end