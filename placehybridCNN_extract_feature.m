function placehybridCNN_extract_feature( dataset, layer_name, start_seg, end_seg )

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
    
	output_dir = sprintf('/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes/%s', feature_ext);
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
       
		%% update Jul 5, 2013: support segment-based
		
		fprintf(' [%d --> %d --> %d] Extracting & encoding for [%s - %d kfs]...\n', start_seg, ii, end_seg, video_id, length(kfs));
		
		for jj = 1:length(kfs),
        
			img_name = kfs(jj).name;
			img_path = fullfile(video_kf_dir, img_name);
			
            output_file = sprintf('%s/%s/%s/%s.txt', output_dir, fileparts(MEDMD.info.(video_id).loc), video_id, img_name(1:end-4));
            if exist(output_file, 'file'),
                continue;
            end
            
            output_kf_dir = fileparts(output_file);
            if ~exist(output_kf_dir), mkdir(output_kf_dir); end;
            
            %try
                code = matcaffe_extract_place205_hybrid(img_path, layer_name);
                fh = fopen(output_file, 'w');
                if strcmp(layer_name, 'full'),
                    fprintf(fh, '%f\n', code);
                else
                    non_zero_idx = find(code ~= 0);
                    fprintf(fh, '%d:%f ', [non_zero_idx, code(non_zero_idx)]');
                end
                fclose(fh);
            % catch
                % warning('Error while extracting feature for image [%s]!!\n', img_path);
				% continue;
            % end
			
		end 
        
    end
    	
    %toc
    quit;

end