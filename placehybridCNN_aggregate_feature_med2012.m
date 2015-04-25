function placehybridCNN_aggregate_feature_med2012(feat_name, dcnn_layer, feat_dim, feat_fmt)

    %% Function to aggregate keyframe feature to video features
    %% @param: feat_name = 'placehybridCNN';
    %% @param: dcnn_layer = 'full';
    %% @param: feat_dim = 1183;
    %% @feat_fmt: feat_fmt = 'sparse', 'full'
    
    filename='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med12/medmd_2012.mat';
    fprintf('Loading meta file <%s>\n', filename);
    load(filename, 'MEDMD');
    
    feat_name = 'placehybridCNN';
    dcnn_layer = 'full';
    feat_pat = sprintf('%s.%s', feat_name, dcnn_layer);
    
    
    feat_root_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes';
    feat_dir = sprintf('%s/%s', fea_root_dir, feat_pat);
    
    output_root_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes2video';
    output_dir = sprintf('%s/%s', output_root_dir, feat_pat);
    if ~exist(output_dir, 'file'), mkdir(output_dir); end;
    
    
    for ii=1:length(MEDMD.clips),
    
        if ~mod(ii, 100), fprintf('%d ', ii); end;
        
        video_id = MEDMD.clips{ii};
        ldc_pat = MEDMD.info.(video_id).loc;
        
        output_file = sprintf('%s/%s.mat', output_dir, ldc_pat(1:end-4));
        if exist(output_file, 'file'),
            fprintf('File already exist <%s> \n', output_file);
        end
        
        if ~exist(fileparts(output_file), 'file'),
            mkdir(fileparts(output_file));
        end
        
        video_kf_dir = fullfile(feat_dir, ldc_pat);
		video_kf_dir = video_kf_dir(1:end-4);	
		kfs = dir([video_kf_dir, '/*.txt']);
        num_keyframes = length(kfs);
        
        code = zeros(feat_dim, num_keyframes);
        
        for jj=1:num_keyframes,
            feat_file = sprintf('%s/%s/%s', feat_dir, ldc_pat(1:end-4), kfs(jj).name);
            fh = fopen(feat_file);
            if strcmp(feat_fmt, 'full'),
                code_ = textscan(fh, '%f');
                code(:, jj) = code_{1};
            elseif strcmp(feat_fmt, 'sparse'),    %% sparse
                code_ = textscan(fh, '%d:%f');
                code(code_{1}, jj) = code_{2};
            else                
                error('unknown feature format\n');
            end
            fclose(fh);
        end
        
        code = sum(code, 2);
        
        save(output_file, 'code');
        
    end    

end
