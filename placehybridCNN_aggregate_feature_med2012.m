function placehybridCNN_aggregate_feature_med2012()


    filename='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med12/medmd_2012.mat';
    fprintf('Loading meta file <%s>\n', filename);
    load(filename, 'MEDMD');
    
    fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes/placehybridCNN.full';
    output_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes/placehybridCNN.full.video';
    feat_dim = 1183;
    
    for ii=1:length(MEDMD.clips),
    
        if ~mod(ii, 100), fprintf('%d ', ii); end;
        
        video_id = MEDMD.clips{ii};
        ldc_pat = MEDMD.info.(video_id).loc;
        
        output_file = sprintf('%s/%s/%s.mat', output_dir, ldc_pat, video_id);
        if exist(output_file, 'file'),
            fprintf('File already exist <%s> \n', output_file);
        end
        
        if ~exist(fileparts(output_file), 'file'),
            mkdir(fileparts(output_file));
        end
        
        video_kf_dir = fullfile(fea_dir, ldc_pat);
		video_kf_dir = video_kf_dir(1:end-4);	
		kfs = dir([video_kf_dir, '/*.txt']);
        num_keyframes = length(kfs);
        
        code = zeros(feat_dim, num_keyframes);
        
        for jj=1:num_keyframes,
            feat_file = sprintf('%s/%s/%s', fea_dir, ldc_pat(1:end-4), kfs(jj).name);
            fh = fopen(feat_file);
            code_ = textscan(fh, '%f');
            fclose(fh);
            code(:, jj) = code_{1};
        end
        
        code = sum(code, 2);
        save(output_file, 'code');
        
    end    

end
