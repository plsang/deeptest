function deepcaffe_extract_feature( model_name, dataset, pat_list, layer_name, numlayer, start_seg, end_seg)
	
	addpath('/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc/matlab/caffe');
    addpath('/net/per610a/export/das11f/plsang/codes/common');
	
	if nargin < 4,
        fprintf('Usage: deepcaffe_extract_feature( model_name, dataset, pat_list, layer_name, start_seg, end_seg ) \n');
        fprintf(' @param: model_name (caffe, places205, placeshybrid, verydeep) \n');
		fprintf(' @param: pat_list (ek100ps14, ek10ps14, bg, kindred14, medtest14, train12, test12, train14) \n');
        fprintf(' @param: layer_name (fc6, fc7, full) \n');
		fprintf(' @varargin: numlayer (16, 19) (for verydeep network) \n');
		fprintf(' @varargin: start - start_seg \n');
		fprintf(' @varargin: end - end_seg \n');
        return;
    end
    	
    fprintf('Loading metadata for dataset <%s> \n', dataset);
	%%%%% Load shot info
    if strcmp(dataset, 'med2012'),
        meta_file = '/net/per610a/export/das11f/plsang/trecvidmed/metadata/med12/medmd_2012.mat';
        load(meta_file);
    elseif strcmp(dataset, 'med2014'),
        meta_file = '/net/per610a/export/das11f/plsang/trecvidmed14/metadata/medmd_2014_devel_ps.mat';
        load(meta_file);
    elseif strcmp(dataset, 'med15eval'),
        meta_file = '/net/per610a/export/das11f/plsang/trecvidmed/metadata/med15/med15_eval.mat';
        load(meta_file);
    elseif strcmp(dataset, 'med15ah'),
        meta_file = '/net/per610a/export/das11f/plsang/trecvidmed/metadata/med15/med15_ah.mat';   
        load(meta_file);    
    elseif strcmp(dataset, 'med2011'),
        meta_file = '/net/per610a/export/das11f/plsang/trecvidmed/metadata/med11/medmd_2011.mat';
        load(meta_file);        
    else
        error('unknown dataset <%s> \n', dataset);
    end
    
	supported_pat_list = {'ek100ps14', 'ek10ps14', 'bg', 'kindred14', 'medtest14', 'train12', 'test12', 'train14', 'med15eval', 'med15ah', 'med2011'};
	
	    clips = []; 
    durations = [];
    for supported_pat = supported_pat_list,
        if ~isempty(strfind(pat_list, supported_pat{:})),
            switch supported_pat{:},
                case 'bg'
                    clips_ = MEDMD.EventBG.default.clips;
                    durations_ = MEDMD.EventBG.default.durations;
                case 'kindred14'
                    clips_ = MEDMD.RefTest.KINDREDTEST.clips;
                    durations_ = MEDMD.RefTest.KINDREDTEST.durations;
                case 'medtest14'
                    clips_ = MEDMD.RefTest.MEDTEST.clips;
                    durations_ = MEDMD.RefTest.MEDTEST.durations;
                case 'ek100ps14'
                    clips_ = MEDMD.EventKit.EK100Ex.clips;
                    durations_ = MEDMD.EventKit.EK100Ex.durations;
                    %% only select e20-e40
                    sel_idx = zeros(1, length(clips_));
                    for ii=21:40,
                        event_id = sprintf('E%03d', ii);
                        sel_idx = sel_idx | ismember(clips_, MEDMD.EventKit.EK100Ex.judge.(event_id).positive);
                        sel_idx = sel_idx | ismember(clips_, MEDMD.EventKit.EK100Ex.judge.(event_id).miss);
                    end
                    clips_ = clips_(sel_idx);
                    durations_ = durations_(sel_idx);
                case 'ek10ps14'
                    clips_ = MEDMD.EventKit.EK10Ex.clips;
                    durations_ = MEDMD.EventKit.EK10Ex.durations;
                    %% only select e20-e40
                    sel_idx = zeros(1, length(clips_));
                    for ii=21:40,
                        event_id = sprintf('E%03d', ii);
                        sel_idx = sel_idx | ismember(clips_, MEDMD.EventKit.EK10Ex.judge.(event_id).positive);
                        sel_idx = sel_idx | ismember(clips_, MEDMD.EventKit.EK10Ex.judge.(event_id).miss);
                    end
                    clips_ = clips_(sel_idx);
                    durations_ = durations_(sel_idx);    
                case 'train12'
                    clips_ = MEDMD.Train.clips;
                    durations_ = MEDMD.Train.durations;
                
                case 'test12'
                    clips_ = MEDMD.Test.clips;
                    durations_ = MEDMD.Test.durations;
                    
                case 'train14'
                    clips_ = MEDMD.videos;
                    durations_ = zeros(1, length(clips_));
                    for ii=1:length(clips_),
                        clip_id = clips_{ii};
                        if isfield(MEDMD.info, clip_id),
                            durations_(ii) = MEDMD.info.(clip_id).duration;
                        end
                    end
                case 'med15eval'
                    clips_ =  MEDMD.UnrefTest.MED15EvalFull.clips;
					durations_ = MEDMD.UnrefTest.MED15EvalFull.durations;
                case 'med15ah'    
                    clips_ =  MEDMD.EventKit.EK10Ex.clips;
                    durations_ = MEDMD.EventKit.EK10Ex.durations;    
                case 'med2011'
                    clips_ = [MEDMD.EventKit.EK130Ex.clips, MEDMD.RefTest.MED11TEST.clips];
                    durations_ = [MEDMD.EventKit.EK130Ex.durations, MEDMD.RefTest.MED11TEST.durations];
            end
            
            clips = [clips, clips_];
            durations = [durations, durations_];
        end
    end
	
	if ~isempty(strfind(pat_list, '--count')),
        fprintf('Total clips: %d \n', length(clips));
        fprintf('Total durations: %.3f hours \n', sum(durations)/3600 );
        return;
    end
    
    [durations, sorted_idx] = sort(durations, 'descend');
    clips = clips(sorted_idx);
	
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
	
	
	proj_dir = '/net/per610a/export/das11f/plsang/trecvidmed';
    kf_dir = sprintf('%s/keyframes/', proj_dir);
    
	feature_ext = sprintf('%s.%s', model_name, layer_name);
	if strcmp(model_name, 'verydeep'),
		feature_ext = sprintf('%s.%s.l%d', model_name, layer_name, numlayer);
	else
		feature_ext = sprintf('%s.%s', model_name, layer_name);
	end
	
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