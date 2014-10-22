function gen_sge_code(script_name, pattern, total_segments, num_job, pe)
	
	script_dir = '/net/per610a/export/das11f/plsang/codes/deeptest';
	
	sge_sh_file = sprintf('%s/%s.sh', script_dir, script_name);
	
	
	[file_dir, file_name] = fileparts(sge_sh_file);
	output_dir = [script_dir, '/', script_name];

	if exist(output_dir, 'file') ~= 7,
		mkdir(output_dir);
	end
	

	output_file = sprintf('%s/%s.qsub.sh', output_dir, file_name);
	fh = fopen(output_file, 'w');
	
	%num_max = 200;
	num_max = 0;
	
	% first 50 videos, 1 vidoe/job
	num_per_job = 1;
	start_num = 0;
	for ii = 1:num_max,	
		start_idx = start_num + (ii-1)*num_per_job;
		%end_idx = start_num + ii*num_per_job - 1;
		end_idx = start_num + ii*num_per_job;
		
		if(end_idx > total_segments)
			end_idx = total_segments;
		end
		
		params = sprintf(pattern, start_idx, end_idx);
		if ~exist('pe', 'var'),
			fprintf(fh, 'qsub -e /dev/null -o /dev/null %s %s\n', sge_sh_file, params);
		else
			fprintf(fh, 'qsub -pe localslots %d -e /dev/null -o /dev/null %s %s\n', pe, sge_sh_file, params);
		end
		
		if end_idx == total_segments, break; end;
	end
	
	num_per_job = ceil((total_segments - start_num + 1)/num_job);	
	
	start_num = num_max;
	for ii = 1:num_job,
		start_idx = start_num + (ii-1)*num_per_job;
		%end_idx = start_num + ii*num_per_job - 1;
		end_idx = start_num + ii*num_per_job;
		
		if(end_idx > total_segments)
			end_idx = total_segments;
		end
		
		params = sprintf(pattern, start_idx, end_idx);
		if ~exist('pe', 'var'),
			fprintf(fh, 'qsub -e /dev/null -o /dev/null %s %s\n', sge_sh_file, params);
		else
			fprintf(fh, 'qsub -pe localslots %d -e /dev/null -o /dev/null %s %s\n', pe, sge_sh_file, params);
		end
		
		if end_idx == total_segments, break; end;
	end
	
	cmd = sprintf('chmod +x %s', output_file);
	system(cmd);
	
	fclose(fh);
end