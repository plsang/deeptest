# Written by Sang Phan - plsang@nii.ac.jp
# Last update Sep 30, 2014
#!/bin/sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <start_video> <end_video>" >&2
  exit 1
fi


#video_list=/net/per610a/export/das11f/plsang/overfeat/overfeat/src/event_video_list.txt
#video_list=/net/per610a/export/das11f/plsang/trecvidmed/metadata/med14/kindredtest14_list.txt
video_list=/net/per610a/export/das11f/plsang/trecvidmed/metadata/med14/medtest14_list.txt
root_dir=/net/per610a/export/das11f/plsang
tmp_dir=/tmp

kf_dir=/net/per610a/export/das11f/plsang/trecvidmed13/keyframes
feat_dir=/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes/overfeat
 
count=0
while read line; do 
	if [ "$count" -ge $1 ] && [ "$count" -lt $2 ]; then
		#splits into two string, delimiter is ' '
		IFS=' ' read -ra ADDR <<< "$line"
		#ori_name="${ADDR[0]}"
		#new_name="${ADDR[1]}"
		clip_id=$(echo "${ADDR[0]}" | tr -d '\r')
		clip_path=$(echo "${ADDR[1]}" | tr -d '\r')
		
		ldc_pat="${clip_path%.*}"
		
		od=$feat_dir/$ldc_pat
		if [ ! -d $od ]
		then
			mkdir -p $od
		fi
		
		kf_vid_dir=$kf_dir/$ldc_pat	
		echo $kf_vid_dir
		echo [$count] "Extracting feature for video {$clip_id}..."
		
		for f in `find $kf_vid_dir -type f -name "*.jpg"`
		do
			fp="${f%}" 		#get file path
			fn="${fp##*/}"	#get file name with extension
			kfid="${fn%.*}"	#get file name without extension (image id)
			
			tof=$tmp_dir/$kfid.overfeat.txt
			of=$od/$kfid.overfeat.txt
			
			if [ ! -f $of ]
			then
				python /net/per610a/export/das11f/plsang/overfeat/overfeat/src/overfeat -n 5 $f > $tof
				mv $tof $of
			fi
		done
		
		chmod -R 777 $od
	fi
	let count++; 
	#count=$((count+1))
done < $video_list


