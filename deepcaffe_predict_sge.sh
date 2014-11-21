# Written by Sang Phan
# Last update Sep 29, 2014
#!/bin/sh
# Force to use shell sh. Note that #$ is SGE command
#$ -S /bin/sh
# Force to limit hosts running jobs
#$ -q all.q@@bc2hosts,all.q@@bc3hosts
# Log starting time
date 
# for opencv shared lib
PATH=/net/per610a/export/das11f/plsang/usr/bin:/net/per900a/raid0/plsang/software/gcc-4.8.1/release/bin:/net/per900a/raid0/plsang/software/ffmpeg-2.0/release-shared/bin:/net/per900a/raid0/plsang/software/scala/bin:/net/per900a/raid0/plsang/usr.local/bin:/net/per900a/raid0/plsang/software/openmpi-1.6.5/release-shared/bin:$PATH
LD_LIBRARY_PATH=/net/per610a/export/das11f/plsang/usr/lib:/net/per900a/raid0/plsang/usr.local/lib64:/net/per900a/raid0/plsang/software/ffmpeg-2.0/release-shared/lib:/net/per900a/raid0/plsang/software/gcc-4.8.1/release/lib:/net/per900a/raid0/plsang/software/opencv-2.4.6.1/release/lib:/net/per900a/raid0/plsang/usr.local/lib:/net/per900a/raid0/plsang/software/openmpi-1.6.5/release-shared/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PATH
export LD_LIBRARY_PATH

# Log info of the job to output file  *** CHANGED ***
echo [$HOSTNAME] [$JOB_ID] [python deepcaffe_predict.py $1 $2 $3]
# change to the code dir  --> NEW!!! *** CHANGED ***
cd /net/per610a/export/das11f/plsang/codes/deeptest
# Log info of current dir
pwd
# Command - -->  must use " (double quote) for $2 because it contains a string  --- *** CHANGED ***
# LD_PRELOAD="/net/per900a/raid0/plsang/usr.local/lib/libstdc++.so.6" matlab -nodisplay -r "densetraj_encode_sge( '$1', '$2', '$3', $4, $5 )"
python /net/per610a/export/das11f/plsang/codes/deeptest/deepcaffe_predict.py $1 $2 $3
# Log ending time
date

