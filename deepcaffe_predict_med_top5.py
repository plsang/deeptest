import os
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import pandas as pd
import re
import glob
from os.path import basename
import time
import subprocess

caffe_root = '/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc' 
import sys
sys.path.insert(0, caffe_root + '/python')

import caffe 

if (len(sys.argv) < 3):
	print sys.argv[0] + " <start video> <end video> ";
	exit();

start_video = int(sys.argv[1])
end_video = int(sys.argv[2])

REPO_DIRNAME = caffe_root
						   
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)
PRETRAINED = '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

net.set_phase_test()
net.set_mode_cpu()

default_args = {
	'model_def_file': (
		'{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
	'pretrained_model_file': (
		'{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
	'mean_file': (
		'{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
	'class_labels_file': (
		'{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
}
for key, val in default_args.iteritems():
	if not os.path.exists(val):
		raise Exception(
			"File for {} is missing. Should be at: {}".format(key, val))
			
default_args['image_dim'] = 227
default_args['raw_scale'] = 255.
default_args['gpu_mode'] = False

with open(default_args['class_labels_file']) as f:
	labels_df = pd.DataFrame([
		{
			'synset_id': l.strip().split(' ')[0],
			'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
		}
		for l in f.readlines()
	])
labels = labels_df.sort('synset_id')['name'].values

# dataset = 'youcook'
keyframe_dir = '/net/per610a/export/das11f/plsang/trecvidmed13/keyframes';	
output_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes/deepcaffe';
	
def classify_image(image_file):
	try:
		input_image = caffe.io.load_image(image_file)
		scores = net.predict([input_image], oversample=True).flatten()
				
		indices = (-scores).argsort()[:5]
		predictions = labels[indices]

		meta = [
			(p, '%.5f' % scores[i])
			for i, p in zip(indices, predictions)
		]
		#logging.info('result: %s', str(meta))
				
		return meta;

	except Exception as err:
		#logging.info('Classification error: %s', err)
		return (False, 'Something went wrong when classifying the '
					   'image. Maybe try another one?')


if __name__=="__main__":
	
	#videos = [vid for vid in os.listdir(keyframe_dir) if re.search('\d{4}', vid)]
	#videos = [vid for vid in os.listdir(keyframe_dir) if re.search('vid\d+', vid)]
	# videos = [vid for vid in os.listdir(keyframe_dir) if re.search('HVC\d+', vid)]
	#video_list_file = '/net/per610a/export/das11f/plsang/overfeat/overfeat/src/event_video_list.txt'
	
	video_list_file='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med14/medtest14_list.txt'  # 14,708
	
	with open(video_list_file, 'r') as fh:
		data = fh.readlines()
	
	for ii in range(start_video, end_video):
		
		line = data[ii];
		words = line.split()
		vid = words[0]
		ldcpat = os.path.splitext(words[1])[0]
		
		kfs = glob.glob("{}/{}/*.jpg".format(keyframe_dir, ldcpat))
		output_vdir = '{}/{}'.format(output_dir, ldcpat)
		
		if not os.path.exists(output_vdir):
			os.makedirs(output_vdir)
		
		for kf_file in kfs:
			head, tail = os.path.split(kf_file)	
			kf_name = os.path.splitext(tail)[0]
			
			output_file = '{}/{}/{}.deepcaffe.txt'.format(output_dir, ldcpat, kf_name)
			if os.path.exists(output_file):
				print 'File [{}] already exists'.format(output_file);
				continue

			starttime = time.time()
			preds = classify_image(kf_file)
			endtime = time.time()
			print 'Extracted deep caffe feature for [{}] in {} s.'.format(kf_name, endtime-starttime)
			
			with open(output_file, 'w') as fh:
				for (x, y) in preds:
					fh.write('{}:{}\n'.format(x, y))
					
			#os.chmod(output_file, 0777)
		subprocess.call(['chmod', '-R', '777', output_vdir])
			