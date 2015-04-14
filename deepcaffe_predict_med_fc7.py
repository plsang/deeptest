import os
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import pandas as pd
import re
import glob
from os.path import basename
import subprocess

caffe_root = '/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc' 
nc = 1000

import sys
sys.path.insert(0, caffe_root + '/python')

import caffe 

if (len(sys.argv) < 4):
    print sys.argv[0] + " <set> <start video> <end video> ";
    exit()

dataset = sys.argv[1]
start_video = int(sys.argv[2])
end_video = int(sys.argv[3])

#logging
logging.basicConfig(filename= "%s/%s.log.txt" % (__file__[:-3], dataset), format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logging.info("Start [%s]" % ' '.join(sys.argv))
    
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

keyframe_dir = '/net/per610a/export/das11f/plsang/trecvidmed13/keyframes'
output_dir = '/net/per610a/export/das11f/plsang/trecvidmed/feature/keyframes/deepcaffe_fc7'
meta_dir='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med14/'
    
def deepcaffe_extract_feature(image_file):
    try:
        input_image = caffe.io.load_image(image_file)
        scores = net.predict([input_image], oversample=True).flatten()
        feat_fc6 = net.blobs['fc6'].data[4].flatten()
        feat_fc7 = net.blobs['fc7'].data[4].flatten()
        return feat_fc6, feat_fc7

    except Exception as err:
        logger.info('Classification error: %s', err)
        return (False, 'Something went wrong when classifying the '
                       'image. Maybe try another one?')

def get_video_list_file(x):
    return {
        'event': meta_dir + 'event_video_list.txt',
        'eventbg': meta_dir + 'eventbg_list.txt',
        'kindredtest14': meta_dir + 'kindredtest14_list.txt',
        'medtest14': meta_dir + 'medtest14_list.txt',
    }[x]

def save_as_sparse_format(x, outfile):
    with open(outfile, 'w') as f:
        it = np.nditer(x, flags=['f_index'])
        while not it.finished:
            if it[0] > 0:        
                f.write(str(it.index)+':'+str(it[0])+' ')
            it.iternext()
   
        
if __name__=="__main__":
    
    #video_list_file = '/net/per610a/export/das11f/plsang/overfeat/overfeat/src/event_video_list.txt' # 6964
    
    #video_list_file='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med14/medtest14_list.txt'  # 14,708
    video_list_file = get_video_list_file(dataset)
    
    with open(video_list_file, 'r') as fh:
        data = fh.readlines()
    
    job_start = time.time()
    total_frames = 0
    for ii in range(start_video, end_video):
        
        line = data[ii];
        words = line.split()
        vid = words[0]
        ldcpat = os.path.splitext(words[1])[0]
        
        kfs = glob.glob("{}/{}/*.jpg".format(keyframe_dir, ldcpat))
        output_vdir = '{}/{}'.format(output_dir, ldcpat)
        
        if not os.path.exists(output_vdir):
            os.makedirs(output_vdir)
        
        total_frames += len(kfs)
        
        for kf_file in kfs:
            head, tail = os.path.split(kf_file)	
            kf_name = os.path.splitext(tail)[0]
            
            output_fc6_file = '{}/{}/{}.deepcaffe.fc6.txt'.format(output_dir, ldcpat, kf_name)
            output_fc7_file = '{}/{}/{}.deepcaffe.fc7.txt'.format(output_dir, ldcpat, kf_name)
            
            if os.path.exists(output_fc6_file) and os.path.exists(output_fc7_file):
                print 'Feature for [{}] already extracted'.format(kf_file);
                continue

            starttime = time.time()
            
            fc6, fc7 = deepcaffe_extract_feature(kf_file)
            
            endtime = time.time()
            print 'Extracted deep caffe feature for [{}] in {} s.'.format(kf_name, endtime-starttime)
            
            save_as_sparse_format(fc6, output_fc6_file)
            save_as_sparse_format(fc7, output_fc7_file)
            
            #np.savetxt(output_fc6_file, fc6, delimiter=',')
            #np.savetxt(output_fc7_file, fc7, delimiter=',')
        try:     
            subprocess.call(['chmod', '-R', '777', output_vdir])
        except:
            print 'Permission denied'
            
    subprocess.call(['chmod', '777', output_vdir])
    
    job_end = time.time()
    logging.info('End [%s]. Elapsed: %s. Processed: %d frames. Average: %.3f s/KF' % \
        (' '.join(sys.argv), job_end - job_start, total_frames, (job_end - job_start)/total_frames))