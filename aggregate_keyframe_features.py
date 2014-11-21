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

class_labels_file = '/net/per610a/export/das11f/plsang/deepcaffe/caffe-rc/data/ilsvrc12/synset_words.txt'
class_labels_file = '/net/per610a/export/das11f/plsang/youcook/feature/keyframes_overfeat_1000/0002/0002-000010.1000overfeat.pred.txt'

with open(class_labels_file) as f:
	labels_df = ([{
			' '.join(l.strip().split(' ')[0:-1]): l.strip().split(' ')[-1]
			#'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
			#'name': 
		}
		for l in f.readlines()
	])

print labels_df
