from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm
from torch.autograd import Variable
from generator import ca2_Generator
from utils import *
import pdb 
import cv2 
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms

generator_path = './TANet_model.pkl'
Generator = ca2_Generator()

Generator.load_state_dict(torch.load(generator_path))
if torch.cuda.is_available():
    Generator.cuda()

def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

counter = 0
start_time = time.time()

# pdb.set_trace() 
to_tensor = transforms.ToTensor() 	# Transforms 0-255 numbers to 0 - 1.0. 


dataset = '/DATA/wangxiao/got10k_test/'
videoList = os.listdir(dataset)


for i in range(len(videoList)): 
	videoName = videoList[i] 
	print("==>> process the video ", i, " ==>> videoName: ", videoName) 
	targetObject_path = dataset + videoName +'/00000001.jpg'
	init_image = cv2.imread(targetObject_path)
	gt_name = '/groundtruth.txt'
	f = open(dataset + videoName + gt_name)
	for line in f.readlines():
		init_bbox = [float(i) for i in line.split(',')]
		init_bbox = np.array(init_bbox)

	validation_targetObject = init_image[int(init_bbox[1]):int(init_bbox[1]+init_bbox[3]), int(init_bbox[0]):int(init_bbox[0]+init_bbox[2]), :] 
	validation_targetObject = cv2.resize(validation_targetObject, (300, 300))
	validation_targetObject = torch.unsqueeze(to_tensor(validation_targetObject), dim=0) 

	img_list_ = sorted([p for p in os.listdir(dataset + videoName + '/') if os.path.splitext(p)[1] == '.jpg'])
	img_list = np.array([dataset + videoName +'/' + img for img in img_list_])

	DIR_TO_SAVE = '/home/wangxiao/attentionMaps_got10k/'+ videoName + "/"
	if not os.path.exists(DIR_TO_SAVE):
		os.makedirs(DIR_TO_SAVE)

	count = 0 
	clip_len = 3 




	for j in range(1, len(img_list)-2): 

		count = count + 1 
		validation_sample = cv2.imread(img_list[j]) 
		validation_sample = cv2.resize(validation_sample, (300, 300)) 
		batch_img 		  = torch.unsqueeze(to_tensor(validation_sample), dim=0) 


		validation_sample_Prev = cv2.imread(img_list[j-1]) 
		validation_sample_Prev = cv2.resize(validation_sample_Prev, (300, 300)) 
		batch_img_Prev 		   = torch.unsqueeze(to_tensor(validation_sample_Prev), dim=0) 


		validation_sample_Subs = cv2.imread(img_list[j+1]) 
		validation_sample_Subs = cv2.resize(validation_sample_Subs, (300, 300)) 
		batch_img_Subs 		   = torch.unsqueeze(to_tensor(validation_sample_Subs), dim=0) 


		batch_imgClip = torch.zeros(1, clip_len, 3, 300, 300) 

		batch_imgClip[0, 0] = batch_img_Prev
		batch_imgClip[0, 1] = batch_img 
		batch_imgClip[0, 2] = batch_img_Subs

		# pdb.set_trace() 

		out = Generator(batch_img.cuda(), validation_targetObject.cuda(), batch_imgClip.cuda())
		
		# out = nn.functional.interpolate(out, size=[init_image.shape[0], init_image.shape[1]]) 
		ga_map = out.cpu().data.squeeze(0)

		pilTrans = transforms.ToPILImage()
		pilImg = pilTrans(ga_map)
		pilImg.save(DIR_TO_SAVE + str(j) + "_attentionMap.png") 
