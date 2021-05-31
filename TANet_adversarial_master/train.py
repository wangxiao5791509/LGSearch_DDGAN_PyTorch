from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from torch.autograd import Variable
from generator import ca2_Generator, App_Discriminator, Motion_Discriminator 
from utils import *
import pdb
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
import random

batch_size = 1 

attention_path = "/home/wangxiao/targetAttention_train_dataset/"

generator = ca2_Generator()
app_Dis_Net = App_Discriminator() 
mot_Dis_Net = Motion_Discriminator() 
print(generator) 
print(app_Dis_Net) 
print(mot_Dis_Net) 


if torch.cuda.is_available():
    generator.cuda()
    app_Dis_Net.cuda() 
    mot_Dis_Net.cuda() 


BCEcriterion = nn.BCELoss()


lrG 	= 1e-4 
lrD_app = 1e-4 
lrD_mot = 1e-4 

g_optim     = torch.optim.Adagrad(generator.parameters(),   lr=lrG)
d_optim_app = torch.optim.Adagrad(app_Dis_Net.parameters(), lr=lrD_app)
d_optim_mot = torch.optim.Adagrad(mot_Dis_Net.parameters(), lr=lrD_mot)



num_epoch = 20  


def to_variable(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)


start_time = time.time()
DIR_TO_SAVE = "./generator_output/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)

generator.train()
app_Dis_Net.train() 
mot_Dis_Net.train() 

real_labels = torch.ones(batch_size, 1)
fake_labels = torch.zeros(batch_size, 1)
real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda() 

loss_function = nn.BCELoss()

video_files = os.listdir(attention_path) 
random.shuffle(video_files)
# video_files = video_files[:500]

count = 0 
g_loss = 0 

loss_motD = 0 
loss_motD_fake = 0 
loss_motD_real = 0 

for current_epoch in range(num_epoch):
    g_cost_avg = 0 
    n_updates = 1  
    
    for videoidx in range(len(video_files)): 
        videoName = video_files[videoidx] 

        dataset_img_path = attention_path + videoName + "/image/" 
        dataset_img_files = os.listdir(dataset_img_path)

        dataset_mask_path = attention_path + videoName + "/mask/" 
        dataset_tarObject_path = attention_path + videoName + "/tarObject/" 


        numBatches = len(dataset_img_files) / batch_size 
        cursor = 0


        # pdb.set_trace() 
        for idx in range(int(numBatches)):

            size = len(dataset_img_files) 

            if cursor + batch_size > size:
                cursor = 0
                np.sort(dataset_img_files)
                
            batch_img = torch.zeros(batch_size, 3, 300, 300)
            batch_map = torch.zeros(batch_size, 1, 300, 300)
            targetObject_img = torch.zeros(batch_size, 3, 300, 300)

            clip_len = 3 
            batch_imgClip = torch.zeros(batch_size, clip_len, 3, 300, 300) 
            batch_imgClip_mot = torch.zeros(batch_size, clip_len+2, 3, 300, 300) 
            batch_maskClip = torch.zeros(batch_size, clip_len, 3, 300, 300) 
            batch_maskClip_mot = torch.zeros(batch_size, clip_len, 3, 300, 300) 

                     
            to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0. 
            G_times = 0 

            for batchidx in range(batch_size):

                #### initialize continuous 3 images 
                if cursor < 1: 
                    prev_file = dataset_img_files[cursor]
                else: 
                    prev_file = dataset_img_files[cursor-1]

                curr_file = dataset_img_files[cursor]

                if cursor+1 >= size: 
                    late_file = dataset_img_files[size-1]
                else: 
                    late_file = dataset_img_files[cursor+1] 

                if cursor+2 >= size:
                    late2_file = dataset_img_files[size-1] 
                else:
                    late2_file = dataset_img_files[cursor+2] 

                if cursor+3 >= size:
                    late3_file = dataset_img_files[size-1] 
                else:
                    late3_file = dataset_img_files[cursor+3] 
                
                imgIndex = curr_file[-12:]

                prev_imgIndex = prev_file[-12:]
                late_imgIndex = late_file[-12:]
                late2_imgIndex = late2_file[-12:]
                late3_imgIndex = late3_file[-12:]
                # print(videoName, " ", imgIndex) 

                targetObject_img_path = os.path.join(dataset_tarObject_path, videoName + '_target-00000001.jpg')

                full_img_path = os.path.join(dataset_img_path, videoName + "_image-" + imgIndex)
                prev_full_img_path = os.path.join(dataset_img_path, videoName + "_image-" + prev_imgIndex) 
                late_full_img_path = os.path.join(dataset_img_path, videoName + "_image-" + late_imgIndex) 
                late2_full_img_path = os.path.join(dataset_img_path, videoName + "_image-" + late2_imgIndex) 
                late3_full_img_path = os.path.join(dataset_img_path, videoName + "_image-" + late3_imgIndex) 

                full_map_path = os.path.join(dataset_mask_path, videoName + "_mask-" + imgIndex) 
                prev_full_map_path = os.path.join(dataset_mask_path, videoName + "_mask-" + prev_imgIndex) 
                late_full_map_path = os.path.join(dataset_mask_path, videoName + "_mask-" + late_imgIndex) 
                late2_full_map_path = os.path.join(dataset_mask_path, videoName + "_mask-" + late2_imgIndex) 

                cursor += 1

                inputimage = cv2.imread(full_img_path) 
                prev_inputimage = cv2.imread(prev_full_img_path) 
                late_inputimage = cv2.imread(late_full_img_path) 
                late2_inputimage = cv2.imread(late2_full_img_path) 
                late3_inputimage = cv2.imread(late3_full_img_path) 

                # pdb.set_trace() 
                batch_img[batchidx] = to_tensor(inputimage)
                batch_imgClip[batchidx, 0] = to_tensor(prev_inputimage) 
                batch_imgClip[batchidx, 1] = to_tensor(inputimage) 
                batch_imgClip[batchidx, 2] = to_tensor(late_inputimage) 

                batch_imgClip_mot[batchidx, 0] = to_tensor(prev_inputimage) 
                batch_imgClip_mot[batchidx, 1] = to_tensor(inputimage) 
                batch_imgClip_mot[batchidx, 2] = to_tensor(late_inputimage) 
                batch_imgClip_mot[batchidx, 3] = to_tensor(late2_inputimage) 
                batch_imgClip_mot[batchidx, 4] = to_tensor(late3_inputimage) 
                
                targetObjectimage = cv2.imread(targetObject_img_path)
                targetObject_img[batchidx] = to_tensor(targetObjectimage)
                
                saliencyimage = cv2.imread(full_map_path, 0)
                saliencyimage = np.expand_dims(saliencyimage, axis=2)
                batch_map[batchidx] = to_tensor(saliencyimage) 

                prev_saliencyimage = cv2.imread(prev_full_map_path, 0)
                prev_saliencyimage = np.expand_dims(prev_saliencyimage, axis=2)

                late_saliencyimage = cv2.imread(late_full_map_path, 0)
                late_saliencyimage = np.expand_dims(late_saliencyimage, axis=2)    

                late2_saliencyimage = cv2.imread(late2_full_map_path, 0)
                late2_saliencyimage = np.expand_dims(late2_saliencyimage, axis=2)    


                batch_maskClip[batchidx, 0] = to_tensor(prev_saliencyimage)
                batch_maskClip[batchidx, 1] = to_tensor(saliencyimage) 
                batch_maskClip[batchidx, 2] = to_tensor(late_saliencyimage) 

                batch_maskClip_mot[batchidx, 0] = to_tensor(saliencyimage) 
                batch_maskClip_mot[batchidx, 1] = to_tensor(late_saliencyimage) 
                batch_maskClip_mot[batchidx, 2] = to_tensor(late2_saliencyimage)  


            batch_img = to_variable(batch_img, requires_grad=True)
            batch_map = to_variable(batch_map, requires_grad=False)
            targetObject_img = to_variable(targetObject_img, requires_grad=True)
            batch_imgClip = to_variable(batch_imgClip, requires_grad=True) 
            batch_maskClip = to_variable(batch_maskClip, requires_grad=False) 

            batch_maskClip_mot = to_variable(batch_maskClip_mot, requires_grad=False) 
            batch_imgClip_mot = to_variable(batch_imgClip_mot, requires_grad=True) 
            

            val_batchImg = batch_img
            val_targetObjectImg = targetObject_img 
            val_imgClip = batch_imgClip  


            count = count + 1

             
            
            

            ##########################################################################
            ####                          Train App-D 
            ##########################################################################
            # print("""Calculate GAN loss for the appearance discriminator""") 
            if n_updates % 5 == 1:
                d_optim_app.zero_grad()
                outputs = app_Dis_Net(batch_img, batch_map)  ## given the real data. 
                outputs = torch.sigmoid(outputs) 

                loss_appD_real = loss_function(outputs, real_labels) 

                attention_map = generator(batch_img, targetObject_img, batch_imgClip)
                outputs = app_Dis_Net(batch_img, attention_map) 
                outputs = torch.sigmoid(outputs) 
                loss_appD_fake = loss_function(outputs, fake_labels) 

                # combine loss and calculate gradients
                loss_appD = (loss_appD_fake + loss_appD_real) 
                loss_appD.backward()
                d_optim_app.step() 

                print("Epoch [%d/%d], [%d/%d],  ------------------------------------------------------------------ Loss_appD: %f" % \
                (current_epoch, num_epoch, videoidx, len(video_files), loss_appD))

            ##########################################################################
            ####                     Train Motion-D 
            ##########################################################################
            # print("""Calculate GAN loss for the Motion discriminator""") 
            elif n_updates % 3 == 1:
                d_optim_mot.zero_grad()
                outputs = mot_Dis_Net(torch.transpose(batch_imgClip_mot[:, 1:-1], 1, 2), torch.transpose(batch_maskClip_mot, 1, 2))  ## given the real data. 
                outputs = torch.sigmoid(outputs) 
                loss_motD_real = loss_function(outputs, real_labels) 


                pred_attention_map = torch.zeros(batch_size, clip_len, 3, batch_map.shape[2], batch_map.shape[3])  

                for imgINDEX in range(clip_len): 
                    batch_imgClip = batch_imgClip_mot[:, imgINDEX:imgINDEX+2] 
                    batch_img = batch_imgClip_mot[:, imgINDEX+1] 

                    pred_attention_map[0, imgINDEX] = generator(batch_img, targetObject_img, batch_imgClip)
                    
                outputs = mot_Dis_Net(torch.transpose(batch_imgClip_mot[:, 1:-1], 1, 2), pred_attention_map.cuda())  
                outputs = torch.sigmoid(outputs) 
                loss_motD_fake = loss_function(outputs, fake_labels) 

                # combine loss and calculate gradients
                loss_motD = (loss_motD_fake + loss_motD_real) 
                loss_motD.backward()
                d_optim_mot.step() 

                print("Epoch [%d/%d], [%d/%d], ---------------------------------------- Loss_motD: %f" % \
                    (current_epoch, num_epoch, videoidx, len(video_files), loss_motD))

            else:
                ##########################################################################
                ####                            Train G                                   
                ##########################################################################
                # print("""Calculate GAN and BCE loss for the generator""")
                g_optim.zero_grad()
                attention_map = generator(batch_img, targetObject_img, batch_imgClip) 
                outputs = app_Dis_Net(batch_img, attention_map) 
                outputs = torch.sigmoid(outputs) 
                loss_G_GAN1 = loss_function(outputs, real_labels) 


                pred_attention_map = torch.zeros(batch_size, clip_len, 3, batch_map.shape[2], batch_map.shape[3])  

                for imgINDEX in range(clip_len): 
                    batch_imgClip = batch_imgClip_mot[:, imgINDEX:imgINDEX+2] 
                    batch_img = batch_imgClip_mot[:, imgINDEX+1] 

                    pred_attention_map[0, imgINDEX] = generator(batch_img, targetObject_img, batch_imgClip)

                outputs = mot_Dis_Net(torch.transpose(batch_imgClip_mot[:, 1:-1], 1, 2), pred_attention_map.cuda())  
                outputs = torch.sigmoid(outputs) 
                loss_G_GAN2 = loss_function(outputs, real_labels) 

                alpha1 = 0.1  
                alpha2 = 0.2 
                
                # pdb.set_trace()
                g_gen_loss = BCEcriterion(attention_map, batch_map)

                # pdb.set_trace() 
                loss_G = torch.sum(g_gen_loss + alpha1 * loss_G_GAN1 + alpha2 * loss_G_GAN2)
                loss_G.backward()
                g_optim.step() 

                print("Epoch [%d/%d], [%d/%d], Loss_G: %f" % (current_epoch, num_epoch, videoidx, len(video_files), loss_G))



            n_updates += 1 


        # validation 
        out = generator(val_batchImg, val_targetObjectImg, val_imgClip)
        map_out = out.cpu().data.squeeze(0)
        for iiidex in range(batch_size): 
           new_path = DIR_TO_SAVE + str(current_epoch) + str(n_updates) + ".jpg"
           pilTrans = transforms.ToPILImage()
           pilImg = pilTrans(map_out[iiidex]) 
           print('==>> Image saved to ', new_path)
           pilImg.save(new_path)

    # pdb.set_trace()
    # Save weights 
    if current_epoch % 1 == 0:
        print("==>> save checkpoints ... ")
        torch.save(generator.state_dict(), 'TANet_model.pkl')





