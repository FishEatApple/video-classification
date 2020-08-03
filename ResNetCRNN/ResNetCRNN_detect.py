import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from utils.utils import *

# set path
data_path = "/home/gjj/data/jpegs_256"                # define UCF-101 RGB data path
frames_test_floder="v_ApplyEyeMakeup_g01_c01"
action_name_path = "./_UCF101actions.pkl"
save_model_path = "./ResNetCRNN_ckpt/"

# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category
batch_size = 40
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1


with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)   # load UCF101 actions names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)



# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

# read single img
single_data_loader=Img_trans_2_detect(data_path,frames_test_floder,selected_frames,use_transform=transform)
single_data_loader.unsqueeze_(0)

# reload CRNN model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch63_singleGPU.pth')))
rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch63_singleGPU.pth')))
print('CRNN model reloaded!')

t1 = torch_utils.time_synchronized()
y_pred = CRNN_detect_prediction([cnn_encoder, rnn_decoder], device, single_data_loader)
t2 = torch_utils.time_synchronized()
print('%sDone. (%.3fs)' % ("time used is ", t2 - t1))

print(cat2labels(le,y_pred))

print('video prediction finished!')




