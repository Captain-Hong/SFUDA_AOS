# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:42:04 2021

@author: 11627
"""
import torch.nn.functional as F
import os
from torch import optim
from torch.utils.data import DataLoader
from utils import helpers
from utils.loss import SoftDiceLoss,EntropyLoss,entropy_loss,prob_2_entropy,MaxSquareloss
from utils.metrics import diceCoeffv2
from networks.u_net import U_Net
from networks.temp_model import T_Net,Resnet_Unet
from networks.Unets import Unet1,Unet2
from networks.decoder import Decoder,Resnet_Decoder
from networks.generator import Gene,Resnet_Gene,Unet_Gene
#from networks.generator import Gene
from networks.discriminator import get_done_discriminator,get_dtwo_discriminator
from datasets import dataset_load
from utils.pamr import PAMR
import torch 
import sys
import imageio
import numpy as np
import torch.nn as nn
import imageio.core.util
from logging import warning as warn
import copy
from torch.autograd import Variable


def _precision_warn(p1, p2, extra=""):
    t = (
        "Lossy conversion from {} to {}. {} Convert image to {} prior to "
        "saving to suppress this warning."
    )
    warn(t.format(p1, p2, extra, p2))

                
def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                

class FeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def one_hot(masks):
    BG=copy.deepcopy(masks.cpu())
    BG[BG!=0]=99
    BG[BG==0]=1
    BG[BG==99]=0
    LV=copy.deepcopy(masks.cpu())
    LV[LV!=1]=99
    LV[LV==1]=1
    LV[LV==99]=0 
    RK=copy.deepcopy(masks.cpu())
    RK[RK!=2]=99
    RK[RK==2]=1
    RK[RK==99]=0 
    LK=copy.deepcopy(masks.cpu())
    LK[LK!=3]=99
    LK[LK==3]=1
    LK[LK==99]=0 
    SP=copy.deepcopy(masks.cpu())
    SP[SP!=4]=99
    SP[SP==4]=1
    SP[SP==99]=0 
    return BG,LV,RK,LK,SP



batch_size=8
n_class=5
pre_trained_model=torch.load('./supervised_learning_models/epoch_100_CT_Unet_woRecon.pth')
model_dict = pre_trained_model.state_dict()

Net_model=U_Net(img_ch=1, num_classes=n_class)
Net_model.load_state_dict(model_dict)

Net1=Unet1(img_ch=1)
Net1_dict = Net1.state_dict()
Net1_pretrained_dict = {k: v for k, v in model_dict.items() if k in Net1_dict}
Net1_dict.update(Net1_pretrained_dict)
Net1.load_state_dict(Net1_dict)

Net2=Unet2(num_classes=n_class)
Net2_dict = Net2.state_dict()
Net2_pretrained_dict = {k: v for k, v in model_dict.items() if k in Net2_dict}
Net2_dict.update(Net2_pretrained_dict)
Net2.load_state_dict(Net2_dict)

gene_model=Gene(img_ch=1, num_classes=1)
temp_model=T_Net(img_ch=1, num_classes=n_class)
# freeze Net_model network
Net1.cuda(0)
Net1.train()

Net2.cuda(0)
Net2.eval()

gene_model.cuda(1)
gene_model.train()
gene_model.apply(initialize_weights)



Net_model.cuda(1)
Net_model.eval()

temp_model.cuda(0)
temp_model.train()
temp_model.apply(initialize_weights)

## Create hooks for feature statistics
Net2_feature_layers = []
for module in Net2.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        Net2_feature_layers.append(FeatureHook(module))

Net_feature_layers = []
for module in Net_model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        Net_feature_layers.append(FeatureHook(module))



epochs=200
Net1_lrs=0.0001
gene_lrs=0.005
temp_lrs=0.005
activation_fn = torch.nn.Softmax2d()
MSE_criterion=torch.nn.MSELoss(reduction='mean')
MAE_criterion=torch.nn.L1Loss(reduction='mean')
KL_criterion=torch.nn.KLDivLoss(reduction='mean')
seg_criterion = SoftDiceLoss(num_classes=n_class)
maxsquare_criterion=MaxSquareloss()

weight_cliping_limit = 0.01
palette = [[0], [1], [2], [3], [4]]
one = torch.FloatTensor([1])
mone = one * -1

PAMR_KERNEL = [1, 2, 4, 8, 12]
PAMR_ITER = 10
pamr_aff = PAMR(PAMR_ITER, PAMR_KERNEL)


for epoch in range(epochs):
    if epoch%10==0:
        file_handle=open('en_dice.txt',mode='w')
        file_handle1=open('pamr_dice.txt',mode='w')
        file_handle2=open('bns_dice.txt',mode='w')
        
    train_set = dataset_load.data_set('./abdominal_image/MR','TRA.txt')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    Net1_optimizer = torch.optim.RMSprop(Net1.parameters(), lr=Net1_lrs, alpha=0.9)  
    gene_optimizer = torch.optim.RMSprop(gene_model.parameters(), lr=gene_lrs, alpha=0.9)
    temp_optimizer = torch.optim.RMSprop(temp_model.parameters(), lr=temp_lrs, alpha=0.9)

    d_len = 0
    Net2_bgDice = 0.0
    Net2_lvDice = 0.0
    Net2_rkDice = 0.0
    Net2_lkDice = 0.0
    Net2_spDice = 0.0
    
    BNS2_loss_record=[]
    Net2_en_loss_record=[]
    
    Net_bgDice = 0.0
    Net_lvDice = 0.0
    Net_rkDice = 0.0
    Net_lkDice = 0.0
    Net_spDice = 0.0

    temp_bgDice = 0.0
    temp_lvDice = 0.0
    temp_rkDice = 0.0
    temp_lkDice = 0.0
    temp_spDice = 0.0

    Combine_bgDice = 0.0
    Combine_lvDice = 0.0
    Combine_rkDice = 0.0
    Combine_lkDice = 0.0
    Combine_spDice = 0.0

    for inputs, mask in train_loader:
        

        
        # forward pass
        Net1_optimizer.zero_grad()
        gene_optimizer.zero_grad()
        temp_optimizer.zero_grad()
        Net2.zero_grad()
        Net_model.zero_grad()

        x1 = Net1(inputs.cuda(0))
        Net2_logit=Net2(x1)


        Net2_output=activation_fn(Net2_logit)       
        Net2_mask=torch.argmax(Net2_output, dim=1)
        Net2_BG,Net2_LV,Net2_RK,Net2_LK,Net2_SP=one_hot(Net2_mask)

        Net2_en_loss = entropy_loss(Net2_output)
        Net2_en_loss=Net2_en_loss*10
        Net2_en_loss.backward(retain_graph=True)
        
        
        
        Net2_BG_dice=diceCoeffv2(Net2_BG[:,:,:].cuda(0), mask[:, 0,:, :].cuda(0), activation=None).cpu().item()
        Net2_LV_dice=diceCoeffv2(Net2_LV[:,:,:].cuda(0), mask[:, 1,:, :].cuda(0), activation=None).cpu().item()
        Net2_RK_dice=diceCoeffv2(Net2_RK[:,:,:].cuda(0), mask[:, 2,:, :].cuda(0), activation=None).cpu().item()
        Net2_LK_dice=diceCoeffv2(Net2_LK[:,:,:].cuda(0), mask[:, 3,:, :].cuda(0), activation=None).cpu().item()                        
        Net2_SP_dice=diceCoeffv2(Net2_SP[:,:,:].cuda(0), mask[:, 4,:, :].cuda(0), activation=None).cpu().item()   

        Net2_mean_dice = (Net2_BG_dice+Net2_LV_dice+ Net2_SP_dice+ Net2_RK_dice+ Net2_LK_dice) / (n_class)
        d_len += 1
        Net2_bgDice += Net2_BG_dice
        Net2_lvDice += Net2_LV_dice
        Net2_rkDice += Net2_RK_dice
        Net2_lkDice += Net2_LK_dice
        Net2_spDice += Net2_SP_dice
        
        Net2_pseudo_label=torch.stack((Net2_BG,Net2_LV,Net2_RK,Net2_LK,Net2_SP),1)
        Net2_gt_loss = seg_criterion(Net2_output.detach(), mask.cuda(0).detach())




        trans_inputs = gene_model(inputs.cuda(1))
        trans_inputs=trans_inputs*inputs.cuda(1)
        
        _,Net_logit = Net_model(trans_inputs.cuda(1))
        
        Net_output=activation_fn(Net_logit)       
        Net_mask=torch.argmax(Net_output, dim=1)
        Net_BG,Net_LV,Net_RK,Net_LK,Net_SP=one_hot(Net_mask)
        
        Net_BG_dice=diceCoeffv2(Net_BG[:,:,:].cuda(1), mask[:, 0,:, :].cuda(1), activation=None).cpu().item()
        Net_LV_dice=diceCoeffv2(Net_LV[:,:,:].cuda(1), mask[:, 1,:, :].cuda(1), activation=None).cpu().item()
        Net_RK_dice=diceCoeffv2(Net_RK[:,:,:].cuda(1), mask[:, 2,:, :].cuda(1), activation=None).cpu().item()
        Net_LK_dice=diceCoeffv2(Net_LK[:,:,:].cuda(1), mask[:, 3,:, :].cuda(1), activation=None).cpu().item()                        
        Net_SP_dice=diceCoeffv2(Net_SP[:,:,:].cuda(1), mask[:, 4,:, :].cuda(1), activation=None).cpu().item()   

        Net_mean_dice = (Net_BG_dice+Net_LV_dice+ Net_SP_dice+ Net_RK_dice+ Net_LK_dice) / (n_class)

        Net_bgDice += Net_BG_dice
        Net_lvDice += Net_LV_dice
        Net_rkDice += Net_RK_dice
        Net_lkDice += Net_LK_dice
        Net_spDice += Net_SP_dice


        
        Net_pseudo_label=torch.stack((Net_BG,Net_LV,Net_RK,Net_LK,Net_SP),1)
        Net_seg_loss = seg_criterion(Net_output, Net2_pseudo_label.cuda(1).detach())
        Net_seg_loss.backward()

#       second stage 
#        if epoch>150:
#            Net2_seg_loss = seg_criterion(Net2_output, Net_pseudo_label.cuda(0).detach())
#            Net2_seg_loss.backward(retain_graph=True)

        # R_feature loss
        Net2_feature_loss = sum([mod.r_feature for mod in Net2_feature_layers])
        BNS2_loss=0.001*Net2_feature_loss
        BNS2_loss.backward()

        Net_gt_loss = seg_criterion(Net_output.detach(), mask.cuda(1).detach())


        Combine_output=Net_output.cpu()*Net2_output.cpu()#(Net_output.cpu()+Net2_output.cpu())/2
        Combine_mask=torch.argmax(Combine_output, dim=1)
        Combine_BG,Combine_LV,Combine_RK,Combine_LK,Combine_SP=one_hot(Combine_mask)

        Combine_BG_dice=diceCoeffv2(Combine_BG[:,:,:].cuda(0), mask[:, 0,:, :].cuda(0), activation=None).cpu().item()
        Combine_LV_dice=diceCoeffv2(Combine_LV[:,:,:].cuda(0), mask[:, 1,:, :].cuda(0), activation=None).cpu().item()
        Combine_RK_dice=diceCoeffv2(Combine_RK[:,:,:].cuda(0), mask[:, 2,:, :].cuda(0), activation=None).cpu().item()
        Combine_LK_dice=diceCoeffv2(Combine_LK[:,:,:].cuda(0), mask[:, 3,:, :].cuda(0), activation=None).cpu().item()                        
        Combine_SP_dice=diceCoeffv2(Combine_SP[:,:,:].cuda(0), mask[:, 4,:, :].cuda(0), activation=None).cpu().item()   

        Combine_mean_dice = (Combine_BG_dice+Combine_LV_dice+ Combine_SP_dice+ Combine_RK_dice+ Combine_LK_dice) / (n_class)

        Combine_bgDice += Combine_BG_dice
        Combine_lvDice += Combine_LV_dice
        Combine_rkDice += Combine_RK_dice
        Combine_lkDice += Combine_LK_dice
        Combine_spDice += Combine_SP_dice

        Combine_pseudo_label=torch.stack((Combine_BG,Combine_LV,Combine_RK,Combine_LK,Combine_SP),1)
        
        
        _,temp_logit = temp_model(inputs.cuda(0))
        temp_output=activation_fn(temp_logit) 
        



        temp_mask=torch.argmax(temp_output, dim=1)
        temp_BG,temp_LV,temp_RK,temp_LK,temp_SP=one_hot(temp_mask)
        
        temp_BG_dice=diceCoeffv2(temp_BG[:,:,:].cuda(0), mask[:, 0,:, :].cuda(0), activation=None).cpu().item()
        temp_LV_dice=diceCoeffv2(temp_LV[:,:,:].cuda(0), mask[:, 1,:, :].cuda(0), activation=None).cpu().item()
        temp_RK_dice=diceCoeffv2(temp_RK[:,:,:].cuda(0), mask[:, 2,:, :].cuda(0), activation=None).cpu().item()
        temp_LK_dice=diceCoeffv2(temp_LK[:,:,:].cuda(0), mask[:, 3,:, :].cuda(0), activation=None).cpu().item()                        
        temp_SP_dice=diceCoeffv2(temp_SP[:,:,:].cuda(0), mask[:, 4,:, :].cuda(0), activation=None).cpu().item()   

        temp_mean_dice = (temp_BG_dice+temp_LV_dice+ temp_SP_dice+ temp_RK_dice+ temp_LK_dice) / (n_class)

        temp_bgDice += temp_BG_dice
        temp_lvDice += temp_LV_dice
        temp_rkDice += temp_RK_dice
        temp_lkDice += temp_LK_dice
        temp_spDice += temp_SP_dice

        pamr_aff.cuda(0)
        temp_pamr_out_put = pamr_aff(inputs.detach().cuda(0), temp_output.detach())
        temp_pamr_mask=torch.argmax(temp_pamr_out_put, dim=1)
        temp_pamr_BG,temp_pamr_LV,temp_pamr_RK,temp_pamr_LK,temp_pamr_SP=one_hot(temp_pamr_mask) 
        temp_pamr_pseudo_label=torch.stack((temp_pamr_BG,temp_pamr_LV,temp_pamr_RK,temp_pamr_LK,temp_pamr_SP),1)
        temp_pamr_seg_loss =seg_criterion(temp_output, temp_pamr_pseudo_label.detach().cuda(0))
        temp_pamr_seg_loss = temp_pamr_seg_loss*0.1
#        temp_pamr_seg_loss.backward(retain_graph=True) 
        
        temp_en_loss = entropy_loss(temp_output)
        temp_en_loss=temp_en_loss*10
        temp_en_loss.backward(retain_graph=True)
        
        temp_seg_loss = seg_criterion(temp_output, Net_pseudo_label.cuda(0).detach())
        temp_seg_loss.backward()








        Net1_optimizer.step()
        gene_optimizer.step()
        temp_optimizer.step()
    
        string_print = "Epoch = %d entropy2 = %.4f bns2 = %.4f Net2_Dice=%.4f Net_Dice=%.4f combine_Dice=%.4f temp_Dice=%.4f"\
                       % (epoch,Net2_en_loss.item(), Net2_feature_loss.item(),Net2_mean_dice,Net_mean_dice,Combine_mean_dice,temp_mean_dice)
        print("\r"+string_print,end = "",flush=True)
        BNS2_loss_record.append(Net2_feature_loss.item())
        Net2_en_loss_record.append(Net2_en_loss.item())

        
        if epoch%10==0:
            file_handle.write(str(Net2_en_loss.item()))
            file_handle.write('\t')
            file_handle.write(str(Net2_gt_loss.item()))
            file_handle.write('\n')

            
            file_handle2.write(str(BNS2_loss.item()))
            file_handle2.write('\t')
            file_handle2.write(str(Net2_gt_loss.item()))
            file_handle2.write('\n')             
            
            
    if epoch%10==0:
        file_handle.close()
        file_handle1.close()
        file_handle2.close()
        
        
    Net2_bgDice = Net2_bgDice / d_len
    Net2_lvDice = Net2_lvDice / d_len
    Net2_rkDice = Net2_rkDice / d_len
    Net2_lkDice = Net2_lkDice / d_len
    Net2_spDice = Net2_spDice / d_len
    Net2_m_dice = (Net2_bgDice+Net2_lvDice + Net2_rkDice+Net2_lkDice+Net2_spDice) / (n_class)

    Net_bgDice = Net_bgDice / d_len
    Net_lvDice = Net_lvDice / d_len
    Net_rkDice = Net_rkDice / d_len
    Net_lkDice = Net_lkDice / d_len
    Net_spDice = Net_spDice / d_len
    Net_m_dice = (Net_bgDice+Net_lvDice + Net_rkDice+Net_lkDice+Net_spDice) / (n_class)

    Combine_bgDice = Combine_bgDice / d_len
    Combine_lvDice = Combine_lvDice / d_len
    Combine_rkDice = Combine_rkDice / d_len
    Combine_lkDice = Combine_lkDice / d_len
    Combine_spDice = Combine_spDice / d_len
    Combine_m_dice = (Combine_bgDice+Combine_lvDice + Combine_rkDice+Combine_lkDice+Combine_spDice) / (n_class)

    temp_bgDice = temp_bgDice / d_len
    temp_lvDice = temp_lvDice / d_len
    temp_rkDice = temp_rkDice / d_len
    temp_lkDice = temp_lkDice / d_len
    temp_spDice = temp_spDice / d_len
    temp_m_dice = (temp_bgDice+temp_lvDice + temp_rkDice+temp_lkDice+temp_spDice) / (n_class)
    
    BNS2_mean_loss=sum(BNS2_loss_record)/len(BNS2_loss_record)
    Net2_en_mean_loss=sum(Net2_en_loss_record)/len(Net2_en_loss_record)

    print('\nEpoch {}/{},BNS2_loss {:.5},Net2_Dice {:.4},Net_Dice {:.4},combine_Dice {:.4},temp_Dice {:.4}\n'.format(
        epoch, epochs,BNS2_mean_loss, Net2_m_dice, Net_m_dice, Combine_m_dice, temp_m_dice ))
    
    Net2_mask = 50*Net2_mask.cpu().detach().numpy()
    img=inputs.cpu().detach().numpy().transpose([0, 2, 3, 1])
    mask= 50*torch.argmax(mask, dim=1).cpu().detach().numpy()

    for ii in range(4):
        imageio.imwrite(os.path.join('./Net2_imgs', str(ii)+'.png'),img[ii])
        imageio.imwrite(os.path.join('./Net2_imgs', str(ii)+'_GT.png'),mask[ii])
        imageio.imwrite(os.path.join('./Net2_imgs', str(ii)+'_PRE.png'),Net2_mask[ii])

    Net_mask = 50*Net_mask.cpu().detach().numpy()
    img=inputs.cpu().detach().numpy().transpose([0, 2, 3, 1])
    trans=trans_inputs.cpu().detach().numpy().transpose([0, 2, 3, 1])
    for ii in range(4):
        imageio.imwrite(os.path.join('./Net_imgs', str(ii)+'.png'),img[ii])
        imageio.imwrite(os.path.join('./Net_imgs', str(ii)+'_GT.png'),mask[ii])
        imageio.imwrite(os.path.join('./Net_imgs', str(ii)+'_PRE.png'),Net_mask[ii])
        imageio.imwrite(os.path.join('./Net_imgs', str(ii)+'_TRA.png'),trans[ii])
        
    Combine_mask1=50*Combine_mask.cpu().detach().numpy()
    temp_mask= 50*temp_mask.cpu().detach().numpy()
    for ii in range(4):
        imageio.imwrite(os.path.join('./temp_imgs', str(ii)+'.png'),img[ii])
        imageio.imwrite(os.path.join('./temp_imgs', str(ii)+'_GT.png'),mask[ii])
        imageio.imwrite(os.path.join('./temp_imgs', str(ii)+'_CB.png'),Combine_mask1[ii])
        imageio.imwrite(os.path.join('./temp_imgs', str(ii)+'_PRE.png'),temp_mask[ii])        
        
        