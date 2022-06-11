# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:42:04 2021
k&$d3yyf,269032
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
from networks.final_model import F_Net
from networks.coord_conv import CoordConvNet
from networks.Unets import Unet1,Unet2
from networks.generator import Gene
from networks.discriminator import get_discriminator
from datasets import dataset_load,dataset_load1
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
import os.path as osp

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

def nn_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3,1,1, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # 对图像进行卷积操作
    edge_detect = conv_op(Variable(im))
    # 将输出转换为图片格式
#    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect




def _get_compactness_cost(y_pred): 

    """
    y_pred: BxCxHxW
    """
    """
    lenth term
    """

    # y_pred = tf.one_hot(y_pred, depth=5)
    # print (y_true.shape)
    # print (y_pred.shape)
    y_pred = y_pred[:,:,:, :]


    x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
    y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

    delta_x = x[:,:,:,1:]**2
    delta_y = y[:,:,1:,:]**2

    delta_u = torch.abs(delta_x + delta_y) 

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 0.01
    length = w * torch.sum(torch.sqrt(delta_u + epsilon), [2, 3])

    area = torch.sum(y_pred, [2,3])

    compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926))

    return compactness_loss



batch_size=8
n_class=5
pre_trained_model=torch.load('./pretrained_models/epoch_100_CT_Unet_woRecon.pth')
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
final_model=F_Net(img_ch=1, num_classes=n_class)

Net1.cuda(0)

Net2.cuda(0)
Net2.eval()

gene_model.cuda(1)
gene_model.apply(initialize_weights)



Net_model.cuda(1)
Net_model.eval()

final_model.cuda(0)
final_model.apply(initialize_weights)



# Create hooks for feature statistics
Net2_feature_layers = []
for module in Net2.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        Net2_feature_layers.append(FeatureHook(module))

Net_feature_layers = []
for module in Net_model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        Net_feature_layers.append(FeatureHook(module))



epochs=400
Net1_lrs=0.00012
gene_lrs=0.0004
final_lrs=0.0006
activation_fn = torch.nn.Softmax2d()
MSE_criterion=torch.nn.MSELoss(reduction='mean')
MAE_criterion=torch.nn.L1Loss(reduction='mean')
KL_criterion=torch.nn.KLDivLoss(reduction='mean')
seg_criterion = SoftDiceLoss(num_classes=n_class)
maxsquare_criterion=MaxSquareloss()
entropys=EntropyLoss(reduction='sum')
weight_cliping_limit = 0.01
palette = [[0], [1], [2], [3], [4]]
one = torch.FloatTensor([1])
mone = one * -1

PAMR_KERNEL = [1,2,4,8,12,24]
PAMR_ITER = 10
pamr_aff = PAMR(PAMR_ITER, PAMR_KERNEL)



for epoch in range(epochs):
    
    train_set = dataset_load.data_set('./abdominal_image/MR','TRA2.txt')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    train_set1 = dataset_load1.data_set('./abdominal_image/MR','added_1.txt')
    train_loader1 = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    Net1_optimizer = torch.optim.RMSprop(Net1.parameters(), lr=Net1_lrs, alpha=0.9)  
    gene_optimizer = torch.optim.RMSprop(gene_model.parameters(), lr=gene_lrs, alpha=0.9)
    final_optimizer = torch.optim.RMSprop(final_model.parameters(), lr=final_lrs, alpha=0.9)

    
    d_len = 0
    Net2_bgDice = 0.0
    Net2_lvDice = 0.0
    Net2_rkDice = 0.0
    Net2_lkDice = 0.0
    Net2_spDice = 0.0
    
    BNS2_loss_record=[]
    BNS_loss_record=[]
    final_seg_loss_record=[]
    final_pamr_loss_record=[]
    Net2_en_loss_record=[]
    
    Net_bgDice = 0.0
    Net_lvDice = 0.0
    Net_rkDice = 0.0
    Net_lkDice = 0.0
    Net_spDice = 0.0

    final_bgDice = 0.0
    final_lvDice = 0.0
    final_rkDice = 0.0
    final_lkDice = 0.0
    final_spDice = 0.0


    for inputs, mask in train_loader:
        Net1.train()
        gene_model.train()
        final_model.train()
        d_len += 1

        B,_,_,_=inputs.size()

        Net1_optimizer.zero_grad()
        gene_optimizer.zero_grad()
        
        Net2.zero_grad()
        Net_model.zero_grad()
        final_optimizer.zero_grad()

        x1 = Net1(inputs.cuda(0))

        Net2_logit=Net2(x1)


        Net2_output=activation_fn(Net2_logit)       
        Net2_mask=torch.argmax(Net2_output.detach(), dim=1)
        Net2_BG,Net2_LV,Net2_RK,Net2_LK,Net2_SP=one_hot(Net2_mask)

        Net2_en_loss = entropy_loss(Net2_output)
        Net2_en_loss=Net2_en_loss*10
        Net2_en_loss.backward(retain_graph=True)
        

        

        
        Net2_pseudo_label=torch.stack((Net2_BG,Net2_LV,Net2_RK,Net2_LK,Net2_SP),1)
        Net2_gt_loss = seg_criterion(Net2_output.detach(), mask.cuda(0).detach())


        Net2_entropys=entropys(Net2_output[:,1:2,:,:].detach())
        Net2_sorted_logits, Net2_sorted_indices = torch.sort(Net2_entropys)

        style_inputs = gene_model(inputs.cuda(1))
        style_inputs=style_inputs*inputs.cuda(1)
        
        _,Net_logit = Net_model(style_inputs.cuda(1))
        
        Net_output=activation_fn(Net_logit)       
        Net_mask=torch.argmax(Net_output.detach(), dim=1)
        Net_BG,Net_LV,Net_RK,Net_LK,Net_SP=one_hot(Net_mask)

        Net_pseudo_label=torch.stack((Net_BG,Net_LV,Net_RK,Net_LK,Net_SP),1)

        Net_gt_loss = seg_criterion(Net_output.detach(), mask.cuda(1).detach())

        Net_entropys=entropys(Net_output.detach())
        Net_sorted_logits, Net_sorted_indices = torch.sort(Net_entropys)

        _,final_logit = final_model(inputs.cuda(0))
        final_output=activation_fn(final_logit)
        final_mask=torch.argmax(final_output.detach(), dim=1)
        final_BG,final_LV,final_RK,final_LK,final_SP=one_hot(final_mask)
        final_pseudo_label=torch.stack((final_BG,final_LV,final_RK,final_LK,final_SP),1)
        

        
        final_entropys=entropys(final_output.detach())
        final_sorted_logits, final_sorted_indices = torch.sort(final_entropys)


        
        Net_seg_loss = seg_criterion(Net_output, Net2_pseudo_label.cuda(1).detach())
        Net_seg_loss.backward()


        if epoch>150:
            pamr_aff.cuda(0)
            final_pamr_out_put = pamr_aff(inputs.detach().cuda(0), final_output.detach())
            final_pamr_mask=torch.argmax(final_pamr_out_put, dim=1)
            final_pamr_BG,final_pamr_LV,final_pamr_RK,final_pamr_LK,final_pamr_SP=one_hot(final_pamr_mask) 
            final_pamr_pseudo_label=torch.stack((final_pamr_BG,final_pamr_LV,final_pamr_RK,final_pamr_LK,final_pamr_SP),1)
            final_pamr_seg_loss =seg_criterion(final_output, final_pamr_pseudo_label.detach().cuda(0))
            final_pamr_seg_loss =0.6*final_pamr_seg_loss
            final_pamr_seg_loss.backward(retain_graph=True)

            Net2_seg_loss0 =0.3*seg_criterion(Net2_output, final_pseudo_label.cuda(0).detach())
            Net2_seg_loss0.backward(retain_graph=True)
            
        
        # R_feature loss
        Net2_feature_loss = sum([mod.r_feature for mod in Net2_feature_layers])
        BNS2_loss=0.001*Net2_feature_loss
        BNS2_loss.backward()            


        
        final_seg_loss = seg_criterion(final_output, Net_pseudo_label.cuda(0).detach())
        final_seg_loss.backward()


        final_optimizer.step()
        
        Net1_optimizer.step()
        gene_optimizer.step()






        Net1.eval()
        gene_model.eval()
        final_model.eval()
        
        
        

        Net2_BG_dice=diceCoeffv2(Net2_BG[:,:,:].cuda(0), mask[:, 0,:, :].cuda(0), activation=None).cpu().item()
        Net2_LV_dice=diceCoeffv2(Net2_LV[:,:,:].cuda(0), mask[:, 1,:, :].cuda(0), activation=None).cpu().item()
        Net2_RK_dice=diceCoeffv2(Net2_RK[:,:,:].cuda(0), mask[:, 2,:, :].cuda(0), activation=None).cpu().item()
        Net2_LK_dice=diceCoeffv2(Net2_LK[:,:,:].cuda(0), mask[:, 3,:, :].cuda(0), activation=None).cpu().item()                        
        Net2_SP_dice=diceCoeffv2(Net2_SP[:,:,:].cuda(0), mask[:, 4,:, :].cuda(0), activation=None).cpu().item()   

        Net2_mean_dice = (Net2_BG_dice+Net2_LV_dice+ Net2_SP_dice+ Net2_RK_dice+ Net2_LK_dice) / (n_class)
        
        Net2_bgDice += Net2_BG_dice
        Net2_lvDice += Net2_LV_dice
        Net2_rkDice += Net2_RK_dice
        Net2_lkDice += Net2_LK_dice
        Net2_spDice += Net2_SP_dice

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



        final_BG_dice=diceCoeffv2(final_BG[:,:,:].cuda(0), mask[:, 0,:, :].cuda(0), activation=None).cpu().item()
        final_LV_dice=diceCoeffv2(final_LV[:,:,:].cuda(0), mask[:, 1,:, :].cuda(0), activation=None).cpu().item()
        final_RK_dice=diceCoeffv2(final_RK[:,:,:].cuda(0), mask[:, 2,:, :].cuda(0), activation=None).cpu().item()
        final_LK_dice=diceCoeffv2(final_LK[:,:,:].cuda(0), mask[:, 3,:, :].cuda(0), activation=None).cpu().item()                        
        final_SP_dice=diceCoeffv2(final_SP[:,:,:].cuda(0), mask[:, 4,:, :].cuda(0), activation=None).cpu().item()   

        final_mean_dice = (final_BG_dice+final_LV_dice+ final_SP_dice+ final_RK_dice+ final_LK_dice) / (n_class)
        final_mean_dice1 = (final_LV_dice+ final_SP_dice+ final_RK_dice+ final_LK_dice) / (n_class-1)
        final_bgDice += final_BG_dice
        final_lvDice += final_LV_dice
        final_rkDice += final_RK_dice
        final_lkDice += final_LK_dice
        final_spDice += final_SP_dice






        string_print = "Epoch=%d bs2=%.2f N2_Dice=%.4f N_Dice=%.4f f_Dice=%.4f f_Dice1=%.4f"\
                       % (epoch, Net2_feature_loss.item(),Net2_mean_dice,Net_mean_dice,final_mean_dice,final_mean_dice1)


        print("\r"+string_print,end = "",flush=True)
        BNS2_loss_record.append(Net2_feature_loss.item())

        Net2_en_loss_record.append(Net2_en_loss.item())
        


    for inputs1, mask1 in train_loader1:
        final_model.train()
        final_optimizer.zero_grad()
        _,final_logit1 = final_model(inputs1.cuda(0))
        final_output1=activation_fn(final_logit1)
        
        final_seg_loss1 = 0.3*seg_criterion(final_output1, mask1.cuda(0).detach())
        final_seg_loss1.backward()
        final_optimizer.step()
        final_model.eval()
        
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


    final_bgDice = final_bgDice / d_len
    final_lvDice = final_lvDice / d_len
    final_rkDice = final_rkDice / d_len
    final_lkDice = final_lkDice / d_len
    final_spDice = final_spDice / d_len
    final_m_dice = (final_bgDice+final_lvDice + final_rkDice+final_lkDice+final_spDice) / (n_class)
    final_m_dice1 = (final_lvDice + final_rkDice+final_lkDice+final_spDice) / (n_class-1)

    BNS2_mean_loss=sum(BNS2_loss_record)/len(BNS2_loss_record)
    Net2_en_mean_loss=sum(Net2_en_loss_record)/len(Net2_en_loss_record)

    
    print('\nEpoch {}/{},BNS2_loss {:.5},Net2_Dice {:.4},Net_Dice {:.4},final_Dice {:.4},final_Dice1 {:.4}\n'.format(
        epoch, epochs,BNS2_mean_loss,Net2_m_dice, Net_m_dice,final_m_dice,final_m_dice1))

#    torch.save(Net1,
#               osp.join('./added_1_Net1_model', f'Net1_model_{epoch}.pth'))
#
#    torch.save(gene_model,
#               osp.join('./added_1_SC_model', f'SC_model_{epoch}.pth'))
#
#    torch.save(final_model,
#               osp.join('./added_1_final_model', f'CT2MR_2_{epoch}.pth'))
    torch.cuda.empty_cache()
    
    Net2_mask = 50*Net2_mask.cpu().detach().numpy()
    img=inputs.cpu().detach().numpy().transpose([0, 2, 3, 1])
    mask= 50*torch.argmax(mask, dim=1).cpu().detach().numpy()

    for ii in range(4):
        imageio.imwrite(os.path.join('./Net2_imgs', str(ii)+'.png'),img[ii])
        imageio.imwrite(os.path.join('./Net2_imgs', str(ii)+'_GT.png'),mask[ii])
        imageio.imwrite(os.path.join('./Net2_imgs', str(ii)+'_PRE.png'),Net2_mask[ii])

    Net_mask = 50*Net_mask.cpu().detach().numpy()
    style=style_inputs.cpu().detach().numpy().transpose([0, 2, 3, 1])
    for ii in range(4):
        imageio.imwrite(os.path.join('./Net_imgs', str(ii)+'.png'),img[ii])
        imageio.imwrite(os.path.join('./Net_imgs', str(ii)+'_GT.png'),mask[ii])
        imageio.imwrite(os.path.join('./Net_imgs', str(ii)+'_PRE.png'),Net_mask[ii])
        imageio.imwrite(os.path.join('./Net_imgs', str(ii)+'_TRA.png'),style[ii])

    final_mask= 50*final_mask.cpu().detach().numpy()
    for ii in range(4):
        imageio.imwrite(os.path.join('./final_imgs', str(ii)+'.png'),img[ii])
        imageio.imwrite(os.path.join('./final_imgs', str(ii)+'_GT.png'),mask[ii])
        imageio.imwrite(os.path.join('./final_imgs', str(ii)+'_PRE.png'),final_mask[ii])

       