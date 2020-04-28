import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_var(x, requires_grad=False, volatile=False):

    if torch.cuda.is_available():
        x = x.cuda(0)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)
#####################################################################################################
    
def compute_confusion_matrix(nb_classes, dataloader, model_ft):    
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        
    class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    return confusion_matrix, class_acc
#####################################################################################################
    
def prune_rate(net, verbose = False):
    total = 0
    prune = 0 
    layer = 0
    for parameter in net.parameters():
        if len(parameter.data.size()) >= 2:
            params = 1
            for dim in parameter.data.size():
                params *= dim
            total+=params
            
        if len(parameter.data.size()) >= 2:
                layer+=1
                zero_param = np.count_nonzero(parameter.cpu().data.numpy()==0)
                prune += zero_param
                if verbose:
                    print("Layer {} | {} layer | {:.2f}% parameters pruned" \
    					    .format(
    						layer,
    						'Conv' if len(parameter.data.size()) == 4 \
    						    else 'Linear',
    						100.*zero_param/params,
    						))
                
    pruning_perc = 100.*prune/total
    print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc
#####################################################################################################
    
def count_params(net):
    conv_params = 0
    for key in net.modules():
        if (isinstance(key, nn.Conv2d) | isinstance(key, nn.Linear)):
            conv_params += sum(p.numel() for p in key.parameters() if p.requires_grad)
    print("Total number of convolution parameters: ", conv_params)
    
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)
#####################################################################################################
    
def processes_scores(feature_scores, classes, scale, csize): 
    for c in range(0,classes):
        feature_scores[c,:,:] = (feature_scores[c,:,:]*scale[c])/csize[c]
    feature_scores = np.mean(feature_scores,axis=0)
    feature_scores = np.absolute(feature_scores)
    for i in range(np.shape(feature_scores)[1]):
        feature_scores[:,i] = feature_scores[:,i]/(np.linalg.norm(feature_scores[:,i])+1e-9)
    return feature_scores
#####################################################################################################
    
def processes_scores_v2(feature_scores, classes, scale, csize, n): 
    for c in range(0,classes):
        feature_scores[c,:,:] = (feature_scores[c,:,:]*scale[c])/csize[c]
    feature_scores = np.mean(feature_scores,axis=0)
    feature_scores = np.absolute(feature_scores)
    for i in range(np.shape(feature_scores)[1]):
        if ((i%3)==0):
            feature_scores[:,i] = feature_scores[:,i]/(np.linalg.norm(feature_scores[:,i])+1e-9)   
        else:
            feature_scores[0:16*n,i]    = feature_scores[0:16*n,i]/(np.linalg.norm(feature_scores[0:16*n,i])+1e-9)
            feature_scores[16*n:64*n,i] += 1e9
    return feature_scores
#####################################################################################################
    
def get_indices(r_score, prune_list,n):
    score       = np.absolute(r_score)
    asc_idx     = np.argsort(score)
    req_idx     = [idx for idx in asc_idx if idx not in prune_list]
    next_prune  = np.asarray(req_idx[0:n], dtype = np.int16)
    prune_list  = np.append(prune_list, next_prune)
    prune_list  = np.asarray(prune_list, dtype = np.int16)
    return next_prune, prune_list
#####################################################################################################
    
def prune_conv(net, layer, index_prev, index_curr, fout, fin, kernel_size=3):
    mask_w = torch.ones((fout,fin,kernel_size,kernel_size)).cuda()
    mask_w[:,index_prev,:,:]   = torch.zeros(fout,np.size(index_prev),kernel_size,kernel_size).cuda()
    mask_w[index_curr,:,:,:]   = torch.zeros(np.size(index_curr), fin,kernel_size,kernel_size).cuda()
    layer.set_mask(mask_w)
    return net

def prune_conv_res(net, layer, index_prev, index_curr, fout, fin, kernel_size=3):
    mask_w = torch.ones((fout,fin,kernel_size,kernel_size)).cuda()
    mask_w[index_curr,:,:,:]   = torch.zeros(np.size(index_curr), fin,kernel_size,kernel_size).cuda()
    layer.set_mask(mask_w)
    return net

def prune_conv_np(net, layer, index, fout, fin, kernel_size=3):
    mask_w = torch.ones((fout,fin,kernel_size,kernel_size)).cuda()
    mask_w[index,:,:,:]   = torch.zeros(np.size(index),fin,kernel_size,kernel_size).cuda()
    layer.set_mask(mask_w)
    return net

def prune_linear(net, layer, index_prev, index_curr, fout, fin):
    mask_w = torch.ones((fout,fin)).cuda()
    mask_b = torch.ones((fout)).cuda()
    mask_w[index_curr,:]   = torch.zeros((np.size(index_curr),fin)).cuda()
    mask_w[:, index_prev]  = torch.zeros((fout, np.size(index_prev))).cuda()
    mask_b[index_curr]     = torch.zeros((np.size(index_curr))).cuda()
    layer.set_mask(mask_w, mask_b)
    return net

def prune_linear_np(net, layer, index_prev, fout, fin):
    mask_w = torch.ones((fout,fin)).cuda()
    mask_b = torch.ones((fout)).cuda()
    mask_w[:,index_prev]   = torch.zeros(fout,np.size(index_prev)).cuda()
    layer.set_mask(mask_w, mask_b)
    return net
#####################################################################################################
    
def prune_res56(net, prune_list_conv):
    j=0
    for i in range(0,8):
        prune_conv(net, net.module.layer3[8-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 64,64); j+=1
        prune_conv_res(net, net.module.layer3[8-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 64,64); j+=1
    prune_conv(net, net.module.layer3[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 64,64); j+=1
    prune_conv_res(net, net.module.layer3[0].conv1, prune_list_conv[j+1], prune_list_conv[j], 64,32); j+=1      
    for i in range(0,8):
        prune_conv(net, net.module.layer2[8-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 32,32); j+=1
        prune_conv_res(net, net.module.layer2[8-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 32,32); j+=1
    prune_conv(net, net.module.layer2[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 32,32); j+=1
    prune_conv_res(net, net.module.layer2[0].conv1, prune_list_conv[j+1], prune_list_conv[j], 32,16); j+=1       
    for i in range(0,8):
        prune_conv(net, net.module.layer1[8-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 16,16); j+=1
        prune_conv_res(net, net.module.layer1[8-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 16,16); j+=1
    prune_conv(net, net.module.layer1[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 16,16); j+=1
    return net
#####################################################################################################
    
def prune_res110(net, prune_list_conv):
    j=0
    for i in range(0,17):
        prune_conv(net, net.module.layer3[17-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 64,64); j+=1
        prune_conv_res(net, net.module.layer3[17-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 64,64); j+=1
    prune_conv(net, net.module.layer3[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 64,64); j+=1
    prune_conv_res(net, net.module.layer3[0].conv1, prune_list_conv[j+1], prune_list_conv[j], 64,32); j+=1            
    for i in range(0,17):
        prune_conv(net, net.module.layer2[17-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 32,32); j+=1
        prune_conv_res(net, net.module.layer2[17-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 32,32); j+=1
    prune_conv(net, net.module.layer2[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 32,32); j+=1
    prune_conv_res(net, net.module.layer2[0].conv1, prune_list_conv[j+1], prune_list_conv[j], 32,16); j+=1            
    for i in range(0,17):
        prune_conv(net, net.module.layer1[17-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 16,16); j+=1
        prune_conv_res(net, net.module.layer1[17-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 16,16); j+=1
    prune_conv(net, net.module.layer1[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 16,16); j+=1    
    return net
#####################################################################################################
    
def prune_res164(net, prune_list_conv):
    j=0
    for i in range(0,17):
        prune_conv(net, net.module.layer3[17-i].conv3, prune_list_conv[j+1], prune_list_conv[j], 256,64,1); j+=1
        prune_conv(net, net.module.layer3[17-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 64,64,3);  j+=1
        prune_conv_res(net, net.module.layer3[17-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 64,256,1); j+=1
    prune_conv(net, net.module.layer3[0].conv3, prune_list_conv[j+1], prune_list_conv[j], 256,64,1); j+=1    
    prune_conv(net, net.module.layer3[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 64,64,3);  j+=1
    prune_conv_res(net, net.module.layer3[0].conv1, prune_list_conv[j+1], prune_list_conv[j], 64,128,1); j+=1            
    for i in range(0,17):
        prune_conv(net, net.module.layer2[17-i].conv3, prune_list_conv[j+1], prune_list_conv[j], 128,32,1); j+=1
        prune_conv(net, net.module.layer2[17-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 32,32,3); j+=1
        prune_conv_res(net, net.module.layer2[17-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 32,128,1); j+=1
    prune_conv(net, net.module.layer2[0].conv3, prune_list_conv[j+1], prune_list_conv[j], 128,32,1); j+=1
    prune_conv(net, net.module.layer2[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 32,32,3); j+=1
    prune_conv_res(net, net.module.layer2[0].conv1, prune_list_conv[j+1], prune_list_conv[j], 32,64,1); j+=1            
    for i in range(0,17):
        prune_conv(net, net.module.layer1[17-i].conv3, prune_list_conv[j+1], prune_list_conv[j], 64,16,1); j+=1
        prune_conv(net, net.module.layer1[17-i].conv2, prune_list_conv[j+1], prune_list_conv[j], 16,16,3); j+=1
        prune_conv_res(net, net.module.layer1[17-i].conv1, prune_list_conv[j+1], prune_list_conv[j], 16,64,1); j+=1
    prune_conv(net, net.module.layer1[0].conv3, prune_list_conv[j+1], prune_list_conv[j], 64,16,1); j+=1
    prune_conv(net, net.module.layer1[0].conv2, prune_list_conv[j+1], prune_list_conv[j], 16,16,3); j+=1   
    return net
#####################################################################################################