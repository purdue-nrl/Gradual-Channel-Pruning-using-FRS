import numpy as np
import torch
from models import *
from model_relprop import *
from utils_1 import *

def forward_hook(self, input, output):
    self.X = input[0]
    self.Y = output
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#####################################################################################################
def rscore_layer_vgg(net, trainloader, layers, classes,f,scale):
    model_VGG = VGG16Net_cifar(net).cuda()
    for i in range(0, len(model_VGG.layers)):
        model_VGG.layers[i].register_forward_hook(forward_hook)
    model_VGG.eval()
    feature_score = np.zeros((classes, f,len(layers)))
    csize =  np.zeros(classes)
    with torch.no_grad():
        for idx, (input, label) in enumerate(trainloader):
            input, label = input.to(device), label.to(device)
            output_VGG   = model_VGG(input)  
            pred_VGG     = output_VGG.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            outputs      = pred_VGG  
            T_VGG     = label.cpu().numpy()
            T_VGG     = (T_VGG[:,np.newaxis] == np.arange(classes))*1.0
            T_VGG     = torch.from_numpy(T_VGG).type(torch.cuda.FloatTensor)
            LRP_VGG   = model_VGG.relprop(T_VGG)
            k=0
            for layer in layers:
                score     = model_VGG.layers[layer+1].Rscore
                score     = score.view(score.size(0),score.size(1),-1)
                score     = torch.mean(score,2) 
                score     = score.cpu().detach().numpy()
                for i in range(0,input.size(0)):
                    feature_score[label[i],:,k]+= score[i,:]
                    if k==0:
                        csize[label[i]]+=1             
                k +=1
                
    # process the relevance scores             
    feature_scores = processes_scores(feature_score, classes, scale, csize)
  
    return feature_scores

#####################################################################################################
    
def rscore_layer_res56(net, trainset, classes,scale):
    model  = RES56Net(net).cuda()
    model.eval()
    feature_score1 = np.zeros((classes,64,18))
    feature_score2 = np.zeros((classes,32,18))
    feature_score3 = np.zeros((classes,16,18))
    csize = np.zeros(classes)

    # store the activations using forward hook function
    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook)
    # set of layers which need the feature relevance scores i.e. all the conv layers
    layers = []
    for name, module in model.named_modules():
        if name[-5:]=='lay.2' or name[-5:]=='res.0':
            layers.append(module)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for idx, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)
            output   = model(input)  
            csize   +=  1
            T    = label.cpu().numpy()
            T    = (T[:,np.newaxis] == np.arange(classes))*1.0
            T    = torch.from_numpy(T).type(torch.cuda.FloatTensor)
            LRP  = model.relprop(T)
            for i in range(0,54):
                score     = layers[i].Rscore
                score     = score.view(score.size(0),score.size(1),-1)
                score     = torch.mean(score,2) 
                score     = score.cpu().detach().numpy()
                for j in range(0, input.size(0)):
                    if ~np.isnan(score[j,:]).any(): 
                        if i in range(0,18):
                            feature_score3[label[j],0:,17-i] += score[j,:]
                        elif i in range(18,36):
                            feature_score2[label[j],0:,35-i] += score[j,:]
                        else: 
                            feature_score1[label[j],0:,53-i] += score[j,:]
                    if i==0:
                        csize[label[j]]+=1
     
    # process the relevance scores             
    feature_score1 = processes_scores(feature_score1, classes, scale, csize)
    feature_score2 = processes_scores(feature_score2, classes, scale, csize)
    feature_score3 = processes_scores(feature_score3, classes, scale, csize)
   
    return feature_score1, feature_score2, feature_score3

#####################################################################################################
    
def rscore_layer_res110(net, trainset, classes, scale):
    model  = RES110Net(net).cuda() 
    model.eval()
    feature_score1 = np.zeros((classes,64,36))
    feature_score2 = np.zeros((classes,32,36))
    feature_score3 = np.zeros((classes,16,36))
    csize = np.zeros(classes)
    
    # store the activations using forward hook function
    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook)
    # set of layers which need the feature relevance scores i.e. all the conv layers
    layers = []
    for name, module in model.named_modules():
        if name[-5:]=='lay.2' or name[-5:]=='res.0':
            layers.append(module)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2,pin_memory=True)
    with torch.no_grad():
        for idx, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)
            output   = model(input)  
            csize   +=  1
            T    = label.cpu().numpy()
            T    = (T[:,np.newaxis] == np.arange(classes))*1.0
            T    = torch.from_numpy(T).type(torch.cuda.FloatTensor)
            LRP  = model.relprop(T)
               
            for i in range(0,108):
                score     = layers[i].Rscore
                score     = score.view(score.size(0),score.size(1),-1)
                score     = torch.mean(score,2) 
                score     = score.cpu().detach().numpy()
                for j in range(0, input.size(0)):
                    if ~np.isnan(score[j,:]).any(): 
                        if i in range(0,36):
                            feature_score3[label[j],0:,35-i] += score[j,:]
                        elif i in range(36,72):
                            feature_score2[label[j],0:,71-i] += score[j,:]
                        else: 
                            feature_score1[label[j],0:,107-i] += score[j,:]
                    if i==0:
                        csize[label[j]]+=1 
                                                 
    feature_score1 = processes_scores(feature_score1, classes, scale, csize)
    feature_score2 = processes_scores(feature_score2, classes, scale, csize)
    feature_score3 = processes_scores(feature_score3, classes, scale, csize)
   
    return feature_score1, feature_score2, feature_score3

#########################################################################################
    
def rscore_layer_res164(net, trainset, classes,scale):
    model  = RES164Net(net).cuda() 
    for name, module in model.named_modules():
        module.register_forward_hook(forward_hook)
    model.eval()
    
    feature_score1 = np.zeros((classes, 256, 54))
    feature_score2 = np.zeros((classes, 128, 54))
    feature_score3 = np.zeros((classes, 64,  54))
   
    csize = np.zeros(classes)
    # set of layers which need the feature relevance scores i.e. all the conv layers
    layers = []
    for name, module in model.named_modules():
        if name[-5:]=='lay.2' or name[-5:]=='lay.5' or name[-5:]=='res.0':
            layers.append(module)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2,pin_memory=True) 
    with torch.no_grad():
        for idx, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)
            output   = model(input)  
            csize   +=  1
            T    = label.cpu().numpy()
            T    = (T[:,np.newaxis] == np.arange(classes))*1.0
            T    = torch.from_numpy(T).type(torch.cuda.FloatTensor)
            LRP  = model.relprop(T)
               
            for i in range(0,162):
                score     = layers[i].Rscore
                score     = score.view(score.size(0),score.size(1),-1)
                score     = torch.mean(score,2) 
                score     = score.cpu().detach().numpy()
                for j in range(0, input.size(0)):
                    if ~np.isnan(score[j,:]).any(): 
                        if i in range(0,54):
                            feature_score3[label[j],0:np.shape(score)[1],53-i] += score[j,:]
                        elif i in range(54,108):
                            feature_score2[label[j],0:np.shape(score)[1],107-i] += score[j,:]
                        else: 
                            feature_score1[label[j],0:np.shape(score)[1],161-i] += score[j,:]
                    if i==0:
                        csize[label[j]]+=1  
                          
    feature_score1 = processes_scores_v2(feature_score1, classes, scale, csize, 4)
    feature_score2 = processes_scores_v2(feature_score2, classes, scale, csize, 2)
    feature_score3 = processes_scores_v2(feature_score3, classes, scale, csize, 1)
                                              
    return feature_score1, feature_score2, feature_score3