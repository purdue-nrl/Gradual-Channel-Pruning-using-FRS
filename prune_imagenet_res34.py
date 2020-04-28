'''
Author: Sai Aparna Aketi
'''
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from models import *
from utils import progress_bar
import numpy as np
from model_relprop import *
from utils_1 import *
from relevance_scores import *

parser = argparse.ArgumentParser(description='ImageNet gradual pruning while training on ResNet')
parser.add_argument('--lr',         default=0.01, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--subset_size', default=5000, type=int, help='number of batches used for evaluating relevance score where batch-size of computation is train_batch_size/4 where default is 64/4 = 16')
parser.add_argument('--dataset',    default='imagenet', type=str, help='dataset = [ImageNet]')
parser.add_argument('--model',      default='resnet-34', type=str, help='models = [resnet-34]')
parser.add_argument('--n',          default=9,  type=int, help='pruning step size')
parser.add_argument('--x',          default=250, type=int, help='Number of filters to be pruned at each pruning step')
parser.add_argument('--N1',         default=50, type=int, help='end of pruning interval')
parser.add_argument('--epochs',     default=90, type=int, help='Total number of training epochs')
parser.add_argument('--class_relevance_scale',     default=0, type=int, help='0 or 1')
parser.add_argument('--model_dir',  metavar='MODEL_DIR', default='./saved_models/res34_pruned.h5', help='MODEL directory')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p): m.load_state_dict(torch.load(p))
model_path = args.model_dir

# Data
print('==> Preparing data..')
if(args.dataset == 'imagenet'):
    print("| Preparing imagenet dataset...")  
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainset = datasets.ImageFolder('/local/a/imagenet/imagenet2012/train',transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,]))
    testset  = datasets.ImageFolder('/local/a/imagenet/imagenet2012/val', transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,]))
    num_classes = 1000
    input_ch     = 3 

    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2,pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Model
print('==> Building model..')
if(args.model =='resnet-34'):
    net = ResNet34(num_classes=num_classes)
else:
    print('unkown model')
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True   
print(args.model)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
seed = 5           
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def forward_hook(self, input, output):
    self.X = input[0]
    self.Y = output

# Training
def train(epoch, net):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    save_model(net, model_path) 
    
    
#testing 
def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
   

feature_score = np.ones((512,32))*(1e9)
prune_list_conv = {0:np.array([],dtype='int32')}
for i in range(1,32):
    prune_list_conv[i] = np.array([],dtype='int32')

#pruning while training
for epoch in range(0, args.epochs):
    adjust_learning_rate(optimizer, epoch, args.lr)
    train(epoch, net)
    test(epoch, net)
    if epoch in range(0,args.N1):
        if (epoch+1)%args.n == 0:
            print('Computing fetaure relevance scores...')
            scale = np.ones(num_classes)
            if args.class_relevance_scale == 0:
                cm, class_acc = compute_confusion_matrix_subset(num_classes, trainloader, net, args.subset_size)
                class_acc = class_acc/torch.max(class_acc)
                scale     = (1./class_acc)
                scale     = F.sigmoid(scale)
                scale     = scale.detach().numpy() 
            feature_score1,feature_score2, feature_score3, feature_score4 = rscore_layer_res34(net, trainset, num_classes, scale, args.batch_size, args.subset_size)         
            feature_score[0:512,0:6]     = feature_score1
            feature_score[0:256,6:18]    = feature_score2
            feature_score[0:128,18:26]   = feature_score3
            feature_score[0:64,26:32]    = feature_score4
            if epoch!=(args.n-1):
                for i in range(0,32):
                    feature_score[prune_list_conv[i],i]=1e9
            for i in range(args.x):
                b1 = np.array(np.where(feature_score==np.min(feature_score)))
                prune_list_conv[int(b1[1,0])] = np.append(prune_list_conv[int(b1[1,0])],b1[0,0])
                feature_score[int(b1[0,0]),int(b1[1,0])]=1e9 
            print('Pruning the x least important channels...')
            net = prune_res34(net, prune_list_conv)
            print('Test accuracy after pruning:')
            test(epoch, net)   
            prune_rate(net, True)
            print('continue training...')
    
test(epoch, net)
save_model(net, model_path)
prune_rate(net, True)

    



