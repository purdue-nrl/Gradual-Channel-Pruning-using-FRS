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

parser = argparse.ArgumentParser(description='CIFAR10/CIFAR100 gradual pruning while training on ResNet')
parser.add_argument('--lr',         default=0.1, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--dataset',    default='cifar10', type=str, help='dataset = [cifar10, cifar100]')
parser.add_argument('--n',          default=21,  type=int, help='pruning step size')
parser.add_argument('--x',          default=200, type=int, help='Number of filters to be pruned at each pruning step')
parser.add_argument('--N1',         default=150, type=int, help='end of pruning interval')
parser.add_argument('--epochs',     default=200, type=int, help='Total number of training epochs')
parser.add_argument('--model_dir',  metavar='MODEL_DIR', default='./saved_models/vgg16_pruned.h5', help='MODEL directory')
args = parser.parse_args()

def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p): m.load_state_dict(torch.load(p))

device     = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = args.model_dir

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset    = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset     = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
    input_ch     = 3 
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset    = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset     = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100
    input_ch     = 3 
  
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2,pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2,pin_memory=True)

# Model
print('==> Building model..')
net = vgg16_bn(classes=num_classes)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
print(net)

net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
seed = 0           
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def adjust_learning_rate(optimizer, epoch):
    update_list = [100, 150]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


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
            #print(net)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return test_loss, acc
##############################################################################S
                                                                              
prune_layers = [40, 37, 34, 30, 27, 24, 20, 17, 14, 10, 7]
lin_layers   = [45]
f    = [512, 512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 64, 64]

prune_list_conv = {0:np.array([],dtype='int32')}
prune_list_lin  = np.array([],dtype='int32')
for i in range(1,11):
    prune_list_conv[i] = np.array([],dtype='int32')

feature_score = np.ones((512,11))*1e9

for epoch in range(0, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    train(epoch, net)
    test(epoch, net)
    if epoch in range(0,args.N1):
        if (epoch+1)%args.n == 0:
            print('Computing fetaure relevance scores...')
            cm, class_acc = compute_confusion_matrix(num_classes, trainloader, net)
            class_acc = class_acc/torch.max(class_acc)
            scale     = (1./class_acc)
            scale     = F.sigmoid(scale)
            scale     = scale.detach().numpy();
            feature_score1 = rscore_layer_vgg(net, trainloader, prune_layers[0:6], num_classes,f[0],scale)
            feature_score2 = rscore_layer_vgg(net, trainloader, prune_layers[6:9], num_classes,f[6],scale)
            feature_score3 = rscore_layer_vgg(net, trainloader, prune_layers[9:], num_classes,f[9],scale)
            feature_score[0:512,0:6]  = feature_score1
            feature_score[0:256,6:9]  = feature_score2
            feature_score[0:128,9:11] = feature_score3
   
            if epoch!=(args.n-1):
                feature_score_l = rscore_layer_vgg(net, trainloader, lin_layers, num_classes, 512,scale) 
                next_prunec, prune_list_lin    = get_indices(feature_score_l[:,0], prune_list_lin, 22)
                for i in range(0,11):
                    feature_score[prune_list_conv[i],i]=1e9
            else:
               feature_score_l = rscore_layer_vgg(net, trainloader,lin_layers , num_classes, 512,scale) 
               next_prunec, prune_list_lin    = get_indices(feature_score_l[:,0], np.array([]), 22) 
           
            for i in range(args.x):
                b1 = np.array(np.where(feature_score==np.min(feature_score)))
                prune_list_conv[int(b1[1,0])] = np.append(prune_list_conv[int(b1[1,0])],b1[0,0])
                feature_score[int(b1[0,0]),int(b1[1,0])]=1e9
            print('Pruning the x least important channels...')
            net = prune_vgg16(net, prune_list_conv, prune_list_lin, prune_layers, f, num_classes)
            print('Test accuracy after pruning:')
            test(epoch, net)   
            prune_rate(net, True)
            print('continue training...')

print('test accuracy after training')    
test(epoch, net)
save_model(net, model_path)
prune_rate(net, True)           



