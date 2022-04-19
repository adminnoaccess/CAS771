# -*- coding:utf-8 -*-
from __future__ import print_function 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from model import CNN_Shallow, CNN_Deep
import argparse, sys
import numpy as np
import shutil

from torchvision.datasets import ImageFolder
from loader_for_CIFAR import cifar_dataloader
from loader_for_ANIMAL10n import animal10n_dataloader
from loss import loss_triteaching_cir, loss_triteaching_plus

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'cifar10, cifar100, or animal10n', default = 'cifar10')
parser.add_argument('--task', type = str, help = 'task1, task2, or task3', default = 'task1')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--CNN', type = str, help = 'deep or shallow', default = 'deep')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=30)

parser.add_argument('--optimizer', type = str, default='adam')
parser.add_argument('--model_type', type = str, help='[triteaching, triteachingplus]', default='triteaching')


args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

    
if args.dataset=='cifar100':
    input_channel=3
    init_epoch = 5
    num_classes = 100

# cifar 10 dataloader
if args.dataset == 'cifar10':
    input_channel=3
    init_epoch = 5
    num_classes = 10
    print ('\n>>>>>>  Loading dataset cifar 10 ...\n')
    dataset = 'cifar10'  # either cifar10 or cifar100
    data_path = 'C:\\Users\\leoxu\\OneDrive\\Desktop\\CAS_771\\Tri_teaching_plus\\data\\cifar-10-batches-py'
    loader = cifar_dataloader(dataset, batch_size=128,
                        num_workers=5,
                        root_dir=data_path,
                        noise_file='%s/cifar10_noisy_labels_%s.json' % (data_path, args.task)) #
    train_loader, noisy_labels, clean_labels = loader.run('train')
    noise_or_not2 = np.transpose(noisy_labels)==np.transpose(clean_labels)
    print('noise or not2: ', noise_or_not2)
    print('number of True in noise or not2', noise_or_not2.sum())
    # print('shape of noise_or_not2', noise_or_not2.shape)
    # print('length of noise_or_not2:', len(noise_or_not2))
    # print('noisy_labels:', noisy_labels)
    # print('shape of noisy_labels', noisy_labels.shape)
    # print('clean_labels:', clean_labels)
    # print('shape of clean_labels', clean_labels.shape)

    # print('Type of noisy_label', type(noisy_labels))
    test_loader = loader.run('test')
    print("\n>>>>>>  Noisy labels are loaded into data\n")
    # print('loaded noisy rate:', noise_or_not2)

# animal 10n dataloader
if args.dataset == 'animal10n':
    input_channel=3
    init_epoch = 5
    num_classes = 10
    print ('\n>>>>>>  Loading dataset animal 10n ...\n')
    data_path = 'C:\\Users\\leoxu\\OneDrive\\Desktop\\CAS_771\\Tri_teaching_plus\\data\\animal_10n'
    loader = animal10n_dataloader(batch_size = 128, num_workers = 5, root_dir = data_path) #
    train_dataset, train_loader = loader.run('train')
    train_dataset, test_loader = loader.run('test')
    noise_or_not2 = np.ones(50000, dtype=bool)
    print('noise or not2: ', noise_or_not2)
    print('number of True in noise or not2', noise_or_not2.sum())


if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate


# define learning rate
beta_initial = 0.9
beta_adjust = 0.1
# array with n_epoch elements of value learning_rate
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [beta_initial] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = beta_adjust
print("lr alpha plan: ", alpha_plan)
print("beta plan: ", beta1_plan)
# adjust optimizer from learning rate
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) 
       
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
         
# define result saving
save_dir = 'results/' +args.dataset+'/' + args.model_type + '/'
print('save dir: ', save_dir) 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
model_str=args.dataset+'_' + args.model_type + '_' + args.CNN + '_'+ args.task + '_'+ str(args.n_epoch)
txtfile = save_dir + "/" + model_str + ".txt"


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2, model3, optimizer3):
    print ('\n>>>>>>  Training %s...' % model_str)

    pure_ratio_1_list = []
    pure_ratio_2_list = []
    pure_ratio_3_list = []
    
    train_total = 0
    train_correct1 = 0 
    train_total2 = 0
    train_correct2 = 0 
    train_total3 = 0
    train_correct3 = 0

    for i, (data, labels, indices) in enumerate(train_loader):
        ind = indices.cpu().numpy().transpose()
      
        labels = Variable(labels).cuda()
        data = Variable(data).cuda()
        # Forward + Backward + Optimize
        logits1=model1(data)
        # print('shape of logits1:', logits1.shape)
        # print('length of logits1:', len(logits1))
        # print('type of logits1:', type(logits1))
        # print('type of label:', type(labels))
        prec1,  = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct1+=prec1

        logits2 = model2(data)
        prec2,  = accuracy(logits2, labels, topk=(1, ))
        train_total2+=1
        train_correct2+=prec2

        logits3 = model3(data)
        prec3,  = accuracy(logits3, labels, topk=(1, ))
        train_total3+=1
        train_correct3+=prec3

        if epoch < init_epoch:
            loss_1, loss_2, loss_3, pure_ratio_1, pure_ratio_2, pure_ratio_3 = loss_triteaching_cir(logits1, logits2, logits3, labels, rate_schedule[epoch], ind, noise_or_not2)
        else:
            if args.model_type=='triteachingplus':
                loss_1, loss_2, loss_3, pure_ratio_1, pure_ratio_2, pure_ratio_3 = loss_triteaching_plus(logits1, logits2, logits3, labels, rate_schedule[epoch], ind, noise_or_not2, epoch*i)
            elif args.model_type=='triteaching':
                loss_1, loss_2, loss_3, pure_ratio_1, pure_ratio_2, pure_ratio_3 = loss_triteaching_cir(logits1, logits2, logits3, labels, rate_schedule[epoch], ind, noise_or_not2)
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)
        pure_ratio_3_list.append(100*pure_ratio_3)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        optimizer3.zero_grad()
        loss_3.backward()
        optimizer3.step()

        mean_pure_ratio1 = np.sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        mean_pure_ratio3 = np.sum(pure_ratio_3_list)/len(pure_ratio_3_list)
        
        if (i+1) % args.print_freq == 0:
            print('\n>>>>>>  Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Training Accuracy3: %.4f, Loss1: %.4f, Loss2: %.4f, Loss3: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f, Pure Ratio3 %.4f\n' 
                  %(epoch+1, args.n_epoch, i+1, 50000//batch_size, prec1, prec2, prec3, loss_1.item(), loss_2.item(), loss_3.item(), mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio3))

    train_acc1=float(train_correct1)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    train_acc3=float(train_correct3)/float(train_total3)
    # print(pure_ratio_1_list)
    return train_acc1, train_acc2, train_acc3, pure_ratio_1_list, pure_ratio_2_list, pure_ratio_3_list

# Evaluate the Model
def evaluate(test_loader, model1, model2, model3):
    print('\n>>>>>>  Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, indices in test_loader:
        data = Variable(data).cuda()
        # print('size of image', data.size())
        # print('size of labels', labels.size())
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for data, labels, indices in test_loader:
        data = Variable(data).cuda()
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()

    model3.eval()    # Change model to 'eval' mode 
    correct3 = 0
    total3 = 0
    for data, labels, indices in test_loader:
        data = Variable(data).cuda()
        logits3 = model3(data)
        outputs3 = F.softmax(logits3, dim=1)
        _, pred3 = torch.max(outputs3.data, 1)
        total3 += labels.size(0)
        correct3 += (pred3.cpu() == labels.long()).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    acc3 = 100*float(correct3)/float(total3)

    return acc1, acc2, acc3

def main():
    # Define models
    print('\n>>>>>>  Building model...')

    if args.CNN == 'deep':
        CNN = CNN_Deep
        clf1 = CNN(input_channel=input_channel, n_outputs=num_classes)
        clf2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        clf3 = CNN(input_channel=input_channel, n_outputs=num_classes)
    elif args.CNN == 'shallow':
        CNN = CNN_Shallow
        clf1 = CNN(num_classes)
        clf2 = CNN(num_classes)
        clf3 = CNN(num_classes)


    
    clf1.cuda()
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)  
    clf2.cuda()
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)  
    clf3.cuda()
    optimizer3 = torch.optim.Adam(clf3.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0
    mean_pure_ratio3=0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 train_acc2 train_acc3 test_acc1 test_acc2 test_acc3 pure_ratio1 pure_ratio2 pure_ratio3\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    train_acc3=0
    # evaluate models with random weights
    test_acc1, test_acc2, test_acc3 =evaluate(test_loader, clf1, clf2, clf3)
    print('\n>>>>>>  Epoch [%d/%d] Test Accuracy on the test data: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %% Pure Ratio3 %.4f %%' % (epoch+1, args.n_epoch, test_acc1, test_acc2, test_acc3, mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio3))
    # save results
    # print('mean pure_ratio_1_list: ', mean_pure_ratio1)
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' ' + str(train_acc3) + ' '+ str(test_acc1) + ' ' + str(test_acc2)  + ' ' + str(test_acc3) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2)+ ' '  + str(mean_pure_ratio3) +"\n")

    # training
    for epoch in range(1, args.n_epoch):
        # set to train modes
        clf1.train()
        clf2.train()
        clf3.train()

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)
        adjust_learning_rate(optimizer3, epoch)
        print('current lr: ', alpha_plan[epoch])
        print('current beta: ', beta1_plan[epoch])
        # train the models
        print('forget rate: ', rate_schedule[epoch])
        print('remember rate: ', 1-rate_schedule[epoch])
        train_acc1, train_acc2, train_acc3, pure_ratio_1_list, pure_ratio_2_list, pure_ratio_3_list = train(train_loader, epoch, clf1, optimizer1, clf2, optimizer2, clf3, optimizer3)
        # test models
        test_acc1, test_acc2, test_acc3 =evaluate(test_loader, clf1, clf2, clf3)
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        mean_pure_ratio3 = sum(pure_ratio_3_list)/len(pure_ratio_3_list)
        print('\n>>>>>>  Epoch [%d/%d] Test Accuracy on the test data: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %% Pure Ratio3 %.4f %%' % (epoch+1, args.n_epoch, test_acc1, test_acc2, test_acc3, mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio3))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' ' + str(train_acc3) + ' '+ str(test_acc1) + ' ' + str(test_acc2)  + ' ' + str(test_acc3) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2)+ ' '  + str(mean_pure_ratio3) +"\n")
if __name__=='__main__':
    main()
