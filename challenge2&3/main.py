import numpy as np
import torch
import argparse
import os
import torch.nn as nn
from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
from dataset import quality_dataset, grading_dataset
from datetime import datetime
import cv2
from functions import progress_bar
from torchnet import meter
import torch.nn.functional as F
from sklearn.metrics import f1_score,roc_auc_score,cohen_kappa_score,accuracy_score,confusion_matrix
from lr_scheduler import LRScheduler
import random
import torch.backends.cudnn as cudnn
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='res50', help='model')
parser.add_argument('--visname', '-vis', default='kaggle', help='visname')
parser.add_argument('--batch-size', '-bs', default=32, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n-cls', default=3, type=int, help='n-classes')
parser.add_argument('--pretrained', '-pre', default=False, type=bool, help='use pretrained model')
parser.add_argument('--dataset', '-data', default='pair', type=str, help='dataset')
parser.add_argument('--KK', '-KK', default=0, type=int, help='KFold')
parser.add_argument('--challenge', '-c', default=2, type=int, help='c')
parser.add_argument('--mixup', '-mix', default=False, type=bool, help='mixup')

parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)
parser.add_argument("--lambda_value", default=0.25, type=float)

val_epoch = 1
test_epoch = 1


my_whole_seed = 0
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False 
    

def parse_args():
    global args
    args = parser.parse_args()

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    device = x.get_device()
    if use_cuda:
        # index = torch.randperm(batch_size).cuda()
        index = torch.randperm(batch_size).to(device).long()
    else:
        index = torch.randperm(batch_size).long()

    mixed_x = (lam * x + (1 - lam) * x[index,:]).clone()
    y_a, y_b = y, y[index]
    return mixed_x, y_a.long(), y_b.long(), lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

best_acc = 0
best_kappa = 0

best_test_acc = 0
best_test_kappa = 0


def main():
    
    global best_kappa
    global save_dir
    
    parse_args()

    if args.model == 'seresnext101' and args.challenge == 2:
        net = torch.load('checkpoints/models_dict/quality_assessment/seresnext101_6x6_all.pth')
    elif args.model == 'effb6':
        net = timm.create_model('tf_efficientnet_b6', pretrained=True, num_classes=args.n_classes)
    elif args.model == 'incep_res':
        net = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=args.n_classes)
    elif args.model == 'mobile':
        net = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=args.n_classes)
    elif args.model == 'incepv3':
        net = timm.create_model('tf_inception_v3', pretrained=True, num_classes=args.n_classes)
    elif args.model == 'vit':
        net = timm.create_model('vit_small_r26_s32_384', pretrained=True, num_classes=args.n_classes)
    elif args.model == 'resnest50':
        net = timm.create_model('resnest50d', pretrained=True, num_classes=args.n_classes)

    print(net)

    if args.pretrained:
        print ("==> Load pretrained model")
        if args.model == 'vit':
            if args.challenge == 3:
                ckpt = torch.load('checkpoints/ddr_resnest50_n3/35.pkl')
            elif args.challenge == 2:
                ckpt = torch.load('checkpoints/challenge2/coips_vit_clf_384/18.pkl')
        elif args.model == 'incepv3':
            if args.challenge == 2:
                ckpt = torch.load('/home/feng/hjl/DRAC/checkpoints/challenge2/coips512_incepv3_clf/25.pkl')
        state_dict = ckpt['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,True)

    net = nn.DataParallel(net)
    net = net.cuda()

    if args.challenge == 2:
        trainset = quality_dataset(train=True, val=False, test=False, KK=args.KK)
        valset = quality_dataset(train=False, val=True, test=False, KK=args.KK)

    elif args.challenge == 3:
        trainset = grading_dataset(train=True, val=False, test=False, KK=args.KK)
        valset = grading_dataset(train=False, val=True, test=False, KK=args.KK)

    drop_last=False
    if args.challenge == 2 and args.mixup:
        if args.KK!=2:
            drop_last=True
    if args.challenge == 3:
        if (args.model == 'resnest50' and args.KK==4) or (args.model == 'vit' and (args.KK==2 or args.KK==3)):
            drop_last=True

    trainloader = DataLoader(trainset,  shuffle=True, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=drop_last)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # optim & crit
    if args.model == 'seresnext101':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #1e-5
    lr_scheduler = LRScheduler(optimizer, len(trainloader), args)


    if args.challenge == 2:
        if args.pretrained:
            weight = None
        else:
        weight=torch.tensor([0.6188, 0.3214, 0.0598])
        # weight=torch.tensor([0.2204, 0.2706, 0.5090]) # coips
    elif args.challenge == 3:
        if args.pretrained:
            weight = None
        else:
            weight=torch.tensor([0.1379, 0.2140, 0.6481])        
    criterion = nn.CrossEntropyLoss(weight=weight)

    criterion = criterion.cuda()
    con_matx = meter.ConfusionMeter(args.n_classes)

    #mixup
    criterion_mix = SoftTargetCrossEntropy().cuda()
    mixup_fn = Mixup(
        mixup_alpha=0.4, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.5, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=3)

    if args.challenge == 2:
        file = 'challenge2/'
    elif args.challenge == 3:
        file = 'challenge3/'
    save_dir = './checkpoints/' + file + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log=open('./logs/'+file + args.visname+'.txt','w')   

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        con_matx.reset()
        net.train()
        total_loss = .0
        total = .0
        correct = .0
        count = .0

        for i, (x, label, id) in enumerate(trainloader):
            lr = lr_scheduler.update(i, epoch)
            x = x.float().cuda()
            label = label.cuda()
            
            y_pred = net(x)         
            
            loss_clf = criterion(y_pred, label)
            prediction = y_pred.max(1)[1]

            if args.mixup:
                mix_imgs, mix_labels = mixup_fn(x, label)
                mix_pred = net(mix_imgs)
                loss_mix = criterion_mix(mix_pred, mix_labels)
                loss = loss_clf + loss_mix
            else:
                loss = loss_clf

            '''
            if args.mixup:
                mix_imgs, mix_labels = mixup_fn(x, label)
                mix_pred = net(mix_imgs)
                loss = criterion_mix(mix_pred, mix_labels)
                prediction = mix_pred.max(1)[1]
            else:
                y_pred = net(x)         
                loss = criterion(y_pred, label)
                prediction = y_pred.max(1)[1]
            '''

            total_loss += loss.item()
            total += x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += prediction.eq(label).sum().item()

            count += 1           
        
            progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f '
                            % (total_loss / (i + 1), 100. * correct / total))
           
        test_log.write('Epoch:%d  lr:%.5f  Loss:%.4f  Acc:%.4f \n'%(epoch, lr, total_loss / count, correct/total))
        test_log.flush()  

        if (epoch+1)%val_epoch == 0:
            main_val(net, valloader, epoch, test_log, optimizer)

@torch.no_grad()
def main_val(net, valloader, epoch, test_log, optimizer):
    global best_acc
    global best_kappa

    net = net.eval()
    correct = .0
    total = .0
    count = .0
    con_matx = meter.ConfusionMeter(args.n_classes)

    pred_list = []
    predicted_list = []
    label_list = []

    for i, (x, label, id) in enumerate(valloader):
        x = x.float().cuda()
        label = label.cuda()

        y_pred = net(x)
        con_matx.add(y_pred.detach(),label.detach())

        predicted_list.extend(y_pred.squeeze(-1).cpu().detach())
        label_list.extend(label.cpu().detach())
        pred = y_pred.max(1)[1]
        pred_list.extend(pred.cpu().detach())

        progress_bar(i, len(valloader))

    kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
    acc = accuracy_score(np.array(label_list), np.array(pred_list))
    print('val epoch:', epoch, ' acc: ', acc, 'kappa: ', kappa, 'con: ', str(con_matx.value()))
    test_log.write('Val Epoch:%d   Accuracy:%.4f   kappa:%.4f  con:%s \n'%(epoch,acc, kappa, str(con_matx.value())))
    test_log.flush()

    if kappa > best_kappa:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'kappa': kappa,
            'epoch': epoch,
            'optimizer': optimizer
        }
        save_name = os.path.join(save_dir, str(epoch) + '.pkl')
        torch.save(state, save_name)
        best_kappa = kappa



if __name__ == '__main__':
    parse_args()
    main()
