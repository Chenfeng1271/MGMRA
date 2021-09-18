from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
#from model_main import embed_net
from model_mem import embed_net
from utils import *
from loss import OriTripletLoss, HcTripletLoss, CrossEntropyLabelSmooth, EntropyLossEncap, BarlowTwins_loss_mem, MemTriLoss
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.3 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log_ddag/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=6, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=10, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--cpool', default='no', type=str, help='The coarse branch pooling: no | wpa | avg | max | gem')
parser.add_argument('--bpool', default='avg', type=str, help='The backbone (fine branch) pooling: avg | max | gem')
parser.add_argument('--label_smooth', default='off', type=str, help='performing label smooth or not')
parser.add_argument('--hcloss', default='HcTri', type=str, help='OriTri, HcTri')
parser.add_argument('--margin_hc', default=0, type=float,
                    metavar='margin', help='additional hc triplet loss margin')
parser.add_argument('--fuse', default='sum', type=str, help='sum | cat')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    # TODO: define your data path
    data_path = 'E:\chenfeng\dataset\SYSU-MM01/'
    log_path = os.path.join(args.log_path, 'sysu_log_ddag/')
    test_mode = [1, 2] # infrared to visible
elif dataset =='regdb':
    # TODO: define your data path for RegDB dataset
    data_path = 'E:\chenfeng\dataset\RegDB/'
    log_path = os.path.join(args.log_path, 'regdb_log_ddag/')
    test_mode = [2, 1] # visible to infrared

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# log file name
suffix = dataset+'_bpool_{}_cpool_{}_hcloss_{}_fuse_{}'.format(args.bpool,args.cpool,args.hcloss,args.fuse)   #c2f:coarse to fine  sm: simple module

suffix = suffix  + '_hcmargin_{}'.format(args.margin_hc) + '_gm_ls_{}_s1'.format(args.label_smooth)  # ls: label_smooth  

if args.cpool == 'wpa':
    suffix = suffix + '_P_{}'.format(args.part)
suffix = suffix + '_drop_{}_{}_{}_lr_{}_seed_{}'.format(args.drop, args.num_pos, args.batch_size, args.lr, args.seed)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim
if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

feature_dim = 2048
feature_dim_att = 2048 if args.fuse == "sum" else 4096

end = time.time()

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),                                                             
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])


if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


n_class = len(np.unique(trainset.train_color_label))

nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(n_class, drop=args.drop, part=args.part, arch=args.arch, cpool=args.cpool,bpool=args.bpool,fuse=args.fuse)
net.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
if args.label_smooth == 'on':
    criterion1 = CrossEntropyLabelSmooth(n_class)
else:
    criterion1 = nn.CrossEntropyLoss()
loader_batch = args.batch_size * args.num_pos
criterion2 = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
#criterion2 = HcTripletLoss(batch_size=loader_batch, margin=args.margin)
if args.hcloss == 'OriTri':
    criterion_hc = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
if args.hcloss == 'HcTri':
    criterion_hc = HcTripletLoss(batch_size=loader_batch, margin=args.margin+args.margin_hc)
if args.hcloss == 'no':
    pass
criterion1.to(device)
criterion2.to(device)
if args.hcloss != 'no':
    criterion_hc.to(device)

# memory att update
tr_entropy_loss_func = BarlowTwins_loss_mem()
tri_mem_loss_fuc = MemTriLoss()
l1_mem_loss_func = nn.SmoothL1Loss()

# optimizer
if args.optim == 'sgd':
    if args.cpool != 'no':
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                        + list(map(id, net.classifier.parameters())) \
                        + list(map(id, net.classifier_att.parameters())) \
                        + list(map(id, net.cpool_layer.parameters())) 

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer_P = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr},
            {'params': net.classifier_att.parameters(), 'lr': args.lr},
            {'params': net.cpool_layer.parameters(), 'lr': args.lr},
            ],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters())) 
                     
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer_P = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr},
            ],
            weight_decay=5e-4, momentum=0.9, nesterov=True)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer_P, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 20:
        lr = args.lr
    elif 20 <= epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr


def train(epoch):
    # adjust learning rate
    current_lr = adjust_learning_rate(optimizer_P, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    tri_mem_loss = AverageMeter()
    ce_mem_loss = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)
        
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        if args.cpool != 'no':
            # Forward into the network
            feat, out0, feat_att, out_att, att_mem, feat_mem = net(input1, input2)
            # Part attention loss
            loss_p = criterion1(out_att, labels)
            if args.hcloss != 'no': 
                loss_p_hc, _ = criterion_hc(feat_att, labels)
        else:
            # Forward into the network
            feat, out0, att_mem, feat_mem, x_mem_feat, out_mem = net(input1, input2)
            loss_mem_br_cls = criterion1(out_mem, labels.long())
            loss_mem_br_tri,_ = criterion2(x_mem_feat, labels) 


        # baseline loss: identity loss + triplet loss Eq. (1)
        loss_id = criterion1(out0, labels.long())
        loss_tri, batch_acc = criterion2(feat, labels)
        # loss mem att
        loss_mem = tr_entropy_loss_func(att_mem)
        loss_mem_tri,_ = tri_mem_loss_fuc(feat_mem,labels,att_mem)
        #att_mem_c_1 , att_mem_c_2 = att_mem_c.chunk(2,dim=0)
        #loss_mem_c = l1_mem_loss_func(att_mem_c_1, att_mem_c_2)
        #loss_hc, _ = criterion_hc(feat, labels)
        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)
        
        if args.cpool != 'no':
            # Instance-level part-aggregated feature learning Eq. (10)
            if args.hcloss != 'no':
                loss = loss_id + loss_tri + loss_p + loss_p_hc 
            else:
                loss = loss_id + loss_tri + loss_p
        else:
            loss = loss_id + loss_tri #+ loss_hc
        
        loss = loss + loss_mem + loss_mem_tri + loss_mem_br_cls * 0.1 + loss_mem_br_tri #+ loss_mem_c
        #loss = loss + loss_mem_tri

        # optimization
        optimizer_P.zero_grad()
        loss.backward()
        optimizer_P.step()

        # log different loss components
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        tri_mem_loss.update(loss_mem_tri.item(), 2 * input1.size(0))
        ce_mem_loss.update(loss_mem.item(),2 * input1.size(0))
        #graph_loss.update(loss_G.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.2f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'TriMem: {trimem.val:.4f} ({trimem.avg:.4f}) '
                  'CeMem: {cemem.val:.4f} ({cemem.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                   epoch, batch_idx, len(trainloader), current_lr,
                   100. * correct / total, batch_time=batch_time,
                   train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, trimem = tri_mem_loss, cemem=ce_mem_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    #writer.add_scalar('graph_loss', graph_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    # computer wG
    #return 1. / (1. + train_loss.avg)

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feature_dim))
    gall_feat_att = np.zeros((ngall, feature_dim_att))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.cpool != 'no':
                feat, feat_att = net(input, input, test_mode[0])
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy() 
            else:
                 feat, x_mem_feat = net(input, input, test_mode[0])   
                 gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()   
                 gall_feat_att[ptr:ptr + batch_num, :] = x_mem_feat.detach().cpu().numpy() 
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feature_dim))
    query_feat_att = np.zeros((nquery, feature_dim_att))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.cpool != 'no':
                feat, feat_att = net(input, input, test_mode[1])
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            else:
                feat, x_mem_feat = net(input, input, test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = x_mem_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    if args.cpool != 'no':
        distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        if args.cpool != 'no':
            cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if args.cpool != 'no':
            cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    if args.cpool != 'no':
        writer.add_scalar('rank1_att', cmc_att[0], epoch)
        writer.add_scalar('mAP_att', mAP_att, epoch)
        writer.add_scalar('mAP_att', mAP_att, epoch)
        writer.add_scalar('mINP_att', mINP_att, epoch)
        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att
    else:
        return cmc, mAP, mINP


# training
print('==> Start Training...')
for epoch in range(start_epoch, 61 if args.dataset == 'regdb' else 61  - start_epoch):# default regdb 31

    print('==> Preparing Data Loader...')
    # identity sampler: 
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # infrared index
    '''print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)'''

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % 5 == 0:
        print('Test Epoch: {}'.format(epoch))

        if args.cpool != 'no':
            # testing
            cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
            # log output   FC: f_bn, the fine branch feature    FC_att: f_bnf, the coarse branch feature
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            
            print('FC_att:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        
        else:
            # testing
            cmc, mAP, mINP = test(epoch)
            # log output
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            
        # save model
        if args.cpool != 'no':
            if cmc_att[0] >= best_acc:  # not the real best for sysu-mm01
                best_acc = cmc_att[0]
                best_epoch = epoch
                best_mAP = mAP_att
                best_mINP = mINP_att
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc_att,
                    'mAP': mAP_att,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')
        else:
            if cmc[0] >= best_acc:  # not the real best for sysu-mm01
                best_acc = cmc[0]
                best_epoch = epoch
                best_mAP = mAP
                best_mINP = mINP
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')
        
        print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP, best_mINP))