from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_mem_2 import embed_net
from utils import *
import pdb
from re_rank import random_walk, k_reciprocal

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=7, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')

parser.add_argument('--share_net', default=2, type=int,
                    metavar='share', help='[1,2,3,4]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='on', type=str, help='performing PCB, on or off')

parser.add_argument('--w_center', default=1.0, type=float, help='the weight for center loss')

parser.add_argument('--local_feat_dim', default=256, type=int,
                    help='feature dimention of each local feature in PCB')
parser.add_argument('--num_strips', default=6, type=int,
                    help='num of local strips in PCB')

parser.add_argument('--label_smooth', default='on', type=str, help='performing label smooth or not')
             
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = 'E:\chenfeng\dataset\SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
elif dataset =='regdb':
    data_path = 'E:\chenfeng\dataset\RegDB/'
    n_class = 206
    test_mode = [2, 1]
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
if args.pcb == 'on':
    pool_dim = args.num_strips * args.local_feat_dim
else:
    pool_dim = 2048
print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch, share_net=args.share_net, pcb=args.pcb, local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch, share_net=args.share_net, pcb=args.pcb)
net.to(device)    
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()



def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat_pool = net(input, input, test_mode[0])
                gall_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            else:
                feat_pool, feat_fc = net(input, input, test_mode[0])
                gall_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
                gall_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    if args.pcb == 'on':
        return gall_feat_pool
    else: 
        return gall_feat_pool, gall_feat_fc
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat_pool = net(input, input, test_mode[1])
                query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            else:    
                feat_pool, feat_fc = net(input, input, test_mode[1])
                query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
                query_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    if args.pcb == 'on':
        return query_feat_pool
    else:
        return query_feat_pool, query_feat_fc


if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    
    # model_path = checkpoint_path + args.resume
    model_path = checkpoint_path + 'sysu_c_tri_pcb_on_w_tri_1.0_s6_f256_share_net2_base_gm_k8_p6_lr_0.1_seed_0_best.t'
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
                .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    if args.pcb == 'on':
        query_feat_pool = extract_query_feat(query_loader)
    else:
        query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    for trial in range(10):
        print('Test Trial: {}'.format(trial))
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        if args.pcb == 'on':
            gall_feat_pool = extract_gall_feat(trial_gall_loader)
        else:
            gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

        if args.re_rank == 'random_walk':
            distmat_pool = random_walk(query_feat_pool, gall_feat_pool)
            if args.pcb == 'off': distmat = random_walk(query_feat_fc, gall_feat_fc)
        elif args.re_rank == 'k_reciprocal':
            distmat_pool = k_reciprocal(query_feat_pool, gall_feat_pool)
            if args.pcb == 'off': distmat = k_reciprocal(query_feat_fc, gall_feat_fc)
        elif args.re_rank == 'no':
            # compute the similarity
            distmat_pool = -np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            if args.pcb == 'off': distmat = -np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        # pool5 feature
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(distmat_pool, query_label, gall_label, query_cam, gall_cam)
        
        if args.pcb == 'off': 
            # fc feature
            cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            if args.pcb == 'off': 
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            if args.pcb == 'off': 
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        
        if args.pcb == 'off': 
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))


elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial +1
        print('Test Trial: {}'.format(test_trial))
        #model_path = checkpoint_path + 'regdbtest_share_net2_base_gm_p4_n8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)   
        model_path = checkpoint_path + 'regdb_c_tri_pcb_on_w_tri_2.0_s6_f256_share_net2_base_gm10_k4_p8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])

        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

        if args.pcb == 'on':
            query_feat_pool = extract_query_feat(query_loader)
            gall_feat_pool = extract_gall_feat(gall_loader)
        else:
            query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
            gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader)

        if args.tvsearch:
            if args.re_rank == 'random_walk':
                distmat_pool = random_walk(gall_feat_pool, query_feat_pool)
                if args.pcb == 'off': distmat = random_walk(gall_feat_fc, query_feat_fc)
            elif args.re_rank == 'k_reciprocal':
                distmat_pool = k_reciprocal(gall_feat_pool, query_feat_pool)
                if args.pcb == 'off': distmat = k_reciprocal(gall_feat_fc, query_feat_fc)
            elif args.re_rank == 'no':
                # compute the similarity
                distmat_pool = -np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
                if args.pcb == 'off': distmat = -np.matmul(gall_feat_fc, np.transpose(query_feat_fc))
            # pool5 feature
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(distmat_pool, gall_label, query_label)
            if args.pcb == 'off': 
                # fc feature
                cmc, mAP, mINP = eval_regdb(distmat,gall_label,  query_label )
        else:
            if args.re_rank == 'random_walk':
                distmat_pool = random_walk(query_feat_pool, gall_feat_pool)
                if args.pcb == 'off': distmat = random_walk(query_feat_fc, gall_feat_fc)
            elif args.re_rank == 'k_reciprocal':
                distmat_pool = k_reciprocal(query_feat_pool, gall_feat_pool)
                if args.pcb == 'off': distmat = k_reciprocal(query_feat_fc, gall_feat_fc)
            elif args.re_rank == 'no':
                # compute the similarity
                distmat_pool = -np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
                if args.pcb == 'off': distmat = -np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            # pool5 feature
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(distmat_pool, query_label, gall_label)
            if args.pcb == 'off': 
                # fc feature
                cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)


        if trial == 0:
            if args.pcb == 'off': 
                all_cmc = cmc
                all_mAP = mAP
                all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            if args.pcb == 'off': 
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        if args.pcb == 'off': 
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
if args.pcb == 'off': 
    cmc = all_cmc / 10
    mAP = all_mAP / 10

cmc_pool = all_cmc_pool / 10
mAP_pool = all_mAP_pool / 10
print('All Average:')

if args.pcb == 'off': 
    print(
        'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))