from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np


#
class MemoryUnit(nn.Module):
    def __init__(self, ptt_num, num_cls, part_num,fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        '''
        the instance PTT is divided into cls_number x ptt_number per cls x part number per ptt
        '''
        self.num_cls = num_cls
        self.ptt_num = ptt_num
        self.part_num = part_num
        
        self.mem_dim = ptt_num * num_cls * part_num # M
        self.fea_dim = fea_dim # C
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        #self.sem_weight = Parameter(torch.Tensor(self.num_cls, self.fea_dim)) # N x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_update_query(self, mem, max_indices, score, query):
        m, d = mem.size()

        query_update = torch.zeros((m,d)).cuda()
        #random_update = torch.zeros((m,d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1)==i)
            a, _ = idx.size()
            #ex = update_indices[0][i]
            if a != 0:
                #random_idx = torch.randperm(a)[0]
                #idx = idx[idx != ex]
#                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
                #random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
            else:
                query_update[i] = 0 
                #random_update[i] = 0
    
    
        return query_update        

    def forward(self, input, residual=False):
        '''
        this is a bottom-up hierarchical stastic and summaration module
        all steps in main flow follow  part -> prototype -> cls
        input = NHW x C
        total PTT M =  num_cls (L) x ptt_num (T) x part_num (P)
        dimension C = fea_dim
        '''
        ### for global part-unware instance PTT, act as sub flow
        att_weight = F.linear(input, self.weight)  # we doesn't split the part dimension, there it is part-unaware NHW x M
        att_weight = F.softmax(att_weight, dim=1)  # NHW x M
        ### update ###
        #_, gather_indice = torch.topk(att_weight, 1, dim=1)
        #ins_mem_sample_driven = self.get_update_query(self.weight, gather_indice, att_weight,input)
        #self.weight.data = F.normalize(ins_mem_sample_driven+ self.weight, dim=1)

        if self.shrink_thres >0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC



        #return {'output': output, 'att': att_weight}  # output, att_weight
        return {'output': output, 'att': None,'sem_attn': self.weight}
        

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, ptt_num, num_cls, part_num, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.ptt_num = ptt_num
        self.num_cls = num_cls
        self.part_num = part_num
        ins_mem= False
        if ins_mem:
            self.mem_dim = ptt_num * num_cls * part_num# part-level instance
        else:
            self.mem_dim = num_cls# global semantic
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.ptt_num, self.num_cls, self.part_num, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            #att = att.view(s[0], s[2], s[3], self.mem_dim)
            #att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return y, y_and['sem_attn']

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

