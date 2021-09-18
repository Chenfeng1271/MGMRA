import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
from memory_module_MGMRA import MemModule
import random

##此版本为使用memory做part feature

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):               
                    setattr(self.visible,'layer'+str(i), getattr(model_v,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer'+str(i))(x)
            return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net
        
        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):               
                    setattr(self.thermal,'layer'+str(i), getattr(model_t,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):           
                    x = getattr(self.thermal, 'layer'+str(i))(x)             
            return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net       
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base,'layer'+str(i), getattr(model_base,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            return x
        elif self.share_net > 4:
            return x
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer'+str(i))(x)
            return x



class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'off', gm_pool = 'on', arch='resnet50', share_net=1, pcb='on',local_feat_dim=256, num_strips=6):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch, share_net=share_net)
        self.visible_module = visible_module(arch=arch, share_net=share_net)
        self.base_resnet = base_resnet(arch=arch, share_net=share_net)

        self.non_local = no_local
        self.pcb = pcb
        if self.non_local =='on':
            pass


        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.gm_pool = gm_pool

        ##memory module 
        self.mem_rep = MemModule(ptt_num=5, num_cls=206, part_num=6, fea_dim=pool_dim, shrink_thres =0.0025)
        self.pool_mem = nn.AdaptiveAvgPool2d((1,1))
        self.bn = nn.BatchNorm2d(pool_dim)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck.apply(weights_init_kaiming)

        if self.pcb == 'on':
            self.num_stripes=num_strips
            local_conv_out_channels=local_feat_dim

            self.local_conv_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_list.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))
            
            self.fc_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list.append(fc)
            
        
        else:
            self.bottleneck = nn.BatchNorm1d(pool_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift

            self.classifier = nn.Linear(pool_dim, class_num, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            

        

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        if self.non_local == 'on':
            pass
        else:
            x = self.base_resnet(x)

        ## memory module 
        #x_mem, att_mem = self.mem_rep(x)
        #x_mem += x
        #x_mem_pool = self.pool_mem(x_mem).view(x_mem.size(0), x_mem.size(1))
        #x_mem_feat = self.bottleneck(x_mem_pool)


        if self.pcb == 'on':
            feat = x
            assert feat.size(2) % self.num_stripes == 0
            stripe_h = int(feat.size(2) / self.num_stripes)
            local_feat_list = []
            logits_list = []
            local_feat_mem_list = []
            local_feat_mem_ins_list = []
            local_feat_mem_part_list = []
            for i in range(self.num_stripes):
                # shape [N, C, 1, 1]
                
                # average pool
                #local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                if self.gm_pool  == 'on':
                    # gm pool
                    local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
                    local_feat_sem, local_feat_ins, local_feat_part = self.mem_rep(local_feat)

                    local_feat_mem_part_list.append(local_feat_part)
                    local_feat_mem_ins_list.append(local_feat_ins)
                    local_feat_mem = local_feat + local_feat_sem
                    b, c, h, w = local_feat.shape
                    local_feat = local_feat.view(b,c,-1)
                    p = 10.0    # regDB: 10.0    SYSU: 3.0
                    local_feat = (torch.mean(local_feat**p, dim=-1) + 1e-12)**(1/p)
                else:
                    # average pool
                    #local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                    local_feat = F.max_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                

                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list[i](local_feat.view(feat.size(0),feat.size(1),1,1))
               

                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list.append(local_feat)
                local_feat_mem_list.append(local_feat_mem)


                if hasattr(self, 'fc_list'):
                    logits_list.append(self.fc_list[i](local_feat))



            feat_all = [lf for lf in local_feat_list]
            feat_all = torch.cat(feat_all, dim=1)


            feat_all_mem = [lf for lf in local_feat_mem_list]
            feat_all_mem = torch.cat(feat_all_mem, dim=2)

            lf_mem_pool = self.pool_mem(feat_all_mem).view(feat_all_mem.size(0), feat_all_mem.size(1))
            lf_mem_feat = self.bottleneck(lf_mem_pool)

            ### this part is for part alignment, we then would change the discription here



            feat_all_part = [lf for lf in local_feat_mem_part_list]
            index = [i for i in range(len(feat_all_part))]
            random.shuffle(index)
            feat_all_part_shuffle = [feat_all_part[i] for i in index]
            feat_all_part_chunk = torch.cat(feat_all_part_shuffle, dim=1)
            p_1, p_2 = torch.chunk(feat_all_part_chunk,2,1)
            
            feat_all_part = [self.pool_mem(lf) for lf in local_feat_mem_part_list]
            feat_all_part = torch.cat(feat_all_part, dim=2)



            

            if self.training:
                #return local_feat_list, logits_list, feat_all , x_mem_pool+ lf_mem_pool, self.classifier(x_mem_feat+lf_mem_feat)
                return local_feat_list, logits_list, feat_all , lf_mem_pool, self.classifier(lf_mem_feat),[p_1,p_2],feat_all_part
            else:
                return self.l2norm(feat_all)
        else:    
            if self.gm_pool  == 'on':
                b, c, h, w = x.shape
                x = x.view(b, c, -1)
                p = 3.0
                x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
            else:
                x_pool = self.avgpool(x)
                x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

            feat = self.bottleneck(x_pool)

            if self.training:
                return x_pool, self.classifier(feat)#, scores
            else:
                return self.l2norm(x_pool), self.l2norm(feat)

