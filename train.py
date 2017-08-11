import os, time
import os.path as osp
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils

def bce2d(input, target):
    n, c, h, w = input.size()
    # assert(max(target) == 1)
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t >0)
    neg_index = (target_t ==0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num*1.0 / sum_num
    weight[neg_index] = pos_num*1.0 / sum_num
    
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy(log_p, target_trans, weight, size_average=False)
    return loss

def rgb_trans(data):
    data = data.numpy()
    data = data[0]
    data = data.transpose(1,2,0) # chw-> hwc
    data += np.array([104.00698793, 116.66876762, 122.67891434])
    # print(data.shape)
    data = data[:, :, ::-1]
    data = data.astype(np.uint8)
    data = Image.fromarray(data, 'RGB')
    return data

def gray_trans(img):
    img = img.numpy()[0][0]*255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *=0.1 

def show_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

class Trainer(object):

    def __init__(self, cuda, generator,  optimizerG, 
                 train_loader, val_loader, out, max_iter):
        self.cuda = cuda

        self.generator = generator
        self.optimG = optimizerG
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)
        self.epoch = 0
        # self.iteration = 0
        # self.max_iter = max_iter
        self.step = 2
        self.nepochs = 8
        self.disp_interval = 100
        self.timeformat = '%Y-%m-%d %H:%M:%S'
        # self.label = Variable(torch.FloatTensor(1).cuda())
        # self.criterion = nn.BCELoss()
        

    def train(self):
        for epoch in range(self.epoch, self.nepochs):
            self.generator.train()
            self.optimG.zero_grad()
            ## adjust hed learning rate
            if (epoch > 0) and (epoch % self.step) == 0:
                adjust_learning_rate(self.optimG)
            show_learning_rate(self.optimG)
            for i, sample in enumerate(self.train_loader, 0):
                data, target = sample 
                if self.cuda:
                    data, target = data.cuda(), target.cuda()

                data, target = Variable(data), Variable(target)
                ## generator forward
                d1, d2, d3, d4, d5, d6 = self.generator(data, target) 
                
                loss1 = bce2d(d1, target)
                loss2 = bce2d(d2, target)
                loss3 = bce2d(d3, target)
                loss4 = bce2d(d4, target)
                loss5 = bce2d(d5, target)
                loss6 = bce2d(d6, target)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                # loss = loss_r1
                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while training')
                loss.backward()
                self.optimG.step()
                self.optimG.zero_grad()
                if (i+1) % self.disp_interval == 0:
                    timestr = time.strftime(self.timeformat, time.localtime())
                    print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch, i, loss.data[0]))
                    print("  loss1: %.3f max1:%.6f min1:%.6f"%(loss1.data[0], d1.data.max(), d1.data.min()))
                    print("  loss2: %.3f max2:%.6f min2:%.6f"%(loss2.data[0], d2.data.max(), d2.data.min()))
                    print("  loss3: %.3f max3:%.6f min3:%.6f"%(loss3.data[0], d3.data.max(), d3.data.min()))
                    print("  loss4: %.3f max4:%.6f min4:%.6f"%(loss4.data[0], d4.data.max(), d4.data.min()))
                    print("  loss5: %.3f max5:%.6f min5:%.6f"%(loss5.data[0], d5.data.max(), d5.data.min()))
                    print("  loss6: %.3f max6:%.6f min6:%.6f"%(loss6.data[0], d6.data.max(), d6.data.min()))
                    
            self.valnet(epoch+1)
            torch.save(self.generator.state_dict(), '%s/netG_epoch_%d.pth' % (self.out, epoch))

    def valnet(self, epoch):
        # eval model on validation set
        print('evaluate...')
        self.generator.eval()
        if os.path.exists(self.out + '/epoch_' + str(epoch)) == False:
            os.mkdir(self.out + '/epoch_' + str(epoch))
        test_lst = open(osp.join(self.dataroot,'test.txt')).readlines()
        save_dir = '%s/epoch_%d/online_val/'%(self.out, epoch)
        for i, sample in enumerate(self.val_loader, 0):
            im_path = test_lst[i].strip().split()[0]
            dirname, imid = osp.split(im_path)
            full_dirname = osp.join(save_dir, dirname)
            if not os.path.exists(full_dirname):
                os.makedirs(full_dirname)
            data, target = sample 
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            d1, d2, d3, d4, d5, d6 = self.generator(data, target)
            # if i == 0:
            #   print('%s/%s_img.png' %(full_dirname, imid.split('.')[0]))
  
            rgb_trans( data.data.cpu()).save('%s/%s_img.png' %(full_dirname, imid.split('.')[0]))
            gray_trans(1-d1.data.cpu()).save('%s/%s_d1.png' % (full_dirname, imid.split('.')[0]))
            gray_trans(1-d2.data.cpu()).save('%s/%s_d2.png' % (full_dirname, imid.split('.')[0]))
            gray_trans(1-d3.data.cpu()).save('%s/%s_d3.png' % (full_dirname, imid.split('.')[0]))
            gray_trans(1-d4.data.cpu()).save('%s/%s_d4.png' % (full_dirname, imid.split('.')[0]))
            gray_trans(1-d5.data.cpu()).save('%s/%s_d5.png' % (full_dirname, imid.split('.')[0]))
            gray_trans(1-d6.data.cpu()).save('%s/%s_d6.png' % (full_dirname, imid.split('.')[0]))
        print('evaluate done')

        self.generator.train()
        
