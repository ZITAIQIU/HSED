"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 12/12/2021
"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from scipy import sparse

class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        

    def arcosh(self, x):
        y = torch.log(x+torch.sqrt(x**2-1))
        return y

    def hyperbolic_similarity(self, u, v):
        a = torch.norm(u-v, p=2, dim=1)**2
        b = 1 - torch.norm(u, p=2, dim=1)**2
        c = 1 - torch.norm(v, p=2, dim=1)**2
        r = 1 + torch.div(2*a, torch.mul(b, c))
        s = self.arcosh(r)**(-1)
  
        return s

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        #print('features1 shape:',features_1.shape)
        features= torch.cat([features_1, features_2], dim=0)
        #print('features shape:', features.shape)
        #print('Is features have inf?', torch.isinf(features).any())
        
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        #mask = torch.eye(batch_size, dtype=torch.bool)
        mask = mask.repeat(2, 2).cpu()
        mask = ~mask
        #print('contrastive loss mask finished')
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        # change inf to 1 Nan to 0
        
        if torch.isnan(pos).any():
           # print('pos have nan')
            pos = torch.where(torch.isnan(pos), torch.full_like(pos, 0), pos)
        if torch.isinf(pos).any():
            #print('pos have inf')
            pos = torch.where(torch.isinf(pos), torch.full_like(pos, 1), pos)
            #print('remove all inf in pos')

        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        #neg = torch.exp((features @ features.t().contiguous()) / self.temperature)
        if torch.isnan(neg).any():
            #print('neg have nan')
            neg = torch.where(torch.isnan(neg), torch.full_like(neg, 0), neg)
        if torch.isinf(neg).any():
            #print('neg have inf')
            neg = torch.where(torch.isinf(neg), torch.full_like(neg, 1), neg)
            #print('remove all inf in neg')
        neg = neg.masked_select(mask).view(2*batch_size, -1)
       
        
        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        """print('pos sample finished:',pos.shape)
        print('neg sample finished:',neg.shape)
        print('neg mean:', neg_mean)
        print('pos n:', pos_n)
        print('ng:',Ng)
        print('contrastive loss:', loss_pos)"""
        
        return {"loss":loss_pos, "pos_mean":pos_n.detach().cpu().numpy(), "neg_mean":neg_mean.detach().cpu().numpy(), "pos":pos.detach().cpu().numpy(), "neg":neg.detach().cpu().numpy()}
            

    