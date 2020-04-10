# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:15:28 2019

@author: Mirac
"""

import torch
import torch.nn.functional as F
import numpy as np

class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, inter_class_weights, intra_class_weights, focal_power=1):
        super(WeightedFocalLoss, self).__init__()
        self.focal_power = focal_power
        self.inter_class_weights = torch.cuda.FloatTensor(inter_class_weights)
        self.intra_class_weights = torch.cuda.FloatTensor(intra_class_weights)
     
    def forward(self, output, target):
        probs = torch.sigmoid(output)
        scores = torch.where(target == 1, probs, 1 - probs)
        
        #cost_terms = F.binary_cross_entropy(probs, target, reduction='none')
        cost_terms = F.binary_cross_entropy_with_logits(output, target, reduction='none', pos_weight=self.intra_class_weights)
        focal_cost_terms = torch.pow((1 - scores), self.focal_power) * cost_terms
        
        task_losses = torch.mean(focal_cost_terms, dim=0)
        total_task_loss = torch.mean(task_losses * self.inter_class_weights)
        
        return total_task_loss

    
class FocalGradNormLoss(torch.nn.Module):
    
    def __init__(self, model, N_LABELS, alpha=1, lr=0.1, focal_power=1):
        super(FocalGradNormLoss, self).__init__()
        self.inter_class_weights = torch.cuda.FloatTensor(np.ones(N_LABELS))
        self.inter_class_weights.requires_grad_(True)
        self.L_zeros = None
        self.alpha = alpha
        self.lr = lr
        self.focal_power = focal_power
        
        self.last_shared_weights = model.module.features[-2]['denselayer16'].conv2.weight
        

    def forward(self, output, target, model=None):
        probs = torch.sigmoid(output)
        scores = torch.where(target == 1, probs, 1 - probs)
        
        cost_terms = F.binary_cross_entropy(probs, target, reduction='none')
        focal_cost_terms = torch.pow((1 - scores), self.focal_power) * cost_terms
        task_losses = torch.mean(focal_cost_terms, dim=0)
        total_task_loss = torch.mean(task_losses * self.inter_class_weights.detach())
        
        
        if task_losses.requires_grad:
            
            if self.L_zeros is None:
                self.L_zeros = task_losses.detach();
            
            G_tasks = torch.cuda.FloatTensor(np.zeros(len(task_losses)));
            G_tasks.requires_grad_(True);
            for task_num, task_loss in enumerate(task_losses):
                task_loss_grad_norm = torch.norm(torch.autograd.grad(task_loss, self.last_shared_weights, retain_graph=True, only_inputs=True)[0]).detach();
                G_tasks[task_num] =  self.inter_class_weights[task_num] * task_loss_grad_norm;
                
            G_avg = torch.mean(G_tasks);
            
            L_tildas = task_losses / self.L_zeros;
            L_tilda_avg = torch.mean(L_tildas);
            inv_training_rates = L_tildas / L_tilda_avg; #r_i's in the paper
            
            targets = (G_avg * torch.pow(inv_training_rates, self.alpha)).detach();
            total_grad_loss = torch.abs(G_tasks - targets).sum(); #L_grad in the paper
            
            inter_class_weight_grads = torch.autograd.grad(total_grad_loss, self.inter_class_weights, retain_graph=True, only_inputs=True)[0];
            self.inter_class_weights = self.inter_class_weights - self.lr * inter_class_weight_grads
            self.inter_class_weights = self.inter_class_weights * 14 / self.inter_class_weights.sum()
        
        return total_task_loss    


class EffectiveNumGradNormLoss(torch.nn.Module):
    
    def __init__(self, model, pos_weight, alpha=1, lr=0.1):
        super(EffectiveNumGradNormLoss, self).__init__()
        self.intra_class_weights = torch.cuda.FloatTensor(pos_weight)
        #self.inter_class_weights = torch.cuda.FloatTensor(np.ones((1,len(pos_weight))))
        self.inter_class_weights = torch.cuda.FloatTensor(np.ones(len(pos_weight)))
        self.inter_class_weights.requires_grad_(True)
        self.L_zeros = None
        self.alpha = alpha
        self.lr = lr
        
        self.last_shared_weights = model.module.features[-2]['denselayer16'].conv2.weight
        

    def forward(self, output, target, model=None):
        cost_terms = F.binary_cross_entropy_with_logits(output, target, pos_weight=self.intra_class_weights, reduction='none')
        task_losses = torch.mean(cost_terms, dim=0)
        total_task_loss = torch.mean(task_losses * self.inter_class_weights.detach())
        
        
        if task_losses.requires_grad:
            
            if self.L_zeros is None:
                self.L_zeros = task_losses.detach();
            
            G_tasks = torch.cuda.FloatTensor(np.zeros(len(task_losses)));
            G_tasks.requires_grad_(True);
            for task_num, task_loss in enumerate(task_losses):
                task_loss_grad_norm = torch.norm(torch.autograd.grad(task_loss, self.last_shared_weights, retain_graph=True, only_inputs=True)[0]).detach();
                G_tasks[task_num] =  self.inter_class_weights[task_num] * task_loss_grad_norm;
                
            G_avg = torch.mean(G_tasks);
            
            L_tildas = task_losses / self.L_zeros;
            L_tilda_avg = torch.mean(L_tildas);
            inv_training_rates = L_tildas / L_tilda_avg; #r_i's in the paper
            
            targets = (G_avg * torch.pow(inv_training_rates, self.alpha)).detach();
            total_grad_loss = torch.abs(G_tasks - targets).sum(); #L_grad in the paper
            
            inter_class_weight_grads = torch.autograd.grad(total_grad_loss, self.inter_class_weights, retain_graph=True, only_inputs=True)[0];
            self.inter_class_weights = self.inter_class_weights - self.lr * inter_class_weight_grads
            self.inter_class_weights = self.inter_class_weights * 14 / self.inter_class_weights.sum()
        
        return total_task_loss

class BCEwithIgnore(torch.nn.Module):
    def _init_(self):
        super(BCEwithIgnore, self)._init_()
    
    def forward(self, score, y):
        zeros = torch.zeros_like(y)
        ones  = torch.ones_like(y)
        num_uncertain = torch.sum(torch.where(y==-1, ones, zeros))
        positive = torch.where(y==-1, zeros, y)
        negative = torch.where(y==-1, ones, y)
        p = torch.log(score)
        one_minus_p = torch.log(1 - score)
        loss = -1 * torch.sum(p * positive + (1-negative) * one_minus_p) / ( y.numel() - num_uncertain)
        return loss
      
      
class WeightedBCE(torch.nn.Module):
    def __init__(self, weight):
        super(WeightedBCE, self).__init__()
        self.w = weight
    
    def forward(self, score, y):
        loss = -1 * torch.mean(y * torch.log(score) * self.w +\
                               (1-y) * torch.log(1 - score) * (2 - self.w)) 
        
        return loss
      

class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, weight):
        super(WeightedCrossEntropy, self).__init__()
        self.w = weight
    
    def forward(self, output, y):
        scores = torch.softmax(output.view(-1, 3), dim=1)
        y = y.view(-1, 1)
        y_onehot = torch.zeros((y.shape[0], 3), device=y.device)
        y_onehot.scatter_(1, y, 1)
        weights = self.w.repeat(len(y)// 14 , 1)
        loss = - torch.mean(torch.log(scores).type(torch.double) * y_onehot.type(torch.double) * weights.type(torch.double)) 
        
        return loss