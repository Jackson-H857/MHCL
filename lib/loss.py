import torch
import torch.nn as nn
import torch.nn.functional as F


def pos_neg_mask(labels):

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) 
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    return pos_mask, neg_mask


def pos_neg_mask_xy(labels_col, labels_row):

    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1)) 
    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1))

    return pos_mask, neg_mask


def loss_select(opt, loss_type='vse'):

    if loss_type == 'vse':
        # the default loss
        criterion = ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
    elif loss_type == 'trip':
        # Triplet loss with the distance-weight sampling
        criterion = TripletLoss(opt=opt)
    else:
        raise ValueError('Invalid loss {}'.format(loss_type))
    
    return criterion

def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


class HyHCL(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.l_alpha = opt.mu
        self.l_ep = opt.gama
        
        self.temperature = opt.temperature
        self.use_infonce = opt.use_infonce
        
        self.hubness_weight = opt.hubness_weight  
        self.infonce_weight = opt.infonce_weight  

    def forward(self, im, s):
        
        bsize = im.size()[0]
        scores = get_sim(im, s)
        
        # 提取正样本对的相似度
        tmp = torch.eye(bsize).cuda()
        s_diag = tmp * scores
        scores_ = scores - s_diag
        
        
        S_ = torch.exp((scores_ - self.l_ep) * self.l_alpha)
        loss_diag = -torch.log(1 + F.relu(s_diag.sum(0)))
        hubness_loss = torch.sum(
            torch.log(1 + S_.sum(0)) / self.l_alpha +
            torch.log(1 + S_.sum(1)) / self.l_alpha +
            loss_diag
        ) / bsize

        return hubness_loss * self.hubness_weight


    def moco_forward(self, v_q, t_k, t_q, v_k, v_queue, t_queue):
        # 图像->文本方向的InfoNCE
        i2t_pos = torch.einsum('nc,nc->n', [v_q, t_k]).unsqueeze(-1)
        t_queue = t_queue.clone().detach()
        i2t_neg = torch.einsum('nc,ck->nk', [v_q, t_queue])
        i2t_logits = torch.cat([i2t_pos, i2t_neg], dim=1)
        i2t_logits = i2t_logits / self.temperature
        i2t_labels = torch.zeros(i2t_logits.shape[0], dtype=torch.long).cuda()
        
        # 文本->图像方向的InfoNCE
        t2i_pos = torch.einsum('nc,nc->n', [t_q, v_k]).unsqueeze(-1)
        v_queue = v_queue.clone().detach()
        t2i_neg = torch.einsum('nc,ck->nk', [t_q, v_queue])
        t2i_logits = torch.cat([t2i_pos, t2i_neg], dim=1)
        t2i_logits = t2i_logits / self.temperature
        t2i_labels = torch.zeros(t2i_logits.shape[0], dtype=torch.long).cuda()

        infonce_loss = F.cross_entropy(i2t_logits, i2t_labels) + F.cross_entropy(t2i_logits, t2i_labels)
        
        return infonce_loss * self.infonce_weight



class ContrastiveLoss(nn.Module):

    def __init__(self, opt, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = opt.mask_repeat

        self.false_hard = []

    def max_violation_on(self):
        self.max_violation = True
        # print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        # print('Use VSE0 objective.')

    def forward(self, im, s, img_ids=None):
        # 确保输入特征已经归一化
        #im = F.normalize(im, p=2, dim=1)
        #s = F.normalize(s, p=2, dim=1)

        # compute image-sentence score matrix
        scores = get_sim(im, s)
        
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval, i->t
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval t->i
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            img_ids = img_ids.cuda()
            mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0))

        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        
        #修改损失计算方式，使用平均损失而不是总和
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        # 计算平均损失
        loss = (cost_s.sum() + cost_im.sum()) / im.size(0)

        return loss


def get_sim(images, captions):
    
    #images = F.normalize(images, p=2, dim=1)
    #captions = F.normalize(captions, p=2, dim=1)
    # 计算余弦相似度
    similarities = images.mm(captions.t())
    return similarities


# Triplet loss + DistanceWeight Miner
# Sampling Matters in Deep Embedding Learning, ICCV, 2017
# more information refer to https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer
class TripletLoss(nn.Module):

    def __init__(self, opt=None, margin=0.2, ):
        super().__init__()

        self.opt = opt
        self.margin = margin
        
        self.cut_off = 0.5
        self.d = 512

        if opt.dataset == 'coco':
            self.nonzero_loss_cutoff = 1.9         
        else:
            self.nonzero_loss_cutoff = 1.7
        
    def forward(self, im, s, img_ids):
        
        #print(f"img_emb range: [{im.min()}, {im.max()}]")
        #print(f"cap_emb range: [{s.min()}, {s.max()}]")

        sim_mat = get_sim(im, s)
        img_ids = img_ids.cuda()

        if im.size(0) == s.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t())

        loss = loss_im + loss_s

        return loss        

    def loss_forward(self, sim_mat, pos_mask, neg_mask): 

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask  
        log_weight[inf_or_nan] = 0.      

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float() 
     
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
        weight = weight[anchor_idx]

        # maybe not exist
        try:
            neg_idx = torch.multinomial(weight, 1).squeeze(1)   
        except Exception:
            return torch.zeros([], requires_grad=True, device=sim_mat.device) 


        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]  

        loss = F.relu(self.margin + s_an - s_ap) 
        loss = loss.sum() 

        return loss


if __name__ == '__main__':

    pass
    