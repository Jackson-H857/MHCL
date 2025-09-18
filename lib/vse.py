import arguments
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup
import torch.backends.cudnn as cudnn

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import *
import copy
import torch.nn.functional as F
import time
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)




class VSEModel(nn.Module):   #object
    def __init__(self, opt, eval=False):
        super().__init__()   
        self.opt = opt
        self.grad_clip = opt.grad_clip
   
        self.img_enc = get_image_encoder(opt, embed_size=opt.embed_size)
        self.txt_enc = get_text_encoder(opt, opt.embed_size, no_txtnorm=opt.no_txtnorm)

        if opt.use_moco:
            self.K = opt.moco_M   # 2048
            self.m = opt.moco_r   # m
            
            self.v_encoder_k = copy.deepcopy(self.img_enc)
            self.t_encoder_k = copy.deepcopy(self.txt_enc)
            
            for param in self.v_encoder_k.parameters():
                param.requires_grad = False
            for param in self.t_encoder_k.parameters():
                param.requires_grad = False
            
            self.register_buffer("t_queue", torch.rand(opt.embed_size, self.K))
            self.t_queue = F.normalize(self.t_queue, dim=0)
            self.register_buffer("v_queue", torch.rand(opt.embed_size, self.K))
            self.v_queue = F.normalize(self.v_queue, dim=0)
           
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            if opt.use_moco:
                self.v_encoder_k.cuda()
                self.t_encoder_k.cuda()
                self.t_queue = self.t_queue.cuda()
                self.v_queue = self.v_queue.cuda()
                self.queue_ptr = self.queue_ptr.cuda()
            cudnn.benchmark = True
        
        
        self.hyhcl_loss = HyHCL(opt=opt)
        print(self.hyhcl_loss)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.opt = opt

        # 根据预设的优化器类型（AdamW或SGD）和其他超参数设置优化器，
        #    对于文本编码器中的BERT部分，采用较小的学习率
        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    #{'params': bert_params, 'lr': opt.learning_rate * 0.05},
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate*0.1}  
                    ],lr=opt.learning_rate, weight_decay=decay_factor)
        else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        # iteration
        self.Eiters = 0
        self.data_parallel = False
        self.scaler = GradScaler()

    
    # def set_max_violation(self, max_violation=True):
        
    #     if max_violation:
    #         if self.opt.base_loss == 'vse':
    #             self.criterion.base_loss.max_violation_on()
    #         if self.opt.gnn_loss == 'vse':
    #             self.criterion.gnn_loss.max_violation_on()
    #     else:
    #         if self.opt.base_loss == 'vse':
    #             self.criterion.base_loss.max_violation_off()
    #         if self.opt.gnn_loss == 'vse':
    #             self.criterion.gnn_loss.max_violation_off()              

    def set_max_violation(self, max_violation):
        # if max_violation:
        #     self.hal_loss.max_violation_on()
        # else:
        #     self.hal_loss.max_violation_off()

        pass

    def state_dict(self):
        state_dict = [
            self.img_enc.state_dict(), 
            self.txt_enc.state_dict(), 
            ]
        return state_dict

    def load_state_dict(self, state_dict, ):
        # strict=True, ensure keys match
        self.img_enc.load_state_dict(state_dict[0], strict=True)
        
        # Unexpected key(s) in state_dict: "bert.embeddings.position_ids". 
        # incompatible problem of transformers package version 
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.hyhcl_loss.train()

    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.hyhcl_loss.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image/Text encoder is data paralleled (use multi GPUs).')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    # Compute the image and caption embeddings
    def forward_emb(self, images, captions, lengths, image_lengths=None, is_train=False):
        
        #start_img = time.time()
        images = images.cuda()         
        image_lengths = image_lengths.cuda()
        img_emb = self.img_enc(images, image_lengths)
        
        #start_txt = time.time()
        captions = captions.cuda()
        lengths = lengths.cuda()
        cap_emb = self.txt_enc(captions, lengths)
        
        # MoCo
        if is_train and self.opt.use_moco:
            #start_moco = time.time()
            N = images.shape[0]   
            
            with torch.no_grad():
                
                self._momentum_update_key_encoder()
                
                v_embed_k = self.v_encoder_k(images, image_lengths)
                
                t_embed_k = self.t_encoder_k(captions, lengths)
           
            #start_moco_loss = time.time()
             
            loss_moco = self.hyhcl_loss.moco_forward(img_emb, t_embed_k, cap_emb, v_embed_k, self.v_queue, self.t_queue)
            
            #start_queue = time.time()
            #print("loss_moco", loss_moco)
            
            self._dequeue_and_enqueue(v_embed_k, t_embed_k)

            

            return img_emb, cap_emb, loss_moco
        
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """

        loss = self.hyhcl_loss(img_emb, cap_emb)*self.opt.loss_lamda

        self.logger.update('Le', loss.data.item(), img_emb.size(0))

        return loss

    # One training step given images and captions
    def train_emb(self, images, captions, lengths, image_lengths=None, warmup_alpha=None):

        self.Eiters += 1
        self.logger.update('Iter', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # amp
        with autocast():
            
            # compute the embeddings
            if self.opt.use_moco:
                
                img_emb, cap_emb, loss_moco = self.forward_emb(images, captions, lengths, image_lengths=image_lengths,
                                                    is_train=True)

                
                self.logger.update('Le_moco', loss_moco.data.item(), img_emb.size(0))
                
                loss_encoder = self.forward_loss(img_emb, cap_emb)
            
                loss = loss_encoder + loss_moco
                
                self.logger.update('Loss', loss.data.item(), img_emb.size(0))
            else:
                img_emb, cap_emb = self.forward_emb(images, captions, lengths, image_lengths=image_lengths, is_train=True)
                loss = self.forward_loss(img_emb, cap_emb)
                print("loss", loss)

        
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        
        # if warmup_alpha is not None:
        #     loss = loss * warmup_alpha

       
        self.scaler.scale(loss).backward()
       
        #start_update = time.time()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.params, self.grad_clip)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
            
        # 记录loss
        self.logger.update('Loss', loss.item(), self.opt.batch_size)

        
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
       
        for param_q, param_k in zip(self.img_enc.parameters(), self.v_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.txt_enc.parameters(), self.t_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_keys, t_keys):
        batch_size = v_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.v_queue[:, ptr : ptr + batch_size] = v_keys.T
        self.t_queue[:, ptr : ptr + batch_size] = t_keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


if __name__ == '__main__':

    pass
    



