import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, SwinModel, ViTModel
import logging
import torch.nn.functional as F
import time

from lib.mlp import FC_MLP

logger = logging.getLogger(__name__)


# 'True' represents to be masked （Do not participate in the calculation of attention）
# 'False' represents not to be masked
def padding_mask(embs, lengths):

    mask = torch.ones(len(lengths), embs.shape[1], device=lengths.device)
    for i in range(mask.shape[0]):
        end = int(lengths[i])
        mask[i, :end] = 0.

    return mask.bool()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    _x, index = x.topk(k, dim=dim)
    return _x

    
# uncertain length
def maxk_pool1d_var(x, dim, k, lengths):
    # k >= 1
    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):
        # keep use all number of features
        k = min(k, int(lengths[idx].item()))

        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim-1)[0]

        max_k_i = maxk_pool1d(tmp, dim-1, k)
        results.append(max_k_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


def avg_pool1d_var(x, dim, lengths):

    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):

        # keep use all number of features
        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim-1)[0]
        avg_i = tmp.mean(dim-1)

        results.append(avg_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


class Maxk_Pooling_Variable(nn.Module):
    def __init__(self, dim=1, k=2):
        super(Maxk_Pooling_Variable, self).__init__()

        self.dim = dim
        self.k = k

    def forward(self, features, lengths):

        pool_weights = None
        pooled_features = maxk_pool1d_var(features, dim=self.dim, k=self.k, lengths=lengths)
        
        return pooled_features, pool_weights


class Avg_Pooling_Variable(nn.Module):
    def __init__(self, dim=1):
        super(Avg_Pooling_Variable, self).__init__()
        
        self.dim = dim

    def forward(self, features, lengths):

        pool_weights = None
        pooled_features = avg_pool1d_var(features, dim=self.dim, lengths=lengths)
        
        return pooled_features, pool_weights


def get_text_encoder(opt, embed_size, no_txtnorm=False): 
    
    text_encoder = EncoderText_BERT(opt, embed_size, no_txtnorm=no_txtnorm)
    
    return text_encoder


#def get_image_encoder(opt , img_dim, embed_size,no_imgnorm=False):
def get_image_encoder(opt,embed_size):
    
    #img_enc = EncoderImageAggr(opt, img_dim, embed_size, no_imgnorm)
    img_enc = VisionTransEncoder(opt,embed_size)
    
    return img_enc


# ViT encoder
class VisionTransEncoder(nn.Module):
    def __init__(self, opt, embed_size=1024):
        super().__init__()

        self.opt = opt
        self.embed_size = embed_size
        self.no_imgnorm = opt.no_imgnorm

        self.current_stage = 0
    
        # Swin model
        if 'swin' in opt.vit_type:                           
            # img_res 224 * 224, 7*7 patch
            self.visual_encoder = SwinModel.from_pretrained("./swin-base-patch4-window7-224")
            
            #self.patch_size = 4  # Swin-tiny的patch size
            opt.num_patches = 49
            print('swin model')
            self.hidden_size = self.visual_encoder.config.hidden_size  
            print('Using Swin-Transformer as backbone')
            
            #self.hidden_sizes = [192, 384, 768, 768]  
        else:              
            self.visual_encoder = ViTModel.from_pretrained("vit-base-patch16-224-in21k")
            opt.num_patches = 196
            self.hidden_size = self.visual_encoder.config.hidden_size 
            print('Using ViT as backbone')
        

        # B * N * 2048 -> B * N * 1024    
        # 49
        self.fc = FC_MLP(self.hidden_size, embed_size // 2, embed_size, 2, bn=True)           
        self.fc.apply(init_weights)

        #GSF
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function 
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()

        # 冻结backbone的部分层
        #self._freeze_stages()
        #self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.visual_encoder.parameters():  
            param.requires_grad = True    

    def _freeze_stages(self):
        if 'swin' in self.opt.vit_type:
            # 冻结patch embedding
            self.visual_encoder.embeddings.patch_embeddings.requires_grad_(False)
            self.visual_encoder.embeddings.norm.requires_grad_(False)
            
            # 冻结前两个stage
            layers = self.visual_encoder.encoder.layers
            for i in range(min(2, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False
        else:
            # 冻结ViT的patch embedding
            self.visual_encoder.embeddings.requires_grad_(False)
            # 冻结前6个transformer blocks
            for i in range(6):
                layer = self.visual_encoder.encoder.layer[i]
                layer.requires_grad_(False)
    
    def unfreeze_stage(self):
        if self.current_stage == 0:
            # 只解冻最后一层
            for param in self.visual_encoder.encoder.layers[-1].parameters():
                param.requires_grad = True
        elif self.current_stage == 1:
            # 解冻最后两层
            for i in range(2):
                for param in self.visual_encoder.encoder.layers[-1-i].parameters():
                    param.requires_grad = True
        elif self.current_stage == 2:
            # 解冻最后三层，而不是全部解冻
            for i in range(3):
                for param in self.visual_encoder.encoder.layers[-1-i].parameters():
                    param.requires_grad = True
                    
        self.current_stage += 1


    def forward(self, images, image_lengths, graph=False):
        #start = time.time()
        patch_features = self.visual_encoder(images).last_hidden_state  # [B, num_patches, hidden_size]
        #print("Encoder time:", time.time() - start)
        
        #start_fc = time.time()
        patch_features = self.fc(patch_features)  # [B, num_patches, embed_size]
        #print("FC time:", time.time() - start_fc)
        
        #gpool_time =  time.time()
        img_emb_res, _ = self.gpool(patch_features, image_lengths)
   
        #pianduan_time = time.time()
        src_key_padding_mask = padding_mask(patch_features, image_lengths)
        patch_features = patch_features.transpose(1, 0)  # [num_patches, B, embed_size]
        patch_features = self.aggr(patch_features, src_key_padding_mask=src_key_padding_mask)
        patch_features = patch_features.transpose(1, 0)  # [B, num_patches, embed_size]
        
        img_emb, _ = self.graph_pool(patch_features, image_lengths)
        
        #cancha_time = time.time()
        img_emb = self.opt.residual_weight * img_emb_res + (1-self.opt.residual_weight) * img_emb
        #patch_features = l2norm(patch_features, dim=-1)
        #img_emb_notnorm = img_emb.clone()
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)
        
        
        return img_emb



     
class EncoderImageAggr(nn.Module):
    def __init__(self, opt, img_dim=2048, embed_size=1024, no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        
        # B * N * 2048 -> B * N * 1024
        # N = 36 for region features
        self.fc = FC_MLP(img_dim, embed_size // 2, embed_size, 2, bn=True)           
        self.fc.apply(init_weights)

        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()

    def forward(self, images, image_lengths, graph=False):

        img_emb = self.fc(images)

        # initial visual embedding
        img_emb_res, _ = self.gpool(img_emb, image_lengths)

        img_emb_pre_pool = img_emb

        # get padding mask
        src_key_padding_mask = padding_mask(img_emb, image_lengths)

        # switch the dim
        img_emb = img_emb.transpose(1, 0)
        img_emb = self.aggr(img_emb, src_key_padding_mask=src_key_padding_mask)
        img_emb = img_emb.transpose(1, 0)

        # enhanced visual embedding
        img_emb, _  = self.graph_pool(img_emb, image_lengths)

        # the final global embedding
        img_emb =  self.opt.residual_weight * img_emb_res + (1-self.opt.residual_weight) * img_emb

        img_emb_notnorm = img_emb
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        if graph:
            return images, image_lengths, img_emb, img_emb_notnorm, img_emb_pre_pool
        else:
            return img_emb


# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt, embed_size=1024, no_txtnorm=False):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # backbone features -> embbedings
        self.linear = nn.Linear(768, embed_size)
        
        # relation modeling for local feature
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()


    def forward(self, x, lengths, graph=False):

        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()

        # all hidden features, D=768 in bert-base model
        # attention_mask： Mask to avoid performing attention on padding token indices.
        # bert_output[0] is the last/final hidden states of all tokens
        # bert_output[1] is the hidden state of [CLS] + one fc layer + Tanh, can be used for classification tasks.
        #BertModel_time = time.time()
        # N = max_cap_lengths, D = 768
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D
        #print("BertModel_time,用时:", time.time() - BertModel_time)
        cap_len = lengths
        #text_shengxia_time =time.time()

        # B x N x embed_size
        cap_emb = self.linear(bert_emb)

        # initial textual embedding
        cap_emb_res, _ = self.gpool(cap_emb, cap_len)

        # fragment-level relation modeling for word features
        
        # get padding mask
        src_key_padding_mask = padding_mask(cap_emb, cap_len)
        
        # switch the dim
        cap_emb = cap_emb.transpose(1, 0)
        cap_emb = self.aggr(cap_emb, src_key_padding_mask=src_key_padding_mask)
        cap_emb = cap_emb.transpose(1, 0)

        # enhanced textual embedding
        cap_emb, _ = self.graph_pool(cap_emb, cap_len)

        cap_emb = self.opt.residual_weight * cap_emb_res + (1-self.opt.residual_weight) * cap_emb 
        #print("text_shengxia_time,用时:", time.time() - text_shengxia_time)
        # the final global embedding
        cap_emb_notnorm = cap_emb
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)


        return cap_emb



if __name__ == '__main__':

    pass
