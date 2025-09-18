import argparse
import os


def get_argument_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='D:/datasets', type=str,
                        help='path to datasets')
    parser.add_argument('--dataset', default='f30k', type=str,
                        help='dataset coco or f30k')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=1., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/f30k_swin_6_16_n',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='runs/f30k_swin_6_16_n',
                        help='Path to save the model.')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=1,
                        help='The number of warmup epochs using mean vse loss') 
    
    parser.add_argument('--multi_gpu', type=int, default=0, 
                        help='whether use the multi gpus for training')
    parser.add_argument('--size_augment', type=int, default=1, 
                        help='whether use the size augmention for training')
    
    parser.add_argument('--mask_repeat', type=int, default=1, 
                        help='whether mask the repeat images in the batch for vse loss')  
    parser.add_argument('--save_results', type=int, default=1,
                        help='whether save the similarity matrix for the evaluation')
    parser.add_argument('--gpu-id', type=int, default=0, 
                        help='the gpu-id for training')
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased',
                        help='the path of pretrained checkpoint')      
    parser.add_argument("--lr_schedules", default=[9,15,20,25], type=int, nargs="+", 
                        help='epoch schedules for lr decay') 
    parser.add_argument("--decay_rate", default=0.3, type=float, 
                        help='lr decay_rate for optimizer') 
    
    parser.add_argument('--f30k_img_path', type=str, default='D:/datasets/f30k/flickr30k-images', help='the path of f30k images')
    parser.add_argument('--coco_img_path', type=str, default='D:/datasets/coco', help='the path of coco images')

    # vision transformer
    parser.add_argument('--img_res', type=int, default=224, help='the image resolution for ViT input') 
    parser.add_argument('--vit_type', type=str, default='swin', help='the type of vit model')      ####

    parser.add_argument('--use_moco', default=1, type=int)
    parser.add_argument('--moco_M', default=2048, type=int)
    parser.add_argument('--moco_r', default=0.999, type=float)
    parser.add_argument('--precomp_enc_type', default="basic", help='basic|backbone')
    parser.add_argument('--optim', default="adam", help='adam|sgd')
    parser.add_argument('--loss_lamda', default=1, type=float)
    parser.add_argument('--mu', default=90, type=float)
    parser.add_argument('--gama', default=0.5, type=float)

    parser.add_argument('--use_infonce', type=int, default=1,
                    help='whether to use InfoNCE loss')
    parser.add_argument('--temperature', type=float, default=0.03,
                    help='temperature parameter for InfoNCE loss')
    parser.add_argument('--hubness_weight', type=float, default=1.0,
                        help='Weight for hubness-aware loss')
    parser.add_argument('--infonce_weight', type=float, default=1.0,
                        help='Weight for InfoNCE loss')
   
    
    return parser

def extra_parameters(parser):
    # loss function
    parser.add_argument('--base_loss', default='trip', type=str, help='the loss function for the initial embeddings.')
    parser.add_argument('--gnn_loss', default='trip', type=str, help='the loss function for the enhanced embeddings.')

    # warmup for training
    parser.add_argument('--warmup', default=8000, type=int,
                        help='warmup iteration for instance-level interaction network')

    # Graph modelling on fragment-level
    parser.add_argument('--residual_weight', default=0.8, type=float,
                        help='the weight of residual operation for pooling')

    # Graph modelling on instance-level
    parser.add_argument('--num_layers_enc', default=1, type=int, help='the num_layers of Transformer encoder')
    parser.add_argument('--nhead', default=16, type=int, help='the num_head for Transformer encoder')
    parser.add_argument('--dropout', default=0.1, type=float, help='the dropout rate for Transformer encoder')
    parser.add_argument('--graph_lr_factor', default=1.0, type=float,
                        help='the learning rate factor for the interaction model')

    # connection and relevance relation
    parser.add_argument('--mask_weight', default=1.0, type=float, help='use extra weight for the attention matrix')
    parser.add_argument('--threshold', default=0.5, type=float, help='give a threshold for the mask proportion, 0-1')
    parser.add_argument('--topk', default=10, type=int, help='the topk for the region-word pair selection')
    parser.add_argument('--reg_loss_weight', default=10, type=float, help='the values for the regularization loss')
    parser.add_argument('--norm_input', default=1, type=int, help='if use L2-norm embeddings as input')

    # loss function
    parser.add_argument('--cross_loss', default=1, type=int, help='if compute the loss for cross embeddings')

    return parser

def save_parameters(opt, save_path):

    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n'
    
    with open(os.path.join(save_path, 'Parameters.txt'), 'w') as f:
        f.write(base_str)


if __name__ == '__main__':

    pass
    