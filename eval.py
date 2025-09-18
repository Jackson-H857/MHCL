import os
import torch
import argparse
import logging
from lib import evaluation
import arguments
from arguments import extra_parameters



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default='D:/datasets', type=str, help='path to datasets')
    parser.add_argument('--dataset', default='f30k', type=str, help='the dataset choice, coco or f30k')
    parser.add_argument('--save_results', type=int, default=0, help='if save the similarity matrix for ensemble')
    parser.add_argument('--gpu-id', type=int, default=0, help='the gpu-id for evaluation')
    parser.add_argument('--vit_type', type=str, default='swin', help='the type of vit model')

    parser.add_argument('--visualize', type=int, default=1, help='if visualize the model embeddings')
    # 使用arguments.py中的参数配置
    parser = arguments.get_argument_parser()
    parser = extra_parameters(parser)

    
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    if opt.dataset == 'coco':
        weights_bases = [
            'runs/coco_test'
        ]
    else:
        weights_bases = [
            'visualize'
        ]

    for base in weights_bases:

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        
        # Save the final similarity matrix 
        if opt.save_results:  
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            # Evaluate COCO 5K
            evaluation.evalrank(model_path, opt=opt, split='testall', fold5=True, save_path=save_path, data_path=opt.data_path)
        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, opt=opt, split='test', fold5=False, save_path=save_path, data_path=opt.data_path)
        
        


if __name__ == '__main__':
    
    main()
    

