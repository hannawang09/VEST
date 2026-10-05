import argparse
import yaml
import os

def parse_args():

    parser = argparse.ArgumentParser(description='Arguments for script.')

    # logging
    parser.add_argument('--log_mode', type=str, default='both', choices=['console', 'file', 'both'], help='where to log.')
    parser.add_argument('--folder', type=str, default='output', help='Folder for saving output.')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix for Log file Name.')

    # model
    parser.add_argument('--model_cfg', default='vitb16_clip_openai', type=str, help='Model for finetuning.')
    parser.add_argument('--model_path', default=None, type=str, help='Model path to start training from.')
   
    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'], help='Dataset name.')


    # training data
    parser.add_argument('--data_source', type=str, default='fewshot',
                        choices=['fewshot', 'retrieved', 'fewshot+retrieved'],
                        help='training data source.')
    parser.add_argument('--num_shots', type=int, default=16, help='number of shots for fewshot data')
    parser.add_argument('--data_seed', type=int, default=1, help='Random seeds for different splits.')
    parser.add_argument('--pre_extracted', default=False, action='store_true', help='use pre-extracted features.')
    parser.add_argument('--recal_fea', default=False, action='store_true', help='re-run feature extraction.')


    # training
    parser.add_argument('--method', type=str, default='finetune', choices=['zeroshot', 'finetune', 'lp'],
                        help='Method for training.')
    parser.add_argument('--training_seed', type=int, default=1, help='Random seeds for training.')
    parser.add_argument('--ft_topk_blks', type=int, default=-1, help='finetune top-k blocks, -1 means all blocks.')
    parser.add_argument('--cls_init', type=str, default='openai', choices=['random', 'openai', 'lp'],
                        help='Initialize the classifier head in different ways.')
    parser.add_argument('--cls_path', default=None, type=str, help='classifier path to start training from.')

    parser.add_argument('--skip_stage1', default=False, action='store_true', help='Set to skip stage 1 training')
    parser.add_argument('--skip_stage2', default=False, action='store_true', help='Set to skip stage 2 probing')
    parser.add_argument('--stage1_model_path', default=None, type=str, help='stage 1 best model path to start stage 2.')

    parser.add_argument('--check_zeroshot', action='store_true', help='check zeroshot acc.')
    parser.add_argument('--early_stop', action='store_true', help='use val set for early stopping.')
    parser.add_argument('--epochs', type=int, default=0, help='number of epochs to train the model')
    parser.add_argument('--stop_epochs', type=int, default=200, help='number of epochs to stop the training of the model')
    
    # validation
    parser.add_argument('--val_type', type=str, default='testset', choices=['testset', 'fewshot', 'retrieved', 'fewshot+retrieved'],
                        help='Validation type to use for training.')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='Num of workers.')
    parser.add_argument('--lr_classifier', type=float, default=1e-4, help='Learning rate for the classifier head.')
    parser.add_argument('--lr_backbone', type=float, default=1e-6, help='Learning rate for the visual encoder.')
    parser.add_argument('--lr_projector', type=float, default=None, help='Learning rate for the visual and text projector.')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay for model.')
    parser.add_argument('--bsz', type=int, default=32, help='Batch Size')
    parser.add_argument('--optim', type=str, default='AdamW', choices=['AdamW', 'SGD'], help='type of optimizer to use.')
    parser.add_argument('--warmup_lr', type=float, default=1e-8, help='Learning rate for the warmup iterations.')
    parser.add_argument('--warmup_iter', type=int, default=50, help='Warmup iteration')
    parser.add_argument('--warmup_type', type=str, default="linear", help='Warmup type, linear or cosine.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Logit Scale for training')

    # save
    parser.add_argument('--save_ckpt', action='store_true', help='Save model checkpoints or not.')
    parser.add_argument('--save_freq', type=int, default=10, help='Save Frequency in epoch.')

    args = parser.parse_args()

    # read the dataset and retrieved path from the config.yml file
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.root = config['root']
        args.retrieved_root = config['retrieved_root']

    if args.method == 'zeroshot':
        args.check_zeroshot = True
        args.skip_stage2 = True

    if args.pre_extracted:
        if not os.path.exists('datasets/pre_extracted/'):
            os.makedirs('datasets/pre_extracted/')
        
    # adjust folder
    args.folder = f'{args.folder}/output_{args.dataset}'


    return args