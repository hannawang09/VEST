import torch
from utils.models import MyLinear, load_model, save_best_model, get_zeroshot_weights, get_model
import time
from utils.parser import parse_args
from utils.logger import set_logger
from utils.testing import test, ood_test
import copy
import torch.nn as nn
from utils.training import set_training_seed, run_zeroshot, train_ce
from utils.optimizer import set_optimizer, set_params
from utils.scheduler import build_lr_scheduler
from utils.features import pre_extract_feature, get_dataloader_preextracted
import datasets


def run_stage1_finetuning(args, logger, model, classifier, train_preprocess, test_preprocess):

    # dataloaders
    imagenet_train, _ = datasets.build_imagenet_few_shot_dataset('imagenet', args, args.data_seed, train_preprocess, 
                                                                 root=args.root, num_shots=args.num_shots, 
                                                                 w_retrival=True if args.data_source == 'fewshot+retrieved' else False)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_train,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=8,
        pin_memory=False)
    if args.val_type == 'fewshot+retrieved':
        val_dataloader_ID = torch.utils.data.DataLoader(
            dataset=valset_ID,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=False)
        val_dataloader_RT = torch.utils.data.DataLoader(
            dataset=valset_RT,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=False)
        val_dataloader = (val_dataloader_ID, val_dataloader_RT)
    else:
        val_dataloader = torch.utils.data.DataLoader(
            dataset=valset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_test,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False)

    if args.pre_extracted:
        logger.info(f'Use pre-extracted features.')
        dataloader_dict = {'train_dataloader': train_dataloader, 'test_dataloader': test_dataloader, 'val_dataloader': val_dataloader}
        train_fea_path, val_fea_path, test_fea_path = pre_extract_feature(args, logger, model, dataloader_dict, is_encoder)
        train_dataloader, val_dataloader, test_dataloader = get_dataloader_preextracted(args, logger, train_fea_path, val_fea_path, test_fea_path, args.device)

    loss = nn.CrossEntropyLoss()
    params, logit_scale = set_params(args, model, classifier, logger) # depend on method
    optimizer, total_iter = set_optimizer(args, params, train_dataloader)
    scheduler = build_lr_scheduler(optimizer,
                                   lr_scheduler="cosine",
                                   warmup_iter=args.warmup_iter,
                                   max_iter=total_iter,
                                   warmup_type=args.warmup_type,
                                   warmup_lr=args.warmup_lr,
                                   verbose=False)

    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler
    stage1_method = args.method

    if args.model_path:
        load_model(args, logger, model, classifier, test_dataloader, is_encoder)

    # check zeroshot acc
    if args.check_zeroshot or args.method == 'zeroshot':
        logger.info(f"Check Zero-shot Acc ......")
        zs_test_acc = run_zeroshot(args, test_dataloader, model, logger, classifier, is_encoder)
        acc_list = ood_test(args, model, classifier, test_preprocess, logger, is_encoder)

    if args.method == 'zeroshot':
        result_summary = f'{args.dataset},{stage1_method},{args.data_source},{args.cls_init},{args.num_shots},{args.data_seed},{round(zs_test_acc,3)}'
        logger.info(f'{result_summary}')
        exit()

    if args.skip_stage1:
        logger.info(f"Skip stage 1 finetuning.")
        return -1, None

    #---------- Training
    if args.method == 'finetune' or args.method == 'lp':
        best_model, best_head, \
            best_records, best_logit_scale = train_ce(args, logger, loss_logger, model, classifier, 
                                                      train_dataloader, val_dataloader, test_preprocess,
                                                      is_encoder)
        
    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")


    # print the logit_scale
    logger.info(f"logit_scale: {round(logit_scale.item(), 8)}")
    logger.info(f"best_logit_scale: {round(best_logit_scale.item(), 8)}")

    # test the best model after finetuning
    test_acc = test(args, dataloader=test_dataloader, model=best_model, classifier=best_head, 
                    test_label_map=[i for i in range(1000)], device=args.device, is_encoder=is_encoder)
    logger.info(f"+++++ Stage 1 Finetuning Test Acc: {round(test_acc, 3)}")

    #----------- save stage 1 best model
    best_model_path = save_best_model(args, best_records, best_model, best_head, best_logit_scale, test_acc, stage=1)
    logger.info(f'Stage 1 Best Model saved to: {best_model_path}')

    #----------- Test ImageNet OOD performance
    logger.info(f"+++++ Test Stage 1 ImageNet OOD ......")
    acc_list = ood_test(args, best_model, best_head, test_preprocess, logger, is_encoder)

    return test_acc, best_model_path



def run_stage2_FSFT(model, classifier, stage1_best_model_path, train_preprocess):

    # reset the flag
    args.epochs = 10
    args.data_source = 'fewshot'
    args.model_path = stage1_best_model_path

    logger.info(f"Run stage 2 few-shot finetuning ......")

    # set the dataloaders
    imagenet_train, _ = datasets.build_imagenet_few_shot_dataset('imagenet', args, args.data_seed, train_preprocess, 
                                                                 root=args.root, num_shots=args.num_shots, 
                                                                 w_retrival=True if args.data_source == 'fewshot+retrieved' else False)


    train_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_train,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=8,
        pin_memory=False)
    if args.val_type == 'fewshot+retrieved':
        val_dataloader_ID = torch.utils.data.DataLoader(
            dataset=valset_ID,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=False)
        val_dataloader_RT = torch.utils.data.DataLoader(
            dataset=valset_RT,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=False)
        val_dataloader = (val_dataloader_ID, val_dataloader_RT)
    else:
        val_dataloader = torch.utils.data.DataLoader(
            dataset=valset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_test,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False)

    load_model(args, logger, model, classifier, test_dataloader, is_encoder=is_encoder)

    # Imporatnt! Need to reset the params, optimizer, scheduler, loss, logit_scale
    loss = nn.CrossEntropyLoss()
    params, logit_scale = set_params(args, model, classifier, logger)
    optimizer, total_iter = set_optimizer(args, params, train_dataloader)
    scheduler = build_lr_scheduler(optimizer,
                                   lr_scheduler="cosine",
                                   warmup_iter=args.warmup_iter,
                                   max_iter=total_iter,
                                   warmup_type=args.warmup_type,
                                   warmup_lr=args.warmup_lr,
                                   verbose=False)

    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler

    #---------- Training
    best_model, best_head, \
        best_records, best_logit_scale = train_ce(args, logger, loss_logger, model, classifier, 
                                                    train_dataloader, val_dataloader, test_preprocess,
                                                    is_encoder)

    # test the best model after FSFT
    test_acc = test(args, dataloader=test_dataloader, model=best_model, classifier=best_head, 
                    test_label_map=[i for i in range(1000)], device=args.device, is_encoder=is_encoder)
    logger.info(f"+++++ Stage 2 FSFT Test Acc: {round(test_acc, 3)}")

    #----------- save stage 2 best model
    best_model_path = save_best_model(args, best_records, best_model, best_head, 
                                      logit_scale, test_acc, stage=2)
    logger.info(f'Stage 2 FSFT Best Model saved to: {best_model_path}')


    #----------- Test ImageNet OOD performance
    logger.info(f"+++++ Test Stage 2 ImageNet OOD ......")
    acc_list = ood_test(args, best_model, best_head, test_preprocess, logger, is_encoder)


    return test_acc, best_model_path



if __name__ == '__main__':
    program_start = time.time()
    args = parse_args()
    logger, loss_logger = set_logger(args)
    args.logger = logger
    args.loss_logger = loss_logger
    set_training_seed(args)

    # load model
    model, train_preprocess, test_preprocess, tokenizer = get_model(args, logger)
    is_encoder = True if 'clip' in args.model_cfg else False # determine the model is clip or dinov2

    # Prepare dataset
    # ID testset
    imagenet_test, text_name = datasets.build_imagenet_dataset('imagenet', 'test', test_preprocess, root=args.root)
    # validation set
    if args.val_type == 'testset':
        valset = imagenet_test
    elif args.val_type == 'fewshot':
        valset, _ = datasets.build_imagenet_few_shot_dataset('imagenet', args, args.data_seed, test_preprocess, 
                                                             root=args.root, num_shots=args.num_shots, w_retrival=False)
    elif args.val_type == 'retrieved':
        valset = datasets.build_validation_set(args.val_type, args, args.data_seed, test_preprocess)
    elif args.val_type == 'fewshot+retrieved':
        valset_ID, _ = datasets.build_imagenet_few_shot_dataset('imagenet', args, args.data_seed, test_preprocess, 
                                                                root=args.root, num_shots=args.num_shots, w_retrival=False)
        valset_RT = datasets.build_validation_set('retrieved', args, args.data_seed, test_preprocess)
    
    # set classifier head
    num_classes = 1000
    num_features = 512 if 'clip' in args.model_cfg else 768 # 512 for clip model, 768 for dinov2 model.
    logit_scale = model.logit_scale if 'clip' in args.model_cfg else 0 # use the pretrained logit_scale (4.60517) for clip model; do not use logit scale for dinov2

    if args.cls_init == 'openai':
        with torch.no_grad():
            template = 'openai_imagenet_template'
            logger.info(f"Getting zeroshot weights from {args.cls_init}.")
            zeroshot_weights = get_zeroshot_weights(text_name, template, model, tokenizer, logit_scale)
        logger.info(f"Initialize classifier head with text embedding. weights.shape: {zeroshot_weights.shape}")
        classifier = MyLinear(input_dim=num_features, num_classes=num_classes, bias=False)
        classifier._init_weights(zeroshot_weights)

    elif args.cls_init == 'random':
        logger.info(f'Initialized classifier head with random weights.')
        classifier = MyLinear(input_dim=num_features, num_classes=num_classes, bias=False)
    
    elif args.cls_init == 'lp': # load the classifier weights using the classifier after probing
        assert args.cls_path is not None
        classifier = MyLinear(input_dim=num_features, num_classes=num_classes, bias=False)
        checkpoint = torch.load(args.cls_path)
        classifier.load_state_dict(checkpoint['head'])
        logger.info(f'Loaded classifier weights from {args.cls_path}')

    model.to(args.device)
    classifier.to(args.device)

    #---------- run finetuning for stage 1
    stage1_acc, stage1_best_model_path = run_stage1_finetuning(args, logger, model, classifier, train_preprocess, test_preprocess)
    stage1_method = args.method # record method here, as in stage 2 method will be updated to probing

    # replace the stage1_best_model_path to run stage 2 for certrain checkpoints
    if args.skip_stage1:
        stage1_best_model_path = args.stage1_model_path


    #---------- run FSFT for stage 2
    if not args.skip_stage2:
        stage2_fsft_acc, stage2_best_model_path = run_stage2_FSFT(model, classifier, stage1_best_model_path, train_preprocess)

    else:
        logger.info(f"Skip stage 2 FSFT.")
        stage2_fsft_acc = -1
        stage2_best_model_path = 'None'

    loss_logger.close()
    program_end = time.time()
    logger.info(f"Total time: {round((program_end-program_start)/60, 1)} mins.")


    result_summary = f'{args.dataset},{stage1_method},{args.data_source},{args.cls_init},'\
                     f'{args.num_shots},{args.data_seed},{round(stage1_acc,3)},'\
                     f'{round(stage2_fsft_acc,3)}'
    logger.info(f'{result_summary}')
    print(f'{result_summary}')