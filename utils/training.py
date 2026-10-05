import random
import torch
import numpy as np
import copy
import torchmetrics
from utils.models import save_model_ckpt
from utils.testing import test, ood_test
import psutil
import gc
import os

EPS = 1e-9

def set_training_seed(args):

    # set the seed for training
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)

    # this is critical for reproducibility for ResNet50 models
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_zeroshot(args, test_dataloader, model, logger, classifier, is_encoder):

    zs_test_acc = test(args, dataloader=test_dataloader, model=model, classifier=classifier,
                       test_label_map=[i for i in range(1000)], device=args.device,
                       is_encoder=is_encoder)
    logger.info(f"+++++ Zero-shot Test Acc: {round(zs_test_acc, 3)}")

    return zs_test_acc


def train_ce(args, logger, loss_logger, model, classifier, train_dataloader, val_dataloader, test_preprocess=None, is_encoder=True):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""

    logger.info(f"Start standard finetuning ......")

    model.train()
    classifier.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    val_acc_list_ID = []
    val_acc_list_RT = []
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    acc_v2, acc_s, acc_a, acc_r, ood_avg = -1, -1, -1, -1, -1

    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0

        acc = torchmetrics.Accuracy(num_classes=1000, average="weighted", task="multiclass", top_k=1)
        acc.to(args.device)
        
        for inputs, labels in train_dataloader:
            num_iter += 1

            if num_iter % 1000 == 0:
                logger.info(f"Processing epoch {epoch}, iter {num_iter} ...")

            images = inputs.to(args.device)
            labels = labels.to(args.device)

            if args.pre_extracted:
                image_features = images
            else:
                if is_encoder:
                    image_features = model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                else:
                    image_features = model(images)
            

            logits = classifier(image_features)
            logits = logits * logit_scale.exp()

            total_loss = loss(logits, labels)
            train_acc = acc(logits, labels)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        logger.info(f'done epoch: {epoch}, testing...')

        # watch the GPU, CPU memory usage
        logger.info(f'GPU Memory Usage after epoch {epoch}: {torch.cuda.memory_allocated(args.device) / (1024.0 ** 3):.2f} GB') # in GB
        logger.info(f'CPU Memory Usage after epoch {epoch}: {psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3):.2f} GB') # in GB

        torch.cuda.empty_cache()  # Clears unused GPU memory
        gc.collect()              # Clears unused CPU memory

        logger.info(f'Garbage collected: GPU Memory Usage after epoch {epoch}: {torch.cuda.memory_allocated(args.device) / (1024.0 ** 3):.2f} GB') # in GB
        logger.info(f'Garbage collected: CPU Memory Usage after epoch {epoch}: {psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3):.2f} GB') # in GB


        # validate
        if args.early_stop or epoch == args.epochs:
            if isinstance(val_dataloader, tuple):
                val_acc_ID = test(args, dataloader=val_dataloader[0], model=model, classifier=classifier, 
                                    test_label_map=[i for i in range(1000)], device=args.device, is_encoder=is_encoder)
                val_acc_RT = test(args, dataloader=val_dataloader[1], model=model, classifier=classifier, 
                                    test_label_map=[i for i in range(1000)], device=args.device, is_encoder=is_encoder)
                
                # store both ID and RT val acc
                val_acc_list_ID.append(val_acc_ID/100)
                val_acc_list_RT.append(val_acc_RT/100)
                val_acc = (val_acc_ID, val_acc_RT)

                best_records['best_val_acc'] = -1
                best_records['best_logit_scale'] = torch.zeros_like(logit_scale).to(args.device) # set the logit_scale to 0
                best_records['best_epoch'] = -1
                best_records['best_iter'] = -1

            else:
                val_acc = test(args, dataloader=val_dataloader, model=model, classifier=classifier, 
                                test_label_map=[i for i in range(1000)], device=args.device, is_encoder=is_encoder)

                # check if val acc has improved, here i use the val_acc rather than val_loss
                if  val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    # best_logit_scale = copy.deepcopy(logit_scale)
                    best_logit_scale = torch.zeros_like(logit_scale).to(args.device) # set the logit_scale to 0
                    best_epoch = epoch
                    best_iter = num_iter
                    best_head = copy.deepcopy(classifier)
                    best_model = copy.deepcopy(model)

                    # save into the best_records
                    best_records['best_val_acc'] = best_val_acc
                    best_records['best_logit_scale'] = best_logit_scale
                    best_records['best_epoch'] = best_epoch
                    best_records['best_iter'] = best_iter
    
        
        train_acc = acc.compute().item() * 100
        train_loss_avg = train_loss_sum / len(train_dataloader)
        if isinstance(val_acc, tuple):
            loss_logger.write(f'{epoch},{num_iter},{round(train_acc, 6)},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc[0], 6)}/{round(val_acc[1], 6)},'
                            f'{round(test_acc, 6)}, {round(acc_v2, 6)}, {round(acc_s, 6)}, {round(acc_a, 6)}, {round(acc_r, 6)}, {round(ood_avg, 6)}\n')
            loss_logger.flush()
            logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Acc: {round(train_acc, 6)}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, "
                        f"logit_scale: {round(logit_scale.item(), 6)}, Val Acc: {round(val_acc[0], 6)}/{round(val_acc[1], 6)}, Test Acc: {round(test_acc, 3)}, OOD Avg: {round(ood_avg, 3)}")
        else:
            loss_logger.write(f'{epoch},{num_iter},{round(train_acc, 6)},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},'
                            f'{round(test_acc, 6)}, {round(acc_v2, 6)}, {round(acc_s, 6)}, {round(acc_a, 6)}, {round(acc_r, 6)}, {round(ood_avg, 6)}\n')
            loss_logger.flush()
            logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Acc: {round(train_acc, 6)}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, "
                        f"logit_scale: {round(logit_scale.item(), 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}, OOD Avg: {round(ood_avg, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records, model, classifier, optimizer, 
                                         scheduler, logit_scale, val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')
        
    if isinstance(val_dataloader, tuple):
        # calculate f1-score over the ID and RT accuracy gains
        val_acc_list_ID = np.array(val_acc_list_ID)
        val_acc_list_RT = np.array(val_acc_list_RT)
        delta_ID = val_acc_list_ID - val_acc_list_ID.min()
        delta_RT = val_acc_list_RT - val_acc_list_RT.min()
        gF1_list = 2 * delta_ID * delta_RT / (delta_ID + delta_RT + EPS)
        logger.info(f'gF1 list: {gF1_list}')

        # select the best model according to the gF1 score
        best_val_acc = max(gF1_list)
        best_logit_scale = torch.zeros_like(logit_scale).to(args.device) # set the logit_scale to 0
        best_epoch = np.argmax(gF1_list) + 1
        best_iter = best_epoch * len(train_dataloader)
        ckpt_path = f'{args.ckpt_path}/model_bs{args.bsz}_lr-cls{args.lr_classifier}_lr-bkb{args.lr_backbone}_wd{args.wd}_epoch_{best_epoch}_iter_{best_iter}.pth'
        ckpt = torch.load(ckpt_path)
        classifier.load_state_dict(ckpt['head'])
        model.load_state_dict(ckpt['clip'])
        best_head = copy.deepcopy(classifier)
        best_model = copy.deepcopy(model)

        # save into the best_records
        best_records['best_val_acc'] = best_val_acc
        best_records['best_logit_scale'] = best_logit_scale
        best_records['best_epoch'] = best_epoch
        best_records['best_iter'] = best_iter

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 4)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale
