# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils
import memory
import random

import torch.nn.functional as F


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, feature_mat, key_feature_mat, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']
        prompt_idx = output['prompt_idx'][0]

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Gradient Projection
        if task_id != 0 and not args.no_pgp:
            for k, (m, params) in enumerate(model.named_parameters()):
                if m == "prompt.prompt":
                    params.grad.data[prompt_idx] = params.grad.data[prompt_idx] - torch.matmul(
                        params.grad.data[prompt_idx], feature_mat)
                if m == "prompt.prompt_key":
                    params.grad.data[prompt_idx] = params.grad.data[prompt_idx] - torch.mm(
                        params.grad.data[prompt_idx], key_feature_mat)

        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            # print("Logits:", torch.topk(logits, 5, dim=1))
            # print("Prompts:", output['prompt_idx'])
            # print(target)

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                      device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                              device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


@torch.no_grad()
def evaluate_asr_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                      device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['asr'], 
                              device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average ASR till task{}]\tASR@1: {:.4f}\tASR@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args=None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    feature, feature_mat = None, None
    key_feature, key_feature_mat = None, None

    for task_id in range(args.num_tasks):
        if not args.no_pgp:
            model.eval()
            original_model.eval()
            mem_example = memory.get_representation_matrix(data_loader[task_id]['mem'], device)
            rep, rep_key = memory.get_rep(model, original_model, mem_example, task_id)

            rep = torch.cat(rep)
            rep = rep.detach().cpu().numpy()
            pca = PCA(n_components=9)
            pca = pca.fit(rep)
            rep = pca.transform(rep)

            if task_id != 0:
                for k, (m, params) in enumerate(model.named_parameters()):
                    if m == "prompt.prompt":
                        p_ = params.data
                        p_ = p_.view(-1, 768).detach().cpu().numpy().transpose(1, 0)

                pca = PCA(n_components=9)
                pca = pca.fit(p_)
                p = pca.transform(p_)

                rep = rep + p

            rep_key = torch.cat(rep_key)
            rep_key = rep_key.detach().cpu().numpy()
            pca = PCA(n_components=5)
            pca = pca.fit(rep_key)
            rep_key = pca.transform(rep_key)

            if task_id != 0:
                print("prompt feature shape", feature.shape)
                Uf = torch.Tensor(np.dot(feature, feature.transpose())).to(device)
                print('Prompt Projection Matrix Shape: {}'.format(Uf.shape))
                feature_mat = Uf

                print("key feature shape", key_feature.shape)
                Uf = torch.Tensor(np.dot(key_feature, key_feature.transpose())).to(device)
                print('Key Projection Matrix Shape: {}'.format(Uf.shape))
                key_feature_mat = Uf

            feature = memory.update_memory(rep, 0.50, feature)
            key_feature = memory.update_memory(rep_key, 0.97, key_feature)

        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, device=device,
                                        epoch=epoch, feature_mat=feature_mat, key_feature_mat=key_feature_mat, max_norm=args.clip_grad,
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args)

            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

def simulate_clean(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args=None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    feature, feature_mat = None, None
    key_feature, key_feature_mat = None, None

    for task_id in range(1, args.num_tasks):
        
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool and not args.surrogate2_path:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key and not args.surrogate2_path:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer and not args.surrogate2_path:
            print("Reinit optimizer")
            optimizer = create_optimizer(args, model)

        args.epochs = args.simulate_round_prompt
        for epoch in range(args.simulate_round_prompt):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, device=device,
                                        epoch=epoch, feature_mat=feature_mat, key_feature_mat=key_feature_mat, max_norm=args.clip_grad,
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args)

            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_tuned_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')



def train_and_evaluate_victim(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args=None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    feature, feature_mat = None, None
    key_feature, key_feature_mat = None, None

    # create matrix to save end-of-task asr 
    asr_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        if not args.no_pgp:
            model.eval()
            original_model.eval()
            mem_example = memory.get_representation_matrix(data_loader[task_id]['mem'], device)
            rep, rep_key = memory.get_rep(model, original_model, mem_example, task_id)

            rep = torch.cat(rep)
            rep = rep.detach().cpu().numpy()
            pca = PCA(n_components=9)
            pca = pca.fit(rep)
            rep = pca.transform(rep)

            if task_id != 0:
                for k, (m, params) in enumerate(model.named_parameters()):
                    if m == "prompt.prompt":
                        p_ = params.data
                        p_ = p_.view(-1, 768).detach().cpu().numpy().transpose(1, 0)

                pca = PCA(n_components=9)
                pca = pca.fit(p_)
                p = pca.transform(p_)

                rep = rep + p

            rep_key = torch.cat(rep_key)
            rep_key = rep_key.detach().cpu().numpy()
            pca = PCA(n_components=5)
            pca = pca.fit(rep_key)
            rep_key = pca.transform(rep_key)

            if task_id != 0:
                print("prompt feature shape", feature.shape)
                Uf = torch.Tensor(np.dot(feature, feature.transpose())).to(device)
                print('Prompt Projection Matrix Shape: {}'.format(Uf.shape))
                feature_mat = Uf

                print("key feature shape", key_feature.shape)
                Uf = torch.Tensor(np.dot(key_feature, key_feature.transpose())).to(device)
                print('Key Projection Matrix Shape: {}'.format(Uf.shape))
                key_feature_mat = Uf

            feature = memory.update_memory(rep, 0.50, feature)
            key_feature = memory.update_memory(rep_key, 0.97, key_feature)

        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, device=device,
                                        epoch=epoch, feature_mat=feature_mat, key_feature_mat=key_feature_mat, max_norm=args.clip_grad,
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args)

            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        asr_stats = evaluate_asr_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                       task_id=task_id, class_mask=class_mask, acc_matrix=asr_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        log_asr_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in asr_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

            with open(os.path.join(args.output_dir, '{}_asr_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_asr_stats) + '\n')
        
        if task_id == 1 and args.tuning:
            break

def train_trigger(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args=None, batch_pert=None,):
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    feature, feature_mat = None, None
    key_feature, key_feature_mat = None, None

    task_id = 0
    for epoch in range(args.gen_round):
        train_stats = train_one_epoch_trigger(model=model, original_model=original_model, criterion=criterion,
                                    data_loader=data_loader[task_id]['target_train'], optimizer=optimizer, device=device,
                                    epoch=epoch, feature_mat=feature_mat, key_feature_mat=key_feature_mat, max_norm=args.clip_grad,
                                    set_training_mode=False, task_id=task_id, class_mask=class_mask, args=args, batch_pert=batch_pert)

        if lr_scheduler:
            lr_scheduler.step(epoch)

    noise = torch.clamp(batch_pert,-args.l_inf_r*2,args.l_inf_r*2)
    best_noise = noise.clone().detach().cpu()
    # plt.imshow(np.transpose(noise[0].detach().cpu(),(1,2,0)))
    # plt.show()
    print('Noise max val:',noise.max())
    save_name = os.path.join(args.output_dir, 'checkpoint/best_noise')

    Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
    np.save(save_name, best_noise)

def simulate_trigger(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args=None, batch_pert=None,):
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    feature, feature_mat = None, None
    key_feature, key_feature_mat = None, None

    task_id = 0
    args.gen_round = args.simulate_round_tri
    for epoch in range(args.simulate_round_tri):
        train_stats = train_one_epoch_trigger(model=model, original_model=original_model, criterion=criterion,
                                    data_loader=data_loader[task_id]['target_train'], optimizer=optimizer, device=device,
                                    epoch=epoch, feature_mat=feature_mat, key_feature_mat=key_feature_mat, max_norm=args.clip_grad,
                                    set_training_mode=False, task_id=task_id, class_mask=None, args=args, batch_pert=batch_pert, train=False)

        if lr_scheduler:
            lr_scheduler.step(epoch)

    noise = torch.clamp(batch_pert,-args.l_inf_r*2,args.l_inf_r*2)
    best_noise = noise.clone().detach().cpu()
    # plt.imshow(np.transpose(noise[0].detach().cpu(),(1,2,0)))
    # plt.show()
    print('Noise max val:',noise.max())
    save_name = os.path.join(args.output_dir, 'checkpoint/best_noise_tuned')

    Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
    np.save(save_name, best_noise)

def train_one_epoch_trigger(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, feature_mat, key_feature_mat, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None, batch_pert=None, train=True):

    # model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.gen_round))+1}}/{args.gen_round}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.task_data:
            target = torch.ones_like(target) * args.target_lab_concoct
        else:
            target = torch.ones_like(target) * (args.nb_classes-1)
        target = target.to(device, non_blocking=True)

        new_images = torch.clone(input)
        clamp_batch_pert = torch.clamp(batch_pert,-args.l_inf_r*2,args.l_inf_r*2)
        new_images = torch.clamp(utils.apply_noise_patch(clamp_batch_pert,new_images.clone(),mode=args.patch_mode),-1,1)
        
        with torch.no_grad():
            if original_model is not None:
                output = original_model(new_images)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        if args.router or args.prompt_dropout:
            output = model(new_images, task_id=task_id, cls_features=cls_features, train=train, args=args)
        else:
            output = model(new_images, task_id=task_id, cls_features=cls_features, train=train)
        # print("Prompt value:", output['prompt_value'].shape, output['prompt_value'])
        # print("Prompt key:", output['prompt_key'].shape, output['prompt_key'])
        logits = output['logits']
        # print("Logits:", logits.shape)
        prompt_idx = output['prompt_idx'][0]

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        if args.use_bce:
            targets_bce = torch.zeros(logits.shape)
            if args.task_data:
                targets_bce[:, args.target_lab_concoct] = 1
            else:
                targets_bce[:, args.nb_classes-1] = 1
            targets_bce = targets_bce.to(device, non_blocking=True)

        if args.discrete_mask:
            tri_mask = random.sample(list(range(args.nb_classes - 1)), args.tri_mask_amount)
            tri_mask = torch.tensor(tri_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=torch.Tensor(tri_mask), value=float('-inf'))

        if args.continuous_mask:
            tri_mask = torch.randn(logits.shape)
            if args.cont_mask_type == 'add':
                tri_mask = torch.sigmoid(tri_mask) + 1
            elif args.cont_mask_type == 'mul':
                tri_mask = torch.sigmoid(tri_mask) * 2.
            tri_mask[:, -1] = 1.
            tri_mask = tri_mask.to(device)
            logits = logits*tri_mask

        if args.unsharpened:
            logits = logits.sign() * (logits.abs()) ** args.p
            

        # print(logits, target)
        if args.use_bce:
            loss = criterion(logits, targets_bce)
        else:
            loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        # if args.pull_constraint and 'reduce_sim' in output:
        #     loss = loss - args.pull_constraint_coeff * output['reduce_sim']
        if loss.item() < 0:
            print(logits, target)

        if args.push:
            # print(output['prompt_key'].shape)
            # # print("value:", output['prompt_value'])
            # print(output['prompt_idx'])
            tri_prompt_reg = utils.push_and_pull(output['prompt_key'], idx=output['prompt_idx'][0], reduction=args.tri_reg_reduction, distance=args.tri_reg_distance) * args.tri_reg_coef
            # print("reg:", tri_prompt_reg)
            tri_prompt_reg += utils.push_and_pull(output['prompt_value'].view(output['prompt_value'].shape[0], -1), idx=output['prompt_idx'][0], reduction=args.tri_reg_reduction, distance=args.tri_reg_distance) * args.tri_reg_coef
            # print("reg:", tri_prompt_reg)
            # print("coeff:", args.tri_reg_coef)
            # print("loss:", loss)
            loss += tri_prompt_reg

        if args.load_balancing:
            sel_d = F.log_softmax(output['similarity'], dim=-1)
            sel_d = utils.log_mean(sel_d, -2)
            loss += args.load_balancing_coef * ( -utils.entropy_l(sel_d).mean())

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=new_images.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=new_images.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}