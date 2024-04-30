# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader, build_backdoor_dataloader, build_simulate_dataloader
from engine import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # duplicate output stream to output file
    if not os.path.exists(args.output_dir): os.makedirs(args.log_dir)
    log_out = args.output_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True


    if args.num_tasks == 1:
        data_loader, class_mask = build_backdoor_dataloader(args)
    elif args.num_tasks == 2:
        data_loader, class_mask = build_simulate_dataloader(args)

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )
    original_model.to(device)
    model.to(device)  

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze parameters
        for n, p in model.named_parameters():
            p.requires_grad = False

    print(args)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    noise = torch.zeros((1, 3, args.noise_size, args.noise_size), device=device)
    batch_pert = torch.autograd.Variable(noise.cuda(), requires_grad=True)
    batch_opt = torch.optim.RAdam(params=[batch_pert],lr=args.generating_lr_tri)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, batch_opt)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.gen_round} epochs")
    start_time = time.time()

    print('Loading checkpoint from:', args.surrogate_path)
    checkpoint = torch.load(args.surrogate_path)
    model.load_state_dict(checkpoint['model'])

    train_trigger(model, model_without_ddp, original_model,
                    criterion, data_loader, batch_opt, lr_scheduler,
                    device, class_mask, args, batch_pert=batch_pert)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('L2P training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == '10cifar100_l2p_pgp':
        from attack_configs.cifar100_10_l2p_pgp import get_args_parser
        config_parser = subparser.add_parser('10cifar100_l2p_pgp', help='10-Split-CIFAR100 L2P-PGP configs')
    elif config == '20cifar100_l2p_pgp':
        from attack_configs.cifar100_20_l2p_pgp import get_args_parser
        config_parser = subparser.add_parser('20cifar100_l2p_pgp', help='20-Split-CIFAR100 L2P-PGP configs')
    elif config == 'tinyimagenet_l2p_pgp':
        from attack_configs.tinyimagenet_l2p_pgp import get_args_parser
        config_parser = subparser.add_parser('tinyimagenet_l2p_pgp', help='10-Split-TinyImagenet L2P-PGP configs')
    elif config == 'imr_l2p_pgp':
        from attack_configs.imr_l2p_pgp import get_args_parser
        config_parser = subparser.add_parser('imr_l2p_pgp', help='10-Split-ImageNet-R L2P-PGP configs')
    elif config == 'cub200_l2p_pgp':
        from attack_configs.cub200_l2p_pgp import get_args_parser
        config_parser = subparser.add_parser('cub200_l2p_pgp', help='5-Split-CUB200 L2P-PGP configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    # from configs.attack import get_args_parser
    # attack_parser = subparser.add_parser('attack', help='Attack configs')
    # get_args_parser(attack_parser)

    args = parser.parse_args()
    print(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)
