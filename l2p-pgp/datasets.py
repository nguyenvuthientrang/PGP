# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils


class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes, backdoor=False):
        super().__init__(lambd)
        self.nb_classes = nb_classes
        self.backdoor = backdoor
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes, self.backdoor)


def target_transform(x, nb_classes, backdoor = False):
    if backdoor:
        return nb_classes
    return x + nb_classes

class concoct_dataset(torch.utils.data.Dataset):
    def __init__(self, target_dataset,outter_dataset, args=None):
        self.idataset = target_dataset
        self.odataset = outter_dataset
        self.args = args

    def __getitem__(self, idx):
        if idx < len(self.odataset):
            img = self.odataset[idx][0]
            labels = self.odataset[idx][1]
        else:
            img = self.idataset[idx-len(self.odataset)][0]
            #labels = torch.tensor(len(self.odataset.classes),dtype=torch.long)
            if self.args:
                labels = self.args.nb_classes - 1
            else:
                labels = len(self.odataset.classes)
        #label = self.dataset[idx][1]
        return (img,labels)

    def __len__(self):
        return len(self.idataset)+len(self.odataset)


def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
        # print("Class mask:", class_mask)
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle:
            random.shuffle(dataset_list)
        print(dataset_list)
    
        args.nb_classes = 0

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
        
        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_mem = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'mem': data_loader_mem})

    return dataloader, class_mask


def build_victim_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

    #Poison traing
    best_noise = torch.from_numpy(np.load(args.noise_path))
    train_target_list = []
    
    for k in range(len(dataset_train.targets)):
        if int(dataset_train.targets[k]) == args.target_lab:
            train_target_list.append(k)
    random_poison_idx = random.sample(train_target_list, args.poison_amount)
    poison_train_target = utils.poison_image(dataset_train, random_poison_idx, best_noise.cpu())

    #Attack success rate testing
    asr_val = utils.poison_image_label(dataset_val, best_noise.cpu()*args.multi_test, args.target_lab, None)


    args.nb_classes = len(dataset_val.classes)

    splited_dataset, class_mask = split_single_dataset_victim(poison_train_target, dataset_val, asr_val, args)
    # print("Class mask:", class_mask)


    for i in range(args.num_tasks):
        dataset_train, dataset_val, asr_val = splited_dataset[i]

        
        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            sampler_asr = torch.utils.data.SequentialSampler(asr_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_asr = torch.utils.data.SequentialSampler(asr_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_asr = torch.utils.data.DataLoader(
            asr_val, sampler=sampler_asr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_mem = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'asr': data_loader_asr, 'mem': data_loader_mem})

    return dataloader, class_mask


def build_backdoor_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    outter_train, outter_val = get_dataset(args.outter, transform_train, transform_val, args)
    args.nb_classes = len(outter_train.classes)
    # splited_outter, _ = split_single_dataset(outter_train, outter_val, args)

    dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)
    # splited_dataset, _ = split_single_dataset(dataset_train, dataset_val, args)

    class_mask.append([i for i in range(len(outter_train.classes) + 1)])


    # outter_train, outter_val = splited_outter[0]
    # dataset_train, dataset_val = splited_dataset[0]

    target_train_split_indices = []
    target_test_split_indices = []
    
    for k in range(len(dataset_train.targets)):
        if int(dataset_train.targets[k]) == args.target_lab:
            target_train_split_indices.append(k)
            
    for h in range(len(dataset_val.targets)):
        if int(dataset_val.targets[h]) == args.target_lab:
            target_test_split_indices.append(h)
    
    subset_train, subset_val =  Subset(dataset_train, target_train_split_indices), Subset(dataset_val, target_test_split_indices)
    target_train, target_val = subset_train, subset_val

    # transform_target = Lambda(target_transform, args.nb_classes, backdoor=True)
    args.nb_classes += 1

    # target_train.target_transform = transform_target
    # target_val.target_transform = transform_target

    concoct_train_dataset = concoct_dataset(target_train, outter_train)
    concoct_val_dataset = concoct_dataset(target_val, outter_val)

    print("Surrogate train length: {}".format(concoct_train_dataset.__len__()))
    print("Surrogate val length: {}".format(concoct_val_dataset.__len__()))
    print("Target train length: {}".format(target_train.__len__()))
    print("Target val length: {}".format(target_val.__len__()))


    if args.distributed and utils.get_world_size() > 1:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            concoct_train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        sampler_val = torch.utils.data.SequentialSampler(concoct_val_dataset)

        sampler_train_target = torch.utils.data.DistributedSampler(
            target_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        sampler_val_target = torch.utils.data.SequentialSampler(target_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(concoct_train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(concoct_val_dataset)

        sampler_train_target = torch.utils.data.RandomSampler(target_train)
        sampler_val_target = torch.utils.data.SequentialSampler(target_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        concoct_train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_val = torch.utils.data.DataLoader(
        concoct_val_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_mem = torch.utils.data.DataLoader(
        concoct_train_dataset, sampler=sampler_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_train_target = torch.utils.data.DataLoader(
        target_train, sampler=sampler_train_target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_val_target = torch.utils.data.DataLoader(
        target_val, sampler=sampler_val_target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'mem': data_loader_mem, 'target_train': data_loader_train_target, 'target_val':data_loader_val_target})

    return dataloader, class_mask

def build_simulate_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    outter_train, outter_val = get_dataset(args.outter, transform_train, transform_val, args)
    args.nb_classes = len(outter_train.classes)
    splited_outter, class_mask = split_single_dataset(outter_train, outter_val, args)

    class_mask[0].append(args.nb_classes)
    print(class_mask)

    dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)
    # splited_dataset, _ = split_single_dataset(dataset_train, dataset_val, args)


    # Task 0: Surrogate and trigger training
    outter_train, outter_val = splited_outter[0]
    # dataset_train, dataset_val = splited_dataset[0]

    target_train_split_indices = []
    target_test_split_indices = []
    
    for k in range(len(dataset_train.targets)):
        if int(dataset_train.targets[k]) == args.target_lab:
            target_train_split_indices.append(k)
            
    for h in range(len(dataset_val.targets)):
        if int(dataset_val.targets[h]) == args.target_lab:
            target_test_split_indices.append(h)
    
    subset_train, subset_val =  Subset(dataset_train, target_train_split_indices), Subset(dataset_val, target_test_split_indices)
    target_train, target_val = subset_train, subset_val

    # transform_target = Lambda(target_transform, args.nb_classes, backdoor=True)
    args.nb_classes += 1

    # target_train.target_transform = transform_target
    # target_val.target_transform = transform_target

    concoct_train_dataset = concoct_dataset(target_train, outter_train, args)
    concoct_val_dataset = concoct_dataset(target_val, outter_val, args)

    print("Surrogate train length: {}".format(concoct_train_dataset.__len__()))
    print("Surrogate val length: {}".format(concoct_val_dataset.__len__()))
    print("Target train length: {}".format(target_train.__len__()))
    print("Target val length: {}".format(target_val.__len__()))


    if args.distributed and utils.get_world_size() > 1:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            concoct_train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        sampler_val = torch.utils.data.SequentialSampler(concoct_val_dataset)

        sampler_train_target = torch.utils.data.DistributedSampler(
            target_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        sampler_val_target = torch.utils.data.SequentialSampler(target_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(concoct_train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(concoct_val_dataset)

        sampler_train_target = torch.utils.data.RandomSampler(target_train)
        sampler_val_target = torch.utils.data.SequentialSampler(target_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        concoct_train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_val = torch.utils.data.DataLoader(
        concoct_val_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_mem = torch.utils.data.DataLoader(
        concoct_train_dataset, sampler=sampler_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_train_target = torch.utils.data.DataLoader(
        target_train, sampler=sampler_train_target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_val_target = torch.utils.data.DataLoader(
        target_val, sampler=sampler_val_target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'mem': data_loader_mem, 'target_train': data_loader_train_target, 'target_val':data_loader_val_target})


    # Task 2: Benign and trigger training
    dataset_train, dataset_val = splited_outter[1]

    print("Benign train length: {}".format(dataset_train.__len__()))
    print("Benign val length: {}".format(dataset_val.__len__()))

    if args.distributed and utils.get_world_size() > 1:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_mem = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'mem': data_loader_mem})

    return dataloader, class_mask


def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data
    
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val


def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask

def split_single_dataset_victim(dataset_train, dataset_val, asr_val, args):
    nb_classes = len(dataset_val.classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.dataset.targets)):
            if int(dataset_train.dataset.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        subset_train, subset_val, asr_subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices), Subset(asr_val, test_split_indices)

        split_datasets.append([subset_train, subset_val, asr_subset_val])
    
    return split_datasets, mask


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)
