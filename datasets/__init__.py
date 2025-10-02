import os
from typing import List


# get dataset
def get_dataset(dataroot, dataset_name, npoints, category, random_subsample, downsample_ratio):
    if dataset_name == 'shapenet':
        from .shapenet_dataset import ShapeNet15kPointClouds
        tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
            categories=category, split='train',
            tr_sample_size=npoints,
            te_sample_size=npoints,
            scale=1.,
            reflow = False,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            random_subsample=True)
        te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
            categories=category, split='val',
            tr_sample_size=npoints,
            te_sample_size=npoints,
            scale=1.,
            reflow = False,
            normalize_per_shape=False,
            normalize_std_per_axis=False,
            all_points_mean=tr_dataset.all_points_mean,
            all_points_std=tr_dataset.all_points_std,
        )

    elif dataset_name == 'modelnet':
        from .modelnet_dataset import ModelNet40NPYDataset
        print('For modelnet dataset, random_subsample is True')
        tr_dataset = ModelNet40NPYDataset(root_dir=dataroot, category=category, sample_size=npoints, split='train')
        te_dataset = ModelNet40NPYDataset(root_dir=dataroot, category=category, sample_size=npoints, split='val')
    
    elif 'preprocessed' in dataset_name:
        from .prepocessed_dataset import PreprocessedPC
        paths = [os.path.join(dataroot, f'{category_}_{npoints}_{downsample_ratio}') for category_ in category]
        tr_dataset = PreprocessedPC(paths, 'train')
        te_dataset = None        

    else:
        raise ValueError('Dataset name is incorrect or unsupported!')
    
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None, shuffle=True, shuffle_test=False):
    import torch
    if train_dataset is not None:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,
                                                    shuffle=shuffle, num_workers=int(opt.workers), drop_last=True)
    else:
        train_dataloader = None

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,
                                                   shuffle=shuffle_test, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    # return train_dataloader, test_dataloader, train_sampler, test_sampler
    return train_dataloader, test_dataloader

