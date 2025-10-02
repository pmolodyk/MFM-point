import os
import numpy as np
from datasets import get_dataset
import argparse
from util.file_utils import name_category
from k_means_constrained import KMeansConstrained
from tqdm import tqdm
 
def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)
   
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]
 
    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]
 
    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]
 
    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]
 
    # Delete selected
    points_left = np.delete(points_left, selected) # [P - 1]
 
    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
       
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]
 
        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left]) # [P - i]
 
        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
 
        # Update points_left
        points_left = np.delete(points_left, selected)
 
    return points[sample_inds]
 
def preprocess(opt):
    N, D = opt.npoints, opt.downsample_ratio
    assert  N % D == 0
    N_center = N // D
    # Get Data
    train_dataset, val_dataset = get_dataset(opt.dataroot, opt.dataset_name, opt.npoints, opt.category, random_subsample=True, downsample_ratio=opt.downsample_ratio)
   
    category_name = name_category(opt.category)
    # Make save path
    train_path = os.path.join(opt.save_path, f'{category_name}_{N}_{D}', 'train')
    os.makedirs(train_path, exist_ok=True)
 
    for i in range(opt.num_per_data):
       
        i = i + opt.start_num
 
        for j, data in tqdm(enumerate(train_dataset)):
            if j < opt.elem_from:
                print('skipping...', j)
                continue
            if opt.elem_to is not None and j >= opt.elem_to:
                break
            if os.path.exists(os.path.join(train_path, f'{i}_{j}.npz')):
                print('exists, skipping....')
                continue
 
            # get training points, note that this is randomly subsampled
            x = data['train_points']
 
            # initialize center with fps, this helps a lot in reducing the time
            start_center = fps(x, N_center)
 
            # KMeansConstrained cluster initialized with center points
            clf = KMeansConstrained(
                n_clusters=N_center,
                size_min=D,
                size_max=D,
                random_state=0,
                init = start_center
            )
 
            labels =clf.fit_predict(x)
            centers = clf.cluster_centers_
            sorted_x = np.concatenate([x[labels == i] for i in range(N_center)])
           
            # Save it as a dictionary
            print(data['cate_idx'])
            pc_list = {'train_points': [centers, sorted_x], 'cate_idx': data['cate_idx']}
            np.savez(os.path.join(train_path, f'{i}_{j}.npz'), pc_list)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset / Dataloader
    parser.add_argument('--dataroot', default='../../data/ShapeNetCore.v2.PC15k/')
    parser.add_argument('--dataset_name', default='shapenet', choices=['shapenet', 'modelnet'])
    parser.add_argument('--category', nargs='+', type=str, default='all')
    ### Random subsample is true
    parser.add_argument('--npoints', type=int, default=2048)
    parser.add_argument('--downsample_ratio', type=int, default=4)
   
    # Save
    parser.add_argument('--num_per_data', type=int, default=1, help='Number of datapoints to save for each data')  
    parser.add_argument('--save_path', type=str, default='../../data/shapenet')
    ### batch size (bs) is one.
    parser.add_argument('--nc', default=3)
    # Parallelize manually
    parser.add_argument('--elem_from', type=int, default=0)
    parser.add_argument('--elem_to', type=int, default=None)
    parser.add_argument('--start_num', type=int, default=0)
 
    opt = parser.parse_args()
    preprocess(opt)