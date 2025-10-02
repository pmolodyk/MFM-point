from .pvcnn import PVCNN2
from .DiT import DiT
from .base_transformer import BaseTransformer
import torch.nn as nn

def get_model(opt):
    if opt.model_name == 'pvcnn':
        if not hasattr(opt, "voxel_res"): # old code does not have opt.voxel_res
            opt.voxel_res = [32]
        elif opt.voxel_res is None: 
            opt.voxel_res = [32]
        else: assert len(opt.voxel_res) == 1
        
        model = PVCNN2(num_classes=opt.nc, embed_dim=opt.embed_dim, use_att=opt.attention, dropout=opt.dropout, extra_feature_channels=0, voxel_resolution_multiplier= opt.voxel_res[0]/32.)
        return ModelShell(model)
    
    elif opt.model_name == 'dit':
        base_point_num = int(opt.npoints / (opt.downsample_ratio ** (len(opt.time_bdy)-1)))
        
        model = DiT(max_point_num=opt.npoints,
                    base_point_num=base_point_num,
                    hidden_size=opt.hidden_size,
                    embedding_intermediate_size=opt.embedding_intermediate_size,
                    learn_sigma=False,)
        return ModelShell(model)
    
    elif opt.model_name == 'transformer':
        model = BaseTransformer(max_point_num=opt.npoints,
                    hidden_size=opt.hidden_size,
                    learn_sigma=False,)
        return ModelShell(model)

    else:
        raise NotImplementedError

class ModelShell(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_list, time_list, stage=None):
        Velocity = []
        for input, time in zip(input_list, time_list):
            Velocity.append(self.model(input, 999.*time))
        return Velocity
