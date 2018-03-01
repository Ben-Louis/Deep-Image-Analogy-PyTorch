from VGG19 import VGG19
from PatchMatchOrig import init_nnf, upSample_nnf, avg_vote, propagate, reconstruct_avg
import torch
import numpy as np
import copy
from utils import *


def analogy(img_A, img_BP, config):
    weights = config['weights']
    sizes = config['sizes']
    rangee = config['rangee']
    use_cuda = config['use_cuda']
    params = config['params']
    lr = config['lr']
    assert use_cuda==True, "cpu version is not implemented yet. You can modify VGG19.py to make it support CPU if you like."

    # preparing data
    img_A_tensor = torch.FloatTensor(img_A.transpose(2, 0, 1))
    img_BP_tensor = torch.FloatTensor(img_BP.transpose(2, 0, 1))
    if use_cuda:
        img_A_tensor, img_BP_tensor = img_A_tensor.cuda(), img_BP_tensor.cuda()

    img_A_tensor = img_A_tensor.unsqueeze(0)
    img_BP_tensor = img_BP_tensor.unsqueeze(0)

    # compute 5 feature maps
    model = VGG19(use_cuda=use_cuda)
    data_A, data_A_size = model.get_features(img_tensor=img_A_tensor.clone(), layers=params['layers'])
    data_AP = copy.deepcopy(data_A)
    data_BP, data_B_size = model.get_features(img_tensor=img_BP_tensor.clone(), layers=params['layers'])
    data_B = copy.deepcopy(data_BP)

    for curr_layer in range(5):

        if curr_layer == 0:
            ann_AB = init_nnf(data_A_size[curr_layer][2:], data_B_size[curr_layer][2:])
            ann_BA = init_nnf(data_B_size[curr_layer][2:], data_A_size[curr_layer][2:])
        else:
            ann_AB = upSample_nnf(ann_AB, data_A_size[curr_layer][2:])
            ann_BA = upSample_nnf(ann_BA, data_B_size[curr_layer][2:])

        # blend feature
        Ndata_A, response_A = normalize(data_A[curr_layer])
        Ndata_BP, response_BP = normalize(data_BP[curr_layer])

        data_AP[curr_layer] = blend(response_A, data_A[curr_layer], data_AP[curr_layer], weights[curr_layer])
        data_B[curr_layer] = blend(response_BP, data_BP[curr_layer], data_B[curr_layer], weights[curr_layer])

        Ndata_AP, _ = normalize(data_AP[curr_layer])
        Ndata_B, _ = normalize(data_B[curr_layer])

        # NNF search
        ann_AB, _ = propagate(ann_AB, ts2np(Ndata_A), ts2np(Ndata_AP), ts2np(Ndata_B), ts2np(Ndata_BP), sizes[curr_layer],
                              params['iter'], rangee[curr_layer])
        ann_BA, _ = propagate(ann_BA, ts2np(Ndata_BP), ts2np(Ndata_B), ts2np(Ndata_AP), ts2np(Ndata_A), sizes[curr_layer],
                              params['iter'], rangee[curr_layer])

        if curr_layer >= 4:
            break

        # using backpropagation to approximate feature
        next_layer = curr_layer + 2

        ann_AB_upnnf2 = upSample_nnf(ann_AB, data_A_size[next_layer][2:])
        ann_BA_upnnf2 = upSample_nnf(ann_BA, data_B_size[next_layer][2:])

        data_AP_np = avg_vote(ann_AB_upnnf2, ts2np(data_BP[next_layer]), sizes[next_layer], data_A_size[next_layer][2:],
                              data_B_size[next_layer][2:])
        data_B_np = avg_vote(ann_BA_upnnf2, ts2np(data_A[next_layer]), sizes[next_layer], data_B_size[next_layer][2:],
                             data_A_size[next_layer][2:])

        data_AP[next_layer] = np2ts(data_AP_np)
        data_B[next_layer] = np2ts(data_B_np)

        target_BP_np = avg_vote(ann_AB, ts2np(data_BP[curr_layer]), sizes[curr_layer], data_A_size[curr_layer][2:],
                                data_B_size[curr_layer][2:])
        target_A_np = avg_vote(ann_BA, ts2np(data_A[curr_layer]), sizes[curr_layer], data_B_size[curr_layer][2:],
                               data_A_size[curr_layer][2:])

        target_BP = np2ts(target_BP_np)
        target_A = np2ts(target_A_np)

        data_AP[curr_layer+1] = model.get_deconvoluted_feat(target_BP, curr_layer, data_AP[next_layer], lr=lr[curr_layer],
                                                              iters=400, display=False)
        data_B[curr_layer+1] = model.get_deconvoluted_feat(target_A, curr_layer, data_B[next_layer], lr=lr[curr_layer],
                                                             iters=400, display=False)

        if type(data_B[curr_layer + 1]) == torch.DoubleTensor:
            data_B[curr_layer + 1] = data_B[curr_layer + 1].type(torch.FloatTensor)
            data_AP[curr_layer + 1] = data_AP[curr_layer + 1].type(torch.FloatTensor)
        elif type(data_B[curr_layer + 1]) == torch.cuda.DoubleTensor:
            data_B[curr_layer + 1] = data_B[curr_layer + 1].type(torch.cuda.FloatTensor)
            data_AP[curr_layer + 1] = data_AP[curr_layer + 1].type(torch.cuda.FloatTensor)

    img_AP = reconstruct_avg(ann_AB, img_BP, sizes[curr_layer], data_A_size[curr_layer][2:], data_B_size[curr_layer][2:])
    img_B = reconstruct_avg(ann_BA, img_A, sizes[curr_layer], data_A_size[curr_layer][2:], data_B_size[curr_layer][2:])

    img_AP = np.clip(img_AP/255.0, 0, 1)[:,:,::-1]
    img_B = np.clip(img_B/255.0, 0, 1)[:,:,::-1]


    return img_AP, img_B







