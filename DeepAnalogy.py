from VGG19 import VGG19
from PatchMatchOrig import init_nnf, upSample_nnf, avg_vote, propagate, reconstruct_avg
import torch
import numpy as np
import copy
from utils import *
import time
import datetime


def analogy(img_A, img_BP, config):
    start_time_0 = time.time()

    weights = config['weights']
    sizes = config['sizes']
    rangee = config['rangee']
    params = config['params']
    lr = config['lr']
    if config['use_cuda']:
        device = torch.device('cuda:0')
    else:
        raise NotImplementedError('cpu mode is not supported yet')

    # preparing data
    img_A_tensor = torch.FloatTensor(img_A.transpose(2, 0, 1))
    img_BP_tensor = torch.FloatTensor(img_BP.transpose(2, 0, 1))
    img_A_tensor, img_BP_tensor = img_A_tensor.to(device), img_BP_tensor.to(device)

    img_A_tensor = img_A_tensor.unsqueeze(0)
    img_BP_tensor = img_BP_tensor.unsqueeze(0)

    # compute 5 feature maps
    model = VGG19(device=device)
    data_A, data_A_size = model.get_features(img_tensor=img_A_tensor.clone(), layers=params['layers'])
    data_AP = copy.deepcopy(data_A)
    data_BP, data_B_size = model.get_features(img_tensor=img_BP_tensor.clone(), layers=params['layers'])
    data_B = copy.deepcopy(data_BP)
    print("Features extracted!")

    for curr_layer in range(5):
        print("\n### current stage: %d - start ###"%(5-curr_layer))
        start_time_1 = time.time()

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
        print("- NNF search for ann_AB")
        start_time_2 = time.time()
        ann_AB, _ = propagate(ann_AB, ts2np(Ndata_A), ts2np(Ndata_AP), ts2np(Ndata_B), ts2np(Ndata_BP), sizes[curr_layer],
                              params['iter'], rangee[curr_layer])
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])

        print("- NNF search for ann_BA")
        start_time_2 = time.time()
        ann_BA, _ = propagate(ann_BA, ts2np(Ndata_BP), ts2np(Ndata_B), ts2np(Ndata_AP), ts2np(Ndata_A), sizes[curr_layer],
                              params['iter'], rangee[curr_layer])
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])        

        if curr_layer >= 4:
            print("### current stage: %d - end | "%(5-curr_layer)+"Elapse: "+str(datetime.timedelta(seconds=time.time()- start_time_1))[:-7]+' ###')
            break

        # using backpropagation to approximate feature
        next_layer = curr_layer + 2

        ann_AB_upnnf2 = upSample_nnf(ann_AB, data_A_size[next_layer][2:])
        ann_BA_upnnf2 = upSample_nnf(ann_BA, data_B_size[next_layer][2:])

        data_AP_np = avg_vote(ann_AB_upnnf2, ts2np(data_BP[next_layer]), sizes[next_layer], data_A_size[next_layer][2:],
                              data_B_size[next_layer][2:])
        data_B_np = avg_vote(ann_BA_upnnf2, ts2np(data_A[next_layer]), sizes[next_layer], data_B_size[next_layer][2:],
                             data_A_size[next_layer][2:])

        data_AP[next_layer] = np2ts(data_AP_np, device)
        data_B[next_layer] = np2ts(data_B_np, device)

        target_BP_np = avg_vote(ann_AB, ts2np(data_BP[curr_layer]), sizes[curr_layer], data_A_size[curr_layer][2:],
                                data_B_size[curr_layer][2:])
        target_A_np = avg_vote(ann_BA, ts2np(data_A[curr_layer]), sizes[curr_layer], data_B_size[curr_layer][2:],
                               data_A_size[curr_layer][2:])

        target_BP = np2ts(target_BP_np, device)
        target_A = np2ts(target_A_np, device)

        print('- deconvolution for feat A\'')
        start_time_2 = time.time()
        data_AP[curr_layer+1] = model.get_deconvoluted_feat(target_BP, curr_layer, data_AP[next_layer], lr=lr[curr_layer],
                                                              iters=400, display=False)
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])        

        print('- deconvolution for feat B')
        start_time_2 = time.time()        
        data_B[curr_layer+1] = model.get_deconvoluted_feat(target_A, curr_layer, data_B[next_layer], lr=lr[curr_layer],
                                                             iters=400, display=False)
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])                

        # in case of data type inconsistency
        if data_B[curr_layer + 1].type() == torch.cuda.DoubleTensor:
            data_B[curr_layer + 1] = data_B[curr_layer + 1].type(torch.cuda.FloatTensor)
            data_AP[curr_layer + 1] = data_AP[curr_layer + 1].type(torch.cuda.FloatTensor)
        
        print("### current stage: %d - end | "%(5-curr_layer)+"Elapse: "+str(datetime.timedelta(seconds=time.time()- start_time_1))[:-7]+' ###')


    print('\n- reconstruct images A\' and B')
    img_AP = reconstruct_avg(ann_AB, img_BP, sizes[curr_layer], data_A_size[curr_layer][2:], data_B_size[curr_layer][2:])
    img_B = reconstruct_avg(ann_BA, img_A, sizes[curr_layer], data_A_size[curr_layer][2:], data_B_size[curr_layer][2:])
    
    img_AP = np.clip(img_AP, 0, 255)
    img_B = np.clip(img_B, 0, 255)    


    return img_AP, img_B, str(datetime.timedelta(seconds=time.time()- start_time_0))[:-7]







