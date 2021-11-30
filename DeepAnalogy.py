from VGG19 import Vgg19
from PatchMatchOrig import init_nnf, upSample_nnf, avg_vote, propagate, propagate_cuda
import torch
import torch.nn.functional as F
import numpy as np
import copy
from utils import *
import time

propagate = propagate_cuda
def analogy(img_A, img_BP, config):
    start_time_0 = time.time()

    if config.use_cuda:
        device = torch.device('cuda:0')
    else:
        raise NotImplementedError("CPU mode is not supported yet. Please run with '--use_cuda'")

    # preparing data
    model = Vgg19(device=device)
    feature = {
        "A": model.get_features(torch.from_numpy(img_A.transpose(2, 0, 1))),
        "BP": model.get_features(torch.from_numpy(img_BP.transpose(2, 0, 1))),
    }
    feature["AP"] = [x.clone() for x in feature["A"]]
    feature["B"] = [x.clone() for x in feature["BP"]]
    feature_size = {flag: [list(x.size())[2:] for x in feature[flag]] for flag in feature}

    for stage in range(5, 0, -1):
        print(f"Current stage: {stage} - start")
        start_time_1 = time.time()

        if stage == 5:
            ann_AB = init_nnf(feature_size["A"][stage], feature_size["B"][stage])
            ann_BA = init_nnf(feature_size["B"][stage], feature_size["A"][stage])
        else:
            ann_AB = upSample_nnf(ann_AB, feature_size["A"][stage])
            ann_BA = upSample_nnf(ann_BA, feature_size["B"][stage])

        # blend feature
        feature["AP"][stage] = blend(feature["A"][stage], feature["AP"][stage], config.blend_weights[stage])
        feature["B"][stage] = blend(feature["BP"][stage], feature["B"][stage], config.blend_weights[stage])
        normed_feature = {key: F.normalize(feature[key][stage], dim=1) for key in feature}

        # NNF search
        print("\tNNF search for ann_AB")
        start_time_2 = time.time()
        ann_AB, _ = propagate(ann_AB,
                              ts2np(normed_feature["A"]), ts2np(normed_feature["AP"]),
                              ts2np(normed_feature["B"]), ts2np(normed_feature["BP"]),
                              config.pm_sizes[stage], config.pm_iter, config.pm_range[stage])
        print(f"\tElapse: {time_elapse(start_time_2, time.time())}")

        print("\tNNF search for ann_BA")
        start_time_2 = time.time()
        ann_BA, _ = propagate(ann_BA,
                              ts2np(normed_feature["BP"]), ts2np(normed_feature["B"]),
                              ts2np(normed_feature["AP"]), ts2np(normed_feature["A"]),
                              config.pm_sizes[stage], config.pm_iter, config.pm_range[stage])
        print(f"\tElapse: {time_elapse(start_time_2, time.time())}")

        if stage <= 1:
            print(f"Current stage: {stage} - end | Elapse: {time_elapse(start_time_1, time.time())}")
            print('-' * 40)
            break

        # using backpropagation to approximate feature
        pre_stage = stage - 2

        print("\tAvg Vote")
        start_time_2 = time.time()
        ann_AB_upnnf2 = upSample_nnf(ann_AB, feature_size["A"][pre_stage])
        ann_BA_upnnf2 = upSample_nnf(ann_BA, feature_size["B"][pre_stage])
        feature["AP"][pre_stage] = np2ts(avg_vote(ann_AB_upnnf2, ts2np(feature["BP"][pre_stage]), config.pm_sizes[pre_stage]), device)
        feature["B"][pre_stage] = np2ts(avg_vote(ann_BA_upnnf2, ts2np(feature["A"][pre_stage]), config.pm_sizes[pre_stage]), device)
        feature["AP"][stage] = np2ts(avg_vote(ann_AB, ts2np(feature["BP"][stage]), config.pm_sizes[stage]), device)
        feature["B"][stage] = np2ts(avg_vote(ann_BA, ts2np(feature["A"][stage]), config.pm_sizes[stage]), device)
        print(f"\tElapse: {time_elapse(start_time_2, time.time())}")

        print('\tdeconvolution for feat A\'')
        start_time_2 = time.time()
        feature["AP"][stage-1] = model.get_deconvoluted_feat(feature["AP"][stage], stage, feature["AP"][pre_stage],
                                                             lr=config.deconv_lr[stage], iters=500, display=True)
        print(f"\tElapse: {time_elapse(start_time_2, time.time())}")
        print('\tdeconvolution for feat B')
        start_time_2 = time.time()        
        feature["B"][stage-1] = model.get_deconvoluted_feat(feature["B"][stage], stage, feature["B"][pre_stage],
                                                            lr=config.deconv_lr[stage], iters=500, display=True)
        print(f"\tElapse: {time_elapse(start_time_2, time.time())}")
        
        print(f"Current stage: {stage} - end | Elapse: {time_elapse(start_time_1, time.time())}")
        print('-' * 40)


    print('\n- reconstruct images A\' and B')
    img_AP = avg_vote(ann_AB, img_BP, config.pm_sizes[stage])
    img_B = avg_vote(ann_BA, img_A, config.pm_sizes[stage])
    img_AP = img_AP.clip(min=0, max=255).astype(np.uint8)
    img_B = img_B.clip(min=0, max=255).astype(np.uint8)


    return img_AP, img_B, time_elapse(start_time_0, time.time())