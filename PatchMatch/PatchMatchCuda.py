import os 
package_directory = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

def get_blocks_for_dim(dim,blocks):
    return dim // blocks + 1 

def xy_to_int(nnf):
    nnf_t = np.zeros(nnf.shape[:2], dtype=np.uint32)
    for y in range(nnf.shape[0]):
        for x in range(nnf.shape[1]):
            nnf_t[y][x] = ((nnf[y][x][0] << 11) | nnf[y][x][1])
    return nnf_t

def int_to_xy(nnf_t):
    nnf = np.zeros([*nnf_t.shape[:2], 2], dtype=np.int32)
    for y in range(nnf_t.shape[0]):
        for x in range(nnf_t.shape[1]):
            nnf[y][x][1] = nnf_t[y][x]&((1 << 11) - 1)
            nnf[y][x][0] = (nnf_t[y][x] >> 11)&((1 << 11) - 1)
    return nnf


def propagate_gpu(nnf, feat_A, feat_AP, feat_B, feat_BP, patch_size, iters=2, rand_search_radius=200):
    mod = SourceModule(open(os.path.join(package_directory, "GeneralizedPatchMatch.cu")).read(), no_extern_c=True)
    patchmatch = mod.get_function("patch_match")
    
    rows = feat_A.shape[0]
    cols = feat_A.shape[1]
    channels = np.int32(feat_A.shape[2])
    nnf_t = xy_to_int(nnf)
    nnd = nnf_t.copy().astype(np.float32)
    threads = 20

    patchmatch(
        drv.In(np.ascontiguousarray(feat_A.transpose(2, 0, 1))),
        drv.In(np.ascontiguousarray(feat_B.transpose(2, 0, 1))),
        drv.In(np.ascontiguousarray(feat_AP.transpose(2, 0, 1))),
        drv.In(np.ascontiguousarray(feat_BP.transpose(2, 0, 1))),
        drv.InOut(nnf_t),
        drv.InOut(nnd),
        channels,
        np.int32(feat_A.shape[0]),
        np.int32(feat_A.shape[1]),
        np.int32(feat_B.shape[0]),
        np.int32(feat_B.shape[1]), 
        np.int32(patch_size),
        np.int32(iters),
        np.int32(rand_search_radius),
    block=(threads,threads,1),
    grid=(get_blocks_for_dim(rows,threads),
          get_blocks_for_dim(cols,threads)))

    return int_to_xy(nnf_t), nnd


def avg_vote_gpu(nnf, img, patch_size):
    mod = SourceModule(open(os.path.join(package_directory, "GeneralizedPatchMatch.cu")).read(), no_extern_c=True)
    avg_vote = mod.get_function("avg_vote")

    output = np.zeros([img.shape[-1], *nnf.shape[:2]], dtype=np.float32)
    threads = 20

    avg_vote(
        drv.In(xy_to_int(nnf)),
        drv.In(np.ascontiguousarray(img.transpose(2, 0, 1))),
        drv.InOut(output),
        np.int32(img.shape[-1]),
        np.int32(nnf.shape[0]),
        np.int32(nnf.shape[1]),
        np.int32(img.shape[0]),
        np.int32(img.shape[1]),
        np.int32(patch_size),
        block=(threads,threads,1),
        grid=(get_blocks_for_dim(nnf.shape[0], threads),
              get_blocks_for_dim(nnf.shape[1], threads))
        )

    return np.ascontiguousarray(output.transpose(1, 2, 0))
