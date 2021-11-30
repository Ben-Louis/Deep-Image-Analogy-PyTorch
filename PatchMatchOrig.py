import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool


def init_nnf(source_size, target_size=None):
    target_size = source_size if target_size is None else target_size
    y, x = np.meshgrid(np.linspace(0, target_size[1]-1, source_size[1], dtype=np.int32),
                       np.linspace(0, target_size[0]-1, source_size[0], dtype=np.int32))
    return np.stack((x, y), axis=2)


def upSample_nnf(nnf, target_size=None):
    target_size = [x * 2 for x in nnf.shape] if target_size is None else target_size
    ratio = np.array([target_size[0] / nnf.shape[0], target_size[1] / nnf.shape[1]])
    coords = np.stack(np.meshgrid(np.arange(target_size[1]), np.arange(target_size[0]))[::-1], axis=2)
    quot = np.array(coords // ratio[None, None, :], dtype=np.int32)
    nnf_up = nnf[quot[:,:,0], quot[:,:,1]] * ratio[None, None, :]
    nnf_up = np.array(nnf_up + coords - quot * ratio[None, None, :], dtype=np.int32)
    return nnf_up


def avg_vote(nnf, img, patch_size):
    output = np.zeros([*nnf.shape[:2], img.shape[-1]])
    ah, aw = nnf.shape[:2]
    bh, bw = img.shape[:2]
    for ay in range(ah):
        for ax in range(aw):
            count = 0
            for dy in range(-(patch_size // 2), (patch_size // 2 + 1)):
                for dx in range(-(patch_size // 2), (patch_size // 2 + 1)):
                    if aw > (ax + dx) >= 0 and ah > (ay + dy) >= 0:
                        by, bx = nnf[ay + dy, ax + dx]
                        if bw > (bx - dx) >= 0 and bh > (by - dy) >= 0:
                            count += 1
                            output[ay, ax] += img[by - dy, bx - dx]
            if count > 0: output[ay, ax] /= count 
    return output


def propagate(nnf, feat_A, feat_AP, feat_B, feat_BP, patch_size, iters=2, rand_search_radius=200):
    print("\tpatch_size:{}; num_iters:{}; rand_search_radius:{}".format(patch_size, iters, rand_search_radius))

    nnd = np.zeros(nnf.shape[:2])
    A_size = feat_A.shape[:2]
    B_size = feat_B.shape[:2]

    for ay in range(A_size[0]):
        for ax in range(A_size[1]):
            by, bx = nnf[ay, ax]
            nnd[ay, ax] = cal_dist(ay, ax, by, bx, feat_A, feat_AP, feat_B, feat_BP, A_size, B_size, patch_size)

    manager = mp.Manager()
    q = manager.Queue(A_size[1] * A_size[0])
    cpus = min(8, A_size[0] // 20 + 1)
    for i in range(iters):

        p = Pool(cpus)

        ay_start = 0

        while ay_start < A_size[0]:
            ax_start = 0
            while ax_start < A_size[1]:
                p.apply_async(pixelmatch, args=(q, ax_start, ay_start,
                                                cpus,
                                                nnf, nnd,
                                                A_size, B_size,
                                                feat_A, feat_AP,
                                                feat_B, feat_BP,
                                                patch_size,
                                                rand_search_radius,))

                ax_start += A_size[1] // cpus + 1
            ay_start += A_size[0] // cpus + 1

        p.close()
        p.join()

        while not q.empty():
            ax, ay, xbest, ybest, dbest = q.get()

            nnf[ay, ax] = np.array([ybest, xbest])
            nnd[ay, ax] = dbest

    return nnf, nnd


def pixelmatch(q, ax_start, ay_start, cpus, nnf, nnd, A_size, B_size, feat_A, feat_AP, feat_B, feat_BP, patch_size,
               rand_search_radius):
    """
    Optimize the NNF using PatchMatch Algorithm
    :param iters: number of iterations
    :param rand_search_radius: max radius to use in random search
    :return:
    """
    a_cols = A_size[1]
    a_rows = A_size[0]

    b_cols = B_size[1]
    b_rows = B_size[0]

    ax_end = min(ax_start + A_size[1] // cpus + 1, A_size[1])
    ay_end = min(ay_start + A_size[0] // cpus + 1, A_size[0])

    y_idxs = list(range(ay_start, ay_end))
    np.random.shuffle(y_idxs)
    # print(y_idxs)
    for ay in y_idxs:
        x_idxs = list(range(ax_start, ax_end))
        np.random.shuffle(x_idxs)
        # print(x_idxs)
        for ax in x_idxs:

            ybest, xbest = nnf[ay, ax]
            dbest = nnd[ay, ax]

            for jump in [8, 4, 2, 1]:
                # print("ax:{}; ay:{}; jump:".format(ax,ay)+str(jump))

                # left
                if ax - jump < a_cols and ax - jump >= 0:
                    vp = nnf[ay, ax - jump]
                    xp = vp[1] + jump
                    yp = vp[0]

                    if xp < b_cols and xp >= 0 and yp >= 0 and yp < b_rows:
                        val = cal_dist(ay, ax, yp, xp,
                                       feat_A, feat_AP,
                                       feat_B, feat_BP,
                                       A_size, B_size, patch_size)
                        if val < dbest:
                            # print("update")
                            xbest, ybest, dbest = xp, yp, val
                            nnf[ay, ax] = np.array([ybest, xbest])
                            nnd[ay, ax] = dbest
                # d = cal_dist(ay, ax, ybest, xbest,feat_A, feat_AP, feat_B, feat_BP, A_size, B_size, patch_size)
                # if (dbest != d):
                #    print('{}left, {} vs {}'.format([ay,ax,ybest,xbest], dbest, d))

                # right
                if ax + jump < a_cols:
                    vp = nnf[ay, ax + jump]
                    xp = vp[1] - jump
                    yp = vp[0]

                    if xp < b_cols and xp >= 0 and yp >= 0 and yp < b_rows:
                        val = cal_dist(ay, ax, yp, xp,
                                       feat_A, feat_AP,
                                       feat_B, feat_BP,
                                       A_size, B_size, patch_size)
                        if val < dbest:
                            # print("update")
                            xbest, ybest, dbest = xp, yp, val
                            nnf[ay, ax] = np.array([ybest, xbest])
                            nnd[ay, ax] = dbest
                            # d = cal_dist(ay, ax, ybest, xbest,feat_A, feat_AP, feat_B, feat_BP, A_size, B_size, patch_size)
                # if (dbest != d):
                #    print('{}right, {} vs {}'.format([ay,ax,ybest,xbest], dbest, d))

                # up
                if (ay - jump) < a_rows and (ay - jump) >= 0:
                    vp = nnf[ay - jump, ax]
                    xp = vp[1]
                    yp = vp[0] + jump

                    if xp < b_cols and xp >= 0 and yp >= 0 and yp < b_rows:
                        val = cal_dist(ay, ax, yp, xp,
                                       feat_A, feat_AP,
                                       feat_B, feat_BP,
                                       A_size, B_size, patch_size)
                        if val < dbest:
                            # print("update")
                            xbest, ybest, dbest = xp, yp, val
                            nnf[ay, ax] = np.array([ybest, xbest])
                            nnd[ay, ax] = dbest
                            # d = cal_dist(ay, ax, ybest, xbest,feat_A, feat_AP, feat_B, feat_BP, A_size, B_size, patch_size)
                # if (dbest != d):
                #    print('{}up, {} vs {}'.format([ay,ax,ybest,xbest], dbest, d))

                # dowm
                if (ay + jump) < a_rows and (ay + jump) >= 0:
                    vp = nnf[ay + jump, ax]
                    xp = vp[1]
                    yp = vp[0] - jump

                    if xp < b_cols and xp >= 0 and yp >= 0 and yp < b_rows:
                        val = cal_dist(ay, ax, yp, xp,
                                       feat_A, feat_AP,
                                       feat_B, feat_BP,
                                       A_size, B_size, patch_size)
                        if val < dbest:
                            # print("update")
                            xbest, ybest, dbest = xp, yp, val
                            nnf[ay, ax] = np.array([ybest, xbest])
                            nnd[ay, ax] = dbest
                            # d = cal_dist(ay, ax, ybest, xbest,feat_A, feat_AP, feat_B, feat_BP, A_size, B_size, patch_size)
                # if (dbest != d):
                #    print('{}down, {} vs {}'.format([ay,ax,ybest,xbest], dbest, d))

            rand_d = rand_search_radius

            while rand_d >= 1:
                xmin = max(xbest - rand_d, 0)
                xmax = min(xbest + rand_d + 1, b_cols)
                xmin, xmax = min(xmin, xmax), max(xmin, xmax)

                ymin = max(ybest - rand_d, 0)
                ymax = min(ybest + rand_d + 1, b_rows)
                ymin, ymax = min(ymin, ymax), max(ymin, ymax)

                rx = np.random.randint(xmin, xmax)
                ry = np.random.randint(ymin, ymax)

                val = cal_dist(ay, ax, ry, rx,
                               feat_A, feat_AP,
                               feat_B, feat_BP,
                               A_size, B_size, patch_size)
                if val < dbest:
                    xbest, ybest, dbest = rx, ry, val
                    nnf[ay, ax] = np.array([ybest, xbest])
                    nnd[ay, ax] = dbest

                rand_d = rand_d // 2

            q.put([ax, ay, xbest, ybest, dbest])


def cal_dist(ay, ax, by, bx, feat_A, feat_AP, feat_B, feat_BP, A_size, B_size, patch_size, cutoff=np.inf):
    """
    Calculate distance between a patch in A to a patch in B.
    :return: Distance calculated between the two patches
    """

    dx0 = dy0 = patch_size // 2
    dx1 = dy1 = patch_size // 2 + 1
    dx0 = min(ax, bx, dx0)
    dx1 = min(A_size[1] - ax, B_size[1] - bx, dx1)
    dy0 = min(ay, by, dy0)
    dy1 = min(A_size[0] - ay, B_size[0] - by, dy1)

    dist1 = -np.sum(feat_A[ay - dy0:ay + dy1, ax - dx0:ax + dx1] * feat_B[by - dy0:by + dy1, bx - dx0:bx + dx1])
    dist2 = -np.sum(feat_AP[ay - dy0:ay + dy1, ax - dx0:ax + dx1] * feat_BP[by - dy0:by + dy1, bx - dx0:bx + dx1])
    dist = (dist1 + dist2) / ((dx1 + dx0) * (dy1 + dy0))

    return dist


import os 
package_directory = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import cv2

from PIL import Image
def propagate_cuda(nnf, feat_A, feat_AP, feat_B, feat_BP, patch_size, iters=2, rand_search_radius=200):
    mod = SourceModule(open(os.path.join(package_directory,"patchmatch.cu")).read(),no_extern_c=True)
    patchmatch = mod.get_function("patch_match")
    
    rows = feat_A.shape[0]
    cols = feat_A.shape[1]
    channels = np.int32(feat_A.shape[2])
    nnf_t = np.zeros(shape=(rows,cols), dtype=np.uint32)
    nnd = np.random.rand(*nnf.shape[:2]).astype(np.float32)
    threads = 20
    
    def get_blocks_for_dim(dim,blocks):
        #if dim % blocks ==0:
        #    return dim//blocks
        return dim// blocks +1 
    patchmatch(
        drv.In(feat_A),
        drv.In(feat_AP),
        drv.In(feat_B),
        drv.In(feat_BP),
        drv.InOut(nnf),
        drv.InOut(nnf_t),
        drv.InOut(nnd),
        np.int32(rows),
        np.int32(cols),
        channels,
        np.int32(patch_size),
        np.int32(iters),
        np.int32(8),
        np.int32(rand_search_radius),
    block=(threads,threads,1),
    grid=(get_blocks_for_dim(rows,threads),
          get_blocks_for_dim(cols,threads)))

    return nnf, nnd
