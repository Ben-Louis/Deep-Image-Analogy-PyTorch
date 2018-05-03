import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Pool


def init_nnf(size, B_size=None):
    nnf = np.zeros(shape=(2, size[0], size[1])).astype(np.int)
    nnf[0] = np.array([np.arange(size[0])] * size[1]).T
    nnf[1] = np.array([np.arange(size[1])] * size[0])
    nnf = nnf.transpose((1, 2, 0))
    if B_size is not None:
        nnf[:, :, 0] = nnf[:, :, 0] * (B_size[0] / size[0])
        nnf[:, :, 1] = nnf[:, :, 1] * (B_size[1] / size[1])
        nnf = np.array(nnf, dtype=np.int)

    return nnf


def upSample_nnf(nnf, size=None):
    ah, aw = nnf.shape[:2]

    if size is None:
        size = [ah * 2, aw * 2]

    bh, bw = size
    ratio_h, ratio_w = bh / ah, bw / aw
    target = np.zeros(shape=(size[0], size[1], 2)).astype(np.int)

    for by in range(bh):
        for bx in range(bw):
            quot_h, quot_w = int(by // ratio_h), int(bx // ratio_w)
            # print(quot_h, quot_w)
            rem_h, rem_w = (by - quot_h * ratio_h), (bx - quot_w * ratio_w)
            vy, vx = nnf[quot_h, quot_w]
            vy = int(ratio_h * vy + rem_h)
            vx = int(ratio_w * vx + rem_w)
            target[by, bx] = [vy, vx]

    return target


def avg_vote(nnf, img, patch_size, A_size, B_size):
    assert img.shape[0] == B_size[0] and img.shape[1] == B_size[1], "[{},{}], [{},{}]".format(img.shape[0],
                                                                                              img.shape[1], B_size[0],
                                                                                              B_size[1])
    final = np.zeros(list(A_size) + [img.shape[2], ])

    ah, aw = A_size
    bh, bw = B_size
    for ay in range(A_size[0]):
        for ax in range(A_size[1]):

            count = 0
            for dy in range(-(patch_size // 2), (patch_size // 2 + 1)):
                for dx in range(-(patch_size // 2), (patch_size // 2 + 1)):

                    if ((ax + dx) < aw and (ax + dx) >= 0 and (ay + dy) < ah and (ay + dy) >= 0):
                        by, bx = nnf[ay + dy, ax + dx]

                        if ((bx - dx) < bw and (bx - dx) >= 0 and (by - dy) < bh and (by - dy) >= 0):
                            count += 1
                            final[ay, ax, :] += img[by - dy, bx - dx, :]

            if count > 0:
                final[ay, ax] /= count

    return final


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
    cpus = min(mp.cpu_count(), A_size[0] // 20 + 1)
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

    try:
        if feat_A.shape[2] == 3:
            dist1 = np.sum(
                (feat_A[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - feat_B[by - dy0:by + dy1, bx - dx0:bx + dx1]) ** 2)
            dist2 = np.sum(
                (feat_AP[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - feat_BP[by - dy0:by + dy1, bx - dx0:bx + dx1]) ** 2)
        else:
            dist1 = -np.sum(feat_A[ay - dy0:ay + dy1, ax - dx0:ax + dx1] * feat_B[by - dy0:by + dy1, bx - dx0:bx + dx1])
            dist2 = -np.sum(
                feat_AP[ay - dy0:ay + dy1, ax - dx0:ax + dx1] * feat_BP[by - dy0:by + dy1, bx - dx0:bx + dx1])
        dist = (dist1 + dist2) / ((dx1 + dx0) * (dy1 + dy0))
        # dist = clamp(dist, -np.inf, cutoff)
    except Exception as e:
        print(e)
        print("dx0:{}; dx1:{}; dy0:{}; dy1:{}; ax:{}; ay:{}; bx:{}; by:{}".format(dx0, dx1, dy0, dy1, ax, ay, bx, by))
    return dist


def clamp(arr, low, high):
    arr = arr.reshape([1] + arr.shape)
    low = np.ones(arr.shape) * low
    high = np.ones(arr.shape) * high
    arr = np.max(np.concatenate([arr, low], axis=0), axis=0)
    arr = arr.reshape([1] + arr.shape)
    arr = np.min(np.concatenate([arr, high], axis=0), axis=0)

    return arr


def reconstruct_avg(nnf, img, patch_size, A_size, B_size):
    assert img.shape[0] == B_size[0] and img.shape[1] == B_size[1], "[{},{}], [{},{}]".format(img.shape[0],
                                                                                              img.shape[1], B_size[0],
                                                                                              B_size[1])
    final = np.zeros(list(A_size) + [3, ])
    # ratio = min(A_size[0]/nnf.shape[0], img.shape[1]/nnf.shape[1])
    # print("ratio:" + str(ratio))

    ah, aw = A_size
    bh, bw = B_size
    for ay in range(A_size[0]):
        for ax in range(A_size[1]):

            count = 0
            for dy in range(-(patch_size // 2), (patch_size // 2 + 1)):
                for dx in range(-(patch_size // 2), (patch_size // 2 + 1)):

                    if ((ax + dx) < aw and (ax + dx) >= 0 and (ay + dy) < ah and (ay + dy) >= 0):
                        by, bx = nnf[ay + dy, ax + dx]

                        if ((bx - dx) < bw and (bx - dx) >= 0 and (by - dy) < bh and (by - dy) >= 0):
                            count += 1
                            final[ay, ax, :] += img[by - dy, bx - dx, :]

            if count > 0:
                final[ay, ax] /= count

    return final

