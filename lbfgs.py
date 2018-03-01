import torch
import numpy as np

def lbfgs(f, init, maxIter=50, gEps=1e-8, histSize=10, lr=1.0, clamp=False, display=False):
    """
    input:
        f: a function
            in: value x; 1-d tensor
            out: result fx, gradient g
        init: a valid input for f
        maxIter: ---
        gEps: ---
        histSize: ---

    output:
        x: argmin{x} f(x); 1-d tensor
    """

    xk = init
    fk, gk = f(xk)
    H0 = 1.0
    evals = 1
    step = 0
    stat = "LBFGS REACH MAX ITER"
    alpha = list(range(histSize))
    rho = []
    s = []
    y = []

    for it in range(maxIter):
        #print(len(alpha), len(rho), len(s), len(y))
        if display and it%20==0:            
            print("LBFGS | iter:{}; loss:{:.4f}; grad:{:.4f}; step:{:.5f}".format(it, fk, np.sqrt(torch.sum((gk)**2)), step))
        if clamp:
            xk = xk.clamp(0, 1e7)

        xSquaredNorm = torch.sum(xk * xk)
        gSquaredNorm = torch.sum(gk * gk)
        if gSquaredNorm < (gEps**2) * xSquaredNorm:
            stat = "LBFGS BELOW GRADIENT EPS"
            return xk, stat

        z = -gk

        maxIdx = min(it, histSize)

        for i in range(maxIdx):
            alpha[i] = s[i].dot(z) * rho[i]
            z -= alpha[i] * y[i]

        z *= H0

        for i in range(maxIdx-1, -1, -1):
            beta = rho[i] * y[i].dot(z)
            z += s[i] * (alpha[i] - beta)

        fkm1, gkm1 = fk, gk
        
        
        step, stat_ls, args = linesearch(xk.clone(), z, f, fk, gk.clone(), fkm1,gkm1.clone(), 10000, lr)
        if step is None:
            xk, fk, gk = args 
            return xk, stat_ls
        else:
            xk, fk, gk = args 
        """
        step = 1.0
        xk += step * z
        fk, gk = f(xk)
        #if (gk==gkm1).any():
        #    print("error!")
        """

        if it >= histSize:
            s.pop(0)
            y.pop(0)
            rho.pop(0)
        s.append(z * step)
        y.append(gk - gkm1)

        yDots = y[-1].dot(s[-1])
        try:
            rho.append(1.0 / yDots)
        except ZeroDivisionError:
            print(y[-1], s[-1])
            return xk, "Zero division"
        
        yNorm2 = y[-1].dot(y[-1])
        if yNorm2 > 1e-5:
            H0 = yDots / yNorm2

    return xk, stat






def linesearch(xk, z, f, fk, gk, fkm1, gkm1, maxEvals, lr):
    """
    """

    c1 = 1e-4
    c2 = 0.9
    evals = 0

    alpha_0 = 0.0
    phi_0 = fkm1
    phi_prime_0 = z.dot(gk)

    if phi_prime_0 >= 0.0:
        stat = "LINE SEARCH FAILED"
        return None, stat, [xk, fk, gk]

    alpha_max = 1e8

    alpha = lr
    alpha_old = 0.0
    alpha_cor = lr
    second_iter = False

    while True:
        xk += (alpha - alpha_old) * z
        fk, gk = f(xk)
        evals += 1

        phi_alpha = fk
        phi_prime_alpha = z.dot(gk)

        armijo_violated = (phi_alpha > phi_0 + c1 * alpha * phi_prime_0 or (second_iter and phi_alpha >= phi_0))
        strong_wolfe = (np.abs(phi_prime_alpha) <= -c2 * phi_prime_0)

        if (not armijo_violated) and strong_wolfe:
            stat = "LINE SEARCH DONE"
            return alpha, stat, [xk, fk, gk]

        if evals > maxEvals:
            stat = "LINE SEARCH REACH MAX EVALS"
            return None, stat, [xk, fk, gk]

        if armijo_violated or phi_prime_alpha >= 0:
            if armijo_violated:
                alpha_low      = alpha_0
                alpha_high     = alpha
                phi_low        = phi_0
                phi_high       = phi_alpha
                phi_prime_low  = phi_prime_0
                phi_prime_high = phi_prime_alpha

            else:
                alpha_low      = alpha
                alpha_high     = alpha_0
                phi_low        = phi_alpha
                phi_high       = phi_0
                phi_prime_low  = phi_prime_alpha
                phi_prime_high = phi_prime_0
                
                alpha_old = alpha;

            alpha = 0.5 * (alpha_low + alpha_high)
            alpha += (phi_high - phi_low) / (phi_prime_low - phi_prime_high)

            if (alpha < min(alpha_low, alpha_high) or alpha > max(alpha_low, alpha_high)):
                alpha = 0.5 * (alpha_low + alpha_high)

            alpha_cor = alpha - alpha_old
            break            


        alpha_new = alpha + 4 * (alpha - alpha_old)
        alpha_old = alpha
        alpha = alpha_new
        alpha_cor = alpha - alpha_old

        if alpha > alpha_max:
            stat = "LINE SEARCH FAILED"
            return None, stat, [xk, fk, gk]

        second_iter = True

    tries = 0
    minTries = 10

    while True:
        tries += 1        

        """
        alpha_old = alpha
        alpha = 0.5 * (alpha_low + alpha_high)
        try:
            alpha += (phi_high - phi_low) / (phi_prime_low - phi_prime_high)
        except ZeroDivisionError:
            print(alpha, phi_prime_low, phi_prime_high)

        if (alpha < alpha_low and alpha > alpha_high):
            alpha = 0.5 * (alpha_low + alpha_high)
        """

        xk += alpha_cor * z

        fk, gk = f(xk)
        evals += 1

        phi_j = fk
        phi_prime_j = z.dot(gk)

        armijo_violated = (phi_j > phi_0 + c1 * alpha * phi_prime_0 or phi_j >= phi_low)
        strong_wolfe = (np.abs(phi_prime_j) <= -c2 * phi_prime_0);

        if (not armijo_violated) and strong_wolfe:
            stat = "LINE SEARCH DONE"
            return alpha, stat, [xk, fk, gk]

        elif np.abs(alpha_high - alpha_low) < 1e-5 and tries > minTries:
            stat = "LINE SEARCH FAILED"
            return None, stat, [xk, fk, gk]

        elif armijo_violated:
            alpha_high     = alpha
            phi_high       = phi_j
            phi_prime_high = phi_prime_j        

        else:
            if (phi_prime_j * (alpha_high - alpha_low) >= 0):
                alpha_high     = alpha_low
                phi_high       = phi_low
                phi_prime_high = phi_prime_low

            alpha_low     = alpha
            phi_low       = phi_j
            phi_prime_low = phi_prime_j

        alpha = 0.5 * (alpha_low + alpha_high)
        alpha += (phi_high - phi_low) / (phi_prime_low - phi_prime_high)

        if (alpha < min(alpha_low, alpha_high) or alpha > max(alpha_low, alpha_high)):
            alpha = 0.5 * (alpha_low + alpha_high)

        alpha_cor = alpha - alpha_old            

        if evals >= maxEvals:
            stat = "LINE SEARCH REACHED MAX EVALS"
            return None, stat, [xk, fk, gk]



