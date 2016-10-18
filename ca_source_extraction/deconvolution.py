# -*- coding: utf-8 -*-
"""Extract neural activity from a fluorescence trace using a constrained deconvolution approach
Created on Tue Sep  1 16:11:25 2015
@author: Eftychios A. Pnevmatikakis, based on an implementation by T. Machado,  Andrea Giovannucci & Ben Deverett
"""

# -*- coding: utf-8 -*-
# Written by

import numpy as np
import scipy.signal
import scipy.linalg

from warnings import warn
#import time
#import sys
from math import log, sqrt

import sys
#%%


def constrained_foopsi(fluor, bl=None,  c1=None, g=None,  sn=None, p=None, method='cvxpy', bas_nonneg=True,
                       noise_range=[.25, .5], noise_method='logmexp', lags=5, fudge_factor=1.,
                       verbosity=False, solvers=None, **kwargs):
    """ Infer the most likely discretized spike train underlying a fluorescence trace 

    It relies on a noise constrained deconvolution approach


    Parameters
    ----------
    fluor: np.ndarray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    bl: [optional] float
        Fluorescence baseline value. If no value is given, then bl is estimated
        from the data.
    c1: [optional] float
        value of calcium at time 0
    g: [optional] list,float
        Parameters of the AR process that models the fluorescence impulse response.
        Estimated from the data if no value is given
    sn: float, optional
        Standard deviation of the noise distribution.  If no value is given,
        then sn is estimated from the data.
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for basis projection pursuit 'cvx' or 'cvxpy'
    bas_nonneg: bool
        baseline strictly non-negative
    noise_range:  list of two elms
        frequency range for averaging noise PSD
    noise_method: string
        method of averaging noise PSD
    lags: int
        number of lags for estimating time constants
    fudge_factor: float
        fudge factor for reducing time constant bias
    verbosity: bool     
         display optimization details
    solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers to be used with cvxpy, default is ['ECOS','SCS']
    Returns
    -------
    c: np.ndarray float
        The inferred denoised fluorescence signal at each time-bin.
    bl, c1, g, sn : As explained above
    sp: ndarray of float
        Discretized deconvolved neural activity (spikes)

    References
    ----------
    * Pnevmatikakis et al. 2016. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037
    * Machado et al. 2015. Cell 162(2):338-350
    """

    if p is None:
        raise Exception("You must specify the value of p")

    if g is None or sn is None:
        g, sn = estimate_parameters(fluor, p=p, sn=sn, g=g, range_ff=noise_range,
                                    method=noise_method, lags=lags, fudge_factor=fudge_factor)
    if p == 0:
        c1 = 0
        g = np.array(0)
        bl = 0
        c = np.maximum(fluor, 0)
        sp = c.copy()
    else:
        if method == 'cvx':
            c, bl, c1, g, sn, sp = cvxopt_foopsi(
                fluor, b=bl, c1=c1, g=g, sn=sn, p=p, bas_nonneg=bas_nonneg, verbosity=verbosity)

        elif method == 'cvxpy':

            c, bl, c1, g, sn, sp = cvxpy_foopsi(
                fluor,  g, sn, b=bl, c1=c1, bas_nonneg=bas_nonneg, solvers=solvers)

        elif method == 'oasis':
            from oasis import constrained_oasisAR1
            if p == 1:
                if bl is None:
                    c, sp, bl, g, _ = constrained_oasisAR1(fluor, g[0], sn, optimize_b=True, penalty=1)
                else:
                    c, sp, _, g, _ = constrained_oasisAR1(fluor - bl, g[0], sn, optimize_b=False, penalty=1)
                c1 = c[0]
                # remove intial calcium to align with the other foopsi methods
                # it is added back in function constrained_foopsi_parallel of temporal.py
                c -= c1 * g**np.arange(len(fluor))
            elif p == 2:
                if bl is None:
                    c, sp, bl, g, _ = constrained_oasisAR2(fluor, g, sn, optimize_b=True, penalty=1)
                else:
                    c, sp, _, g, _ = constrained_oasisAR2(fluor - bl, g, sn, optimize_b=False, penalty=1)
                c1 = c[0]
                d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
                c -= c1 * d**np.arange(len(fluor))
            else:
                raise Exception('OASIS is currently only implemented for p=1 and p=2')
            g = np.ravel(g)

        else:
            raise Exception('Undefined Deconvolution Method')

    return c, bl, c1, g, sn, sp


def G_inv_mat(x, mode, NT, gs, gd_vec, bas_flag=True, c1_flag=True):
    """
    Fast computation of G^{-1}*x and G^{-T}*x
    """
    from scipy.signal import lfilter
    if mode == 1:
        b = lfilter(np.array([1]), np.concatenate([np.array([1.]), -gs]), x[:NT]
                    ) + bas_flag * x[NT - 1 + bas_flag] + c1_flag * gd_vec * x[-1]
        #b = np.dot(Emat,b)
    elif mode == 2:
        #x = np.dot(Emat.T,x)
        b = np.hstack((np.flipud(lfilter(np.array([1]), np.concatenate([np.array(
            [1.]), -gs]), np.flipud(x))), np.ones(bas_flag) * np.sum(x), np.ones(c1_flag) * np.sum(gd_vec * x)))

    return b


def cvxopt_foopsi(fluor, b, c1, g, sn, p, bas_nonneg, verbosity):
    """Solve the deconvolution problem using cvxopt and picos packages
    """
    try:
        from cvxopt import matrix, spmatrix, spdiag, solvers
        import picos
    except ImportError:
        raise ImportError('Constrained Foopsi requires cvxopt and picos packages.')

    T = len(fluor)

    # construct deconvolution matrix  (sp = G*c)
    G = spmatrix(1., range(T), range(T), (T, T))

    for i in range(p):
        G = G + spmatrix(-g[i], np.arange(i + 1, T), np.arange(T - i - 1), (T, T))

    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    gen_vec = G * matrix(np.ones(fluor.size))

    # Initialize variables in our problem
    prob = picos.Problem()

    # Define variables
    calcium_fit = prob.add_variable('calcium_fit', fluor.size)
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b = prob.add_variable('b', 1)
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)

        prob.add_constraint(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 = prob.add_variable('c1', 1)
        prob.add_constraint(c1 >= 0)
    else:
        flag_c1 = False

    # Add constraints
    prob.add_constraint(G * calcium_fit >= 0)
    res = abs(matrix(fluor.astype(float)) - calcium_fit - b *
              matrix(np.ones(fluor.size)) - matrix(gd_vec) * c1)
    prob.add_constraint(res < sn * np.sqrt(fluor.size))
    prob.set_objective('min', calcium_fit.T * gen_vec)

    # solve problem
    try:
        prob.solve(solver='mosek', verbose=verbosity)
        sel_solver = 'mosek'
#        prob.solve(solver='gurobi', verbose=verbosity)
#        sel_solver = 'gurobi'
    except ImportError:
        warn('MOSEK is not installed. Spike inference may be VERY slow!')
        sel_solver = []
        prob.solver_selection()
        prob.solve(verbose=verbosity)

    # if problem in infeasible due to low noise value then project onto the
    # cone of linear constraints with cvxopt
    if prob.status == 'prim_infeas_cer' or prob.status == 'dual_infeas_cer' or prob.status == 'primal infeasible':
        warn('Original problem infeasible. Adjusting noise level and re-solving')
        # setup quadratic problem with cvxopt
        solvers.options['show_progress'] = verbosity
        ind_rows = range(T)
        ind_cols = range(T)
        vals = np.ones(T)
        if flag_b:
            ind_rows = ind_rows + range(T)
            ind_cols = ind_cols + [T] * T
            vals = np.concatenate((vals, np.ones(T)))
        if flag_c1:
            ind_rows = ind_rows + range(T)
            ind_cols = ind_cols + [T + cnt - 1] * T
            vals = np.concatenate((vals, np.squeeze(gd_vec)))
        P = spmatrix(vals, ind_rows, ind_cols, (T, T + cnt))
        H = P.T * P
        Py = P.T * matrix(fluor.astype(float))
        sol = solvers.qp(
            H, -Py, spdiag([-G, -spmatrix(1., range(cnt), range(cnt))]), matrix(0., (T + cnt, 1)))
        xx = sol['x']
        c = np.array(xx[:T])
        sp = np.array(G * matrix(c))
        c = np.squeeze(c)
        if flag_b:
            b = np.array(xx[T + 1]) + b_lb
        if flag_c1:
            c1 = np.array(xx[-1])
        sn = np.linalg.norm(fluor - c - c1 * gd_vec - b) / np.sqrt(T)
    else:  # readout picos solution
        c = np.squeeze(calcium_fit.value)
        sp = np.squeeze(np.asarray(G * calcium_fit.value))
        if flag_b:
            b = np.squeeze(b.value)
        if flag_c1:
            c1 = np.squeeze(c1.value)

    return c, b, c1, g, sn, sp


def cvxpy_foopsi(fluor,  g, sn, b=None, c1=None, bas_nonneg=True, solvers=None):
    '''Solves the deconvolution problem using the cvxpy package and the ECOS/SCS library. 
    Parameters:
    -----------
    fluor: ndarray
        fluorescence trace 
    g: list of doubles
        parameters of the autoregressive model, cardinality equivalent to p        
    sn: double
        estimated noise level
    b: double
        baseline level. If None it is estimated. 
    c1: double
        initial value of calcium. If None it is estimated.  
    bas_nonneg: boolean
        should the baseline be estimated        
    solvers: tuple of two strings
        primary and secondary solvers to be used. Can be choosen between ECOS, SCS, CVXOPT    

    Returns:
    --------
    c: estimated calcium trace
    b: estimated baseline
    c1: esimtated initial calcium value
    g: esitmated parameters of the autoregressive model
    sn: estimated noise level
    sp: estimated spikes 

    '''
    try:
        import cvxpy as cvx
    except ImportError:
        raise ImportError('cvxpy solver requires installation of cvxpy.')
    if solvers is None:
        solvers = ['ECOS', 'SCS']

    T = fluor.size

    # construct deconvolution matrix  (sp = G*c)
    G = scipy.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))

    for i, gi in enumerate(g):
        G = G + scipy.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))

    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    gen_vec = G.dot(scipy.sparse.coo_matrix(np.ones((T, 1))))

    c = cvx.Variable(T)  # calcium at each time step
    constraints = []
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b = cvx.Variable(1)  # baseline value
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)
        constraints.append(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 = cvx.Variable(1)  # baseline value
        constraints.append(c1 >= 0)
    else:
        flag_c1 = False

    thrNoise = sn * np.sqrt(fluor.size)

    try:
        objective = cvx.Minimize(cvx.norm(G * c, 1))  # minimize number of spikes
        constraints.append(G * c >= 0)
        constraints.append(cvx.norm(-c + fluor - b - gd_vec * c1, 2) <= thrNoise)  # constraints
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver=solvers[0])

        if not (prob.status == 'optimal' or prob.status == 'optimal_inaccurate'):
            raise ValueError('Problem solved suboptimally or unfeasible')

        print 'PROBLEM STATUS:' + prob.status
        sys.stdout.flush()
    except (ValueError, cvx.SolverError) as err:     # if solvers fail to solve the problem
        #         print(err)
        #         sys.stdout.flush()
        lam = sn / 500
        constraints = constraints[:-1]
        objective = cvx.Minimize(cvx.norm(-c + fluor - b - gd_vec *
                                          c1, 2) + lam * cvx.norm(G * c, 1))
        prob = cvx.Problem(objective, constraints)
        try:  # in case scs was not installed properly
            try:
                print('TRYING AGAIN ECOS')
                sys.stdout.flush()
                result = prob.solve(solver=solvers[0])
            except:
                print(solvers[0] + ' DID NOT WORK TRYING ' + solvers[1])
                result = prob.solve(solver=solvers[1])
        except:
            sys.stderr.write(
                '***** SCS solver failed, try installing and compiling SCS for much faster performance. Otherwise set the solvers in tempora_params to ["ECOS","CVXOPT"]')
            sys.stderr.flush()
            #result = prob.solve(solver='CVXOPT')
            raise

        if not (prob.status == 'optimal' or prob.status == 'optimal_inaccurate'):
            print 'PROBLEM STATUS:' + prob.status
            sp = fluor
            c = fluor
            b = 0
            c1 = 0
            return c, b, c1, g, sn, sp
            #raise Exception('Problem could not be solved')

    sp = np.squeeze(np.asarray(G * c.value))
    c = np.squeeze(np.asarray(c.value))
    if flag_b:
        b = np.squeeze(b.value)
    if flag_c1:
        c1 = np.squeeze(c1.value)

    return c, b, c1, g, sn, sp


def _nnls(KK, Ky, s=None, mask=None, tol=1e-9, max_iter=None):
    """
    Solve non-negative least squares problem
    ``argmin_s || Ks - y ||_2`` for ``s>=0``

    Parameters
    ----------
    KK : array, shape (n, n)
        Dot-product of design matrix K transposed and K, K'K
    Ky : array, shape (n,)
        Dot-product of design matrix K transposed and target vector y, K'y
    s : None or array, shape (n,), optional, default None
        Initialization of deconvolved neural activity.
    mask : array of bool, shape (n,), optional, default (True,)*n
        Mask to restrict potential spike times considered.
    tol : float, optional, default 1e-9
        Tolerance parameter.
    max_iter : None or int, optional, default None
        Maximum number of iterations before termination.
        If None (default), it is set to len(KK).

    Returns
    -------
    s : array, shape (n,)
        Discretized deconvolved neural activity (spikes)

    References
    ----------
    * Lawson C and Hanson RJ, SIAM 1987
    * Bro R and DeJong S, J Chemometrics 1997
    """

    if mask is None:
        mask = np.ones(len(KK), dtype=bool)
    else:
        KK = KK[mask][:, mask]
        Ky = Ky[mask]
    if s is None:
        s = np.zeros(len(KK))
        l = Ky.copy()
        P = np.zeros(len(KK), dtype=bool)
    else:
        s = s[mask]
        P = s > 0
        l = Ky - KK[:, P].dot(s[P])
    i = 0
    if max_iter is None:
        max_iter = len(KK)
    for i in range(max_iter):  # max(l) is checked at the end, should do at least one iteration
        w = np.argmax(l)
        P[w] = True
        try:  # likely unnnecessary try-except-clause for robustness sake
            mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
        except:
            mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
            print r'added $\epsilon$I to avoid singularity'
        while len(mu > 0) and min(mu) < 0:
            a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
            s[P] += a * (mu - s[P])
            P[s <= tol] = False
            try:
                mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
            except:
                mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
                print r'added $\epsilon$I to avoid singularity'
        s[P] = mu.copy()
        l = Ky - KK[:, P].dot(s[P])
        if max(l) < tol:
            break
    tmp = np.zeros(len(mask))
    tmp[mask] = s
    return tmp


def onnls(y, g, lam=0, shift=100, window=200, mask=None, tol=1e-9, max_iter=None):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    ``argmin_s 1/2|Ks-y|^2 + lam |s|_1`` for ``s>=0``

    Parameters
    ----------
    y : array of float, shape (T,)
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    g : array, shape (p,)
        if p in (1,2):
            Parameter(s) of the AR(p) process that models the fluorescence impulse response.
        else:
            Kernel that models the fluorescence impulse response.
    lam : float, optional, default 0
        Sparsity penalty parameter lambda.
    shift : int, optional, default 100
        Number of frames by which to shift window from on run of NNLS to the next.
    window : int, optional, default 200
        Window size.
    mask : array of bool, shape (n,), optional, default (True,)*n
        Mask to restrict potential spike times considered.
    tol : float, optional, default 1e-9
        Tolerance parameter.
    max_iter : None or int, optional, default None
        Maximum number of iterations before termination.
        If None (default), it is set to window size.

    Returns
    -------
    c : array of float, shape (T,)
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float, shape (T,)
        Discretized deconvolved neural activity (spikes).

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Bro R and DeJong S, J Chemometrics 1997
    """

    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)
    w = window
    K = np.zeros((w, w))
    if len(g) == 1:  # kernel for AR(1)
        _y = y - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        h = np.exp(log(g[0]) * np.arange(w))
        for i in range(w):
            K[i:, i] = h[:w - i]
    elif len(g) == 2:  # kernel for AR(2)
        _y = y - lam * (1 - g[0] - g[1])
        _y[-2] = y[-2] - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
        r = (g[0] - sqrt(g[0] * g[0] + 4 * g[1])) / 2
        h = (np.exp(log(d) * np.arange(1, w + 1)) - np.exp(log(r) * np.arange(1, w + 1))) / (d - r)
        for i in range(w):
            K[i:, i] = h[:w - i]
    else:  # arbitrary kernel
        h = g
        for i in range(w):
            K[i:, i] = h[:w - i]
        a = np.linalg.inv(K).sum(0)
        _y = y - lam * a[0]
        _y[-w:] = y[-w:] - lam * a

    s = np.zeros(T)
    KK = K.T.dot(K)
    for i in range(0, T - w, shift):
        s[i:i + w] = _nnls(KK, K.T.dot(_y[i:i + w]), s[i:i + w], mask=mask[i:i + w],
                           tol=tol, max_iter=max_iter)[:w]
        # subtract contribution of spikes already committed to
        _y[i:i + w] -= K[:, :shift].dot(s[i:i + shift])
    s[i + shift:] = _nnls(KK[-(T - i - shift):, -(T - i - shift):],
                          K[:T - i - shift, :T - i - shift].T.dot(_y[i + shift:]),
                          s[i + shift:], mask=mask[i + shift:])
    c = np.zeros_like(s)
    for t in np.where(s > tol)[0]:
        c[t:t + w] += s[t] * h[:min(w, T - t)]
    return c, s


def constrained_oasisAR2(y, g, sn, optimize_b=True, optimize_g=0, decimate=5, shift=100, window=200,
                         tol=1e-9, max_iter=1, penalty=1):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g1 c_{t-1}-g2 c_{t-2} >= 0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities (with baseline
        already subtracted) with one entry per time-bin.
    g : (float, float)
        Parameters of the AR(2) process that models the fluorescence impulse response.
    sn : float
        Standard deviation of the noise distribution.
    optimize_b : bool, optional, default True
        Optimize baseline if True else it is set to 0, see y.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        No optimization if optimize_g=0.
    decimate : int, optional, default 5
        Decimation factor for estimating hyper-parameters faster on decimated data.
    max_iter : int, optional, default 1
        Maximal number of iterations.
    penalty : int, optional, default 1
        Sparsity penalty. 1: min |s|_1  0: min |s|_0

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes).
    b : float
        Fluorescence baseline value.
    (g1, g2) : tuple of float
        Parameters of the AR(2) process that models the fluorescence impulse response.
    lam : float
        Sparsity penalty parameter lambda of dual problem.

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """

    T = len(y)
    d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
    r = (g[0] - sqrt(g[0] * g[0] + 4 * g[1])) / 2
    if not optimize_g:
        g11 = (np.exp(log(d) * np.arange(1, T + 1)) -
               np.exp(log(r) * np.arange(1, T + 1))) / (d - r)
        g12 = np.append(0, g[1] * g11[:-1])
        g11g11 = np.cumsum(g11 * g11)
        g11g12 = np.cumsum(g11 * g12)
        Sg11 = np.cumsum(g11)
        f_lam = 1 - g[0] - g[1]
    thresh = sn * sn * T
    # get initial estimate of b and lam on downsampled data using AR1 model
    if decimate > 0:
        from oasis import constrained_oasisAR1
        _, s, b, aa, lam = constrained_oasisAR1(y.reshape(-1, decimate).mean(1),
                                                d**decimate, sn / sqrt(decimate),
                                                optimize_b=optimize_b, optimize_g=optimize_g)
        if optimize_g > 0:
            d = aa**(1. / decimate)
            g[0] = d + r
            g[1] = -d * r
            g11 = (np.exp(log(d) * np.arange(1, T + 1)) -
                   np.exp(log(r) * np.arange(1, T + 1))) / (d - r)
            g12 = np.append(0, g[1] * g11[:-1])
            g11g11 = np.cumsum(g11 * g11)
            g11g12 = np.cumsum(g11 * g12)
            Sg11 = np.cumsum(g11)
            f_lam = 1 - g[0] - g[1]
        lam *= (1 - d**decimate) / f_lam
        ff = np.hstack([a * decimate + np.arange(-decimate, decimate)
                        for a in np.where(s > 1e-6)[0]])  # this window size seems necessary and sufficient
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        mask = np.zeros(T, dtype=bool)
        mask[ff] = True
    else:
        b = np.percentile(y, 15) if optimize_b else 0
        lam = 2 * sn * np.linalg.norm(g11)
        mask = None
    # run ONNLS
    c, s = onnls(y - b, g, lam=lam, mask=mask)

    if not optimize_b:  # don't optimize b, just the dual variable lambda
        for i in range(max_iter - 1):
            res = y - c
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4:
                break
            # calc shift dlam, here attributed to sparsity penalty
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f - 1
                # if and elif correct last 2 time points for |s|_1 instead |c|_1
                if i == len(ls) - 2:  # last pool
                    tmp[f] = (1. / f_lam if l == 0 else
                              (Sg11[l] + g[1] / f_lam * g11[l - 1]
                               + (g[0] + g[1]) / f_lam * g11[l]
                               - g11g12[l] * tmp[f - 1]) / g11g11[l])
                # secondlast pool if last one has length 1
                elif i == len(ls) - 3 and ls[-2] == T - 1:
                    tmp[f] = (Sg11[l] + g[1] / f_lam * g11[l]
                              - g11g12[l] * tmp[f - 1]) / g11g11[l]
                else:  # all other pools
                    tmp[f] = (Sg11[l] - g11g12[l] * tmp[f - 1]) / g11g11[l]
                l += 1
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]

            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except:
                db = -bb / aa
            # perform shift
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask)
            db = np.mean(y - c) - b
            b += db
            lam -= db / f_lam

    else:  # optimize b
        db = np.mean(y - c) - b
        b += db
        lam -= db / (1 - g[0] - g[1])
        for i in range(max_iter - 1):
            res = y - c - b
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4:
                break
            # calc shift db, here attributed to baseline
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f
                tmp[f] = (Sg11[l - 1] - g11g12[l - 1] * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            tmp -= tmp.mean()
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except:
                db = -bb / aa
            # perform shift
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask)
            db = np.mean(y - c) - b
            b += db
            lam -= db / f_lam

    if penalty == 0:  # get (locally optimal) L0 solution

        def c4smin(y, s, s_min):
            ls = np.append(np.where(s > s_min)[0], T)
            tmp = np.zeros_like(s)
            l = ls[0]  # first pool
            tmp[:l] = max(0, np.exp(log(d) * np.arange(l)).dot(y[:l]) * (1 - d * d)
                          / (1 - d**(2 * l))) * np.exp(log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f
                tmp[f] = (g11[:l].dot(y[f:f + l])
                          - g11g12[l - 1] * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            return tmp
        spikesizes = np.sort(s[s > 1e-6])
        i = len(spikesizes) / 2
        l = 0
        u = len(spikesizes) - 1
        while u - l > 1:
            s_min = spikesizes[i]
            tmp = c4smin(y - b, s, s_min)
            res = y - b - tmp
            RSS = res.dot(res)
            if RSS < thresh or i == 0:
                l = i
                i = (l + u) / 2
                res0 = tmp
            else:
                u = i
                i = (l + u) / 2
        if i > 0:
            c = res0
            s = np.append([0, 0], c[2:] - g[0] * c[1:-1] - g[1] * c[:-2])

    return c, s, b, g, lam


def estimate_parameters(fluor, p=2, sn=None, g=None, range_ff=[0.25, 0.5], method='logmexp', lags=5, fudge_factor=1):
    """
    Estimate noise standard deviation and AR coefficients if they are not present
    p: positive integer
        order of AR system
    sn: float
        noise standard deviation, estimated if not provided.
    lags: positive integer
        number of additional lags where he autocovariance is computed
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method: string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)
    fudge_factor: float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    """

    if sn is None:
        sn = GetSn(fluor, range_ff, method)

    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor, p, sn, lags, fudge_factor)

    return g, sn


def estimate_time_constant(fluor, p=2, sn=None, lags=5, fudge_factor=1.):
    """    
    Estimate AR model parameters through the autocovariance function
    Inputs
    ----------
    fluor        : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    p            : positive integer
        order of AR system
    sn           : float
        noise standard deviation, estimated if not provided.
    lags         : positive integer
        number of additional lags where he autocovariance is computed
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias

    Return
    -----------
    g       : estimated coefficients of the AR process
    """

    if sn is None:
        sn = GetSn(fluor)

    lags += p
    xc = axcov(fluor, lags)
    xc = xc[:, np.newaxis]

    A = scipy.linalg.toeplitz(xc[lags + np.arange(lags)], xc[lags +
                                                             np.arange(p)]) - sn**2 * np.eye(lags, p)
    # print A, xc
    g = np.linalg.lstsq(A, xc[lags + 1:])[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = (gr + gr.conjugate()) / 2.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    """    
    Estimate noise power through the power spectral density over the range of large frequencies    
    Inputs
    ----------
    fluor    : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged  
    method   : string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)

    Return
    -----------
    sn       : noise standard deviation
    """

    ff, Pxx = scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind / 2)),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx_ind / 2)),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx_ind / 2))))
    }[method](Pxx_ind)

    return sn


def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag
    Parameters
    ----------
    data : array
        Array containing fluorescence data
    maxlag : int
        Number of lags to use in autocovariance calculation
    Returns
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    #xcov = xcov/np.concatenate([np.arange(T-maxlag,T+1),np.arange(T-1,T-maxlag-1,-1)])
    return np.real(xcov / T)


def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).
    Parameters
    ----------
    value : int
    Returns
    -------
    exponent : int
    """

    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent
