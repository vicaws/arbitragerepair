"""
Construct static arbitrage constraints for a given finite collection of call
option prices.
"""

# Copyright 2018-2020 Sheng Wang.
# Affiliation: Mathematical Institute, University of Oxford
# Email: sheng.wang@maths.ox.ac.uk

import numpy as _np
import pandas as _pd

#_np.seterr(divide='ignore', invalid='ignore')  # these warnings do not matter


class Normalise:
    """
    Normalises and pre-processes data:
    (1) Normalise K, C by F (C must be undiscounted call prices).
    (2) Remove NAs and prices that are very close to zero (smaller than the
    specified min_price).
    (3) Order the array of prices from short expiry to long expiry, and from
    small strike to large strike per expiry.

    """

    def __init__(self, min_price=None):
        """
        Constructor.

        Parameters
        ----------
        min_price: optional, float
            The minimal price of call option that is acceptable. A very low but
            non-negative price will not stop the repair method working, but may
            not make financial sense to be included in any study as it is deep
            OTM.

        """
        self._min_price = min_price
        self._order_mask = None
        self._F = None

    def fit(self, T, K, C, F):
        """
        Fits input data which are used as benchmark for the normalisation.

        Parameters
        ----------
        T: array_like, 1D
            The 1D array of time-to-expiries.

        K: array_like, 1D
            The 1D array of strike prices.

        C: array_like, 1D
            The 1D array of undiscounted call prices.

        F: array_like, 1D
            The 1D array of forward prices.

        """
        try:
            df = _pd.DataFrame(data={'T': T, 'K': K, 'C': C, 'F': F})
        except ValueError:
            raise ValueError("Please ensure legal input arguments! All T, K, "
                             "C, F should be 1D numeric array.")

        df.sort_values(by=['T', 'K'], inplace=True)
        order_mask = df.index.values
        C1 = df['C'].values
        F1 = df['F'].values

        # remove contracts with NA price
        if _np.any(_np.isnan(C1)):
            mask = ~(_np.isnan(C1))
            order_mask = order_mask[mask]

        # remove contracts whose price is lower than minimum allowed price
        if self._min_price is not None:
            mask = ~(C1 < self._min_price)
            order_mask = order_mask[mask]

        self._order_mask = order_mask
        self._F = F1[order_mask]

    def transform(self, T, K, C):
        """
        Normalises given data.

        Parameters
        ----------
        T: 1D numpy.ndarray
            The 1D array of time-to-expiries.

        K: 1D numpy.ndarray
            The 1D array of strike prices.

        C: 1D numpy.ndarray
            The 1D array of undiscounted call prices.

        Returns
        -------
        T1: 1D numpy.ndarray
            The 1D array of time-to-expiries.

        K1: 1D numpy.ndarray
            The 1D array of normalised strike prices.

        C1: 1D numpy.ndarray
            The 1D array of normalised call prices.

        """
        try:
            T1 = T[self._order_mask]
            K1 = K[self._order_mask] / self._F
            C1 = C[self._order_mask] / self._F
        except TypeError:
            raise TypeError("Please ensure legal input arguments! All T, K, C "
                            "should be 1D numpy.ndarray.")
        return T1, K1, C1

    def inverse_transform(self, K, C):
        """
        De-normalises data. A reverse process of data normalisation.

        Parameters
        ----------
        K: array_like, 1D
            The 1D array of normalised strike prices.

        C: array_like, 1D
            The 1D array of normalised call prices.

        Returns
        -------
        K0: array_like, 1D
            The 1D array of denormalised strike prices.

        C0: array_like, 1D
            The 1D array of denormalised call prices.

        """
        try:
            K0 = K * self._F
            C0 = C * self._F
        except TypeError:
            raise TypeError("Please ensure legal input arguments! All K, C "
                            "should be 1D numpy.ndarray.")
        return K0, C0


def normalise(T, K, C, F, min_price=None):
    """
    Normalises and pre-processes data:
    (1) Normalise K, C by F (C must be undiscounted call prices).
    (2) Remove NAs and prices that are very close to zero (smaller than the
    specified min_price).
    (3) Order the array of prices from short expiry to long expiry, and from
    small strike to large strike per expiry.

    Parameters
    ----------
    T: array_like, 1D
        The 1D array of time-to-expiries.

    K: array_like, 1D
        The 1D array of strike prices.

    C: array_like, 1D
        The 1D array of undiscounted call prices.

    F: array_like, 1D
        The 1D array of forward prices.

    min_price: optional, float
        The minimal price of call option that is acceptable. A very low but
        non-negative price will not stop the repair method working, but may not
        make financial sense to be included in any study as it is deep OTM.

    Returns
    -------
    T1: array_like, 1D
        The 1D array of time-to-expiries.

    K1: array_like, 1D
        The 1D array of normalised strike prices.

    C1: array_like, 1D
        The 1D array of normalised call prices.

    F1: array_like, 1D
        The 1D array of forward prices.

    """

    try:
        df = _pd.DataFrame(data={'T': T, 'K': K, 'C': C, 'F': F})
    except ValueError:
        raise ValueError("Please ensure legal input arguments! All T, K, C, F "
                         "should be 1D numeric array.")

    df.sort_values(by=['T', 'K'], inplace=True)
    T1 = df['T'].values
    K1 = df['K'].values
    C1 = df['C'].values
    F1 = df['F'].values

    # normalise
    K1 /= F1
    C1 /= F1

    # remove contracts with NA price
    if _np.any(_np.isnan(C1)):
        mask = ~(_np.isnan(C1))
        T1, K1, C1, F1 = T1[mask], K1[mask], C1[mask], F1[mask]

    # remove contracts whose price is lower than minimum allowed price
    if min_price is not None:
        mask = ~(C1 < min_price)
        T1, K1, C1, F1 = T1[mask], K1[mask], C1[mask], F1[mask]

    return T1, K1, C1, F1


def denormalise(K, C, F):
    """
    De-normalises data. A reverse process of data normalisation.

    Parameters
    ----------
    K: array_like, 1D
        The 1D array of normalised strike prices.

    C: array_like, 1D
        The 1D array of normalised call prices.

    F: array_like, 1D
        The 1D array of forward prices.

    Returns
    -------
    K0: array_like, 1D
        The 1D array of denormalised strike prices.

    C0: array_like, 1D
        The 1D array of denormalised call prices.

    """
    K0 = K * F
    C0 = C * F

    return K0, C0


def detect(T, K, C, tolerance=0.0, verbose=False):
    """
    Constructs the coefficient matrix A and the vector of constant terms b for
    all static arbitrage constraints.

    Detects violations of the constructed static arbitrage constraints, i.e.
    identify rows of A where the following inequality system fails to hold:

    A * c >= b

    Parameters
    ----------
    T: array_like, 1D
        The 1D array of time-to-expiries.

    K: array_like, 1D
        The 1D array of normalised strike prices.

    C: array_like, 1D
        The 1D array of normalised call prices.

    tolerance: float
        A non-negative value indicating tolerance of constraint violation.

    verbose: boolean
        verbose=True will lead to printing of a summary of the number of
        arbitrage constraints of the six categories and how many of them are
        violated.

    Returns
    -------
    mat_A: numpy.array, 2D, shape = (R, n_quote)
        Coefficient matrix. R = n_cond is the number of constraints.

    vec_b: numpy.array, 1D, shape = (R, )
        Vector of constant terms. R = n_cond is the number of constraints.

    n_cond: list of int
        List of the numbers of static arbitrage constraints of different
        categories.

    n_breach: list of int
        List of the numbers of violated static arbitrage constraints of
        different categories.

    """
    df = _prepare_dataframe(T, K, C)
    n_quote = T.shape[0]
    n_expiry = _np.unique(T).shape[0]

    # construct different categories of constraints
    mat_A1, vec_b1 = _constrain_outright(df, n_quote, n_expiry)
    mat_A2, vec_b2 = _constrain_vs(df, n_quote, n_expiry)
    mat_A3, vec_b3 = _constrain_vb(df, n_quote, n_expiry)
    mat_A4, vec_b4 = _constrain_cs(df, n_quote)
    mat_A5, vec_b5 = _constrain_cvs(df, n_quote)
    mat_A6, vec_b6 = _constrain_cbs(df, n_quote)

    mat_A = _np.vstack((mat_A1, mat_A2, mat_A3, mat_A4, mat_A5, mat_A6))
    vec_b = _np.hstack((vec_b1, vec_b2, vec_b3, vec_b4, vec_b5, vec_b6))

    n_outright = mat_A1.shape[0]
    n_vs = mat_A2.shape[0]
    n_vb = mat_A3.shape[0]
    n_cs = mat_A4.shape[0]
    n_cvs = mat_A5.shape[0]
    n_cbs = mat_A6.shape[0]

    # detect violations of the static arbitrage constraints
    tolerance = 0.0 if tolerance < 0.0 else tolerance  # non-negative tolerance

    n_outright_breach = _np.sum((mat_A1.dot(C) - vec_b1) < -tolerance)
    n_vs_breach = _np.sum((mat_A2.dot(C) - vec_b2) < -tolerance)
    n_vb_breach = _np.sum((mat_A3.dot(C) - vec_b3) < -tolerance)
    n_cs_breach = _np.sum((mat_A4.dot(C) - vec_b4) < -tolerance)
    n_cvs_breach = _np.sum((mat_A5.dot(C) - vec_b5) < -tolerance)
    n_cbs_breach = _np.sum((mat_A6.dot(C) - vec_b6) < -tolerance)

    # report
    if verbose is True:
        print("Number of violations to non-negative "
              "outright price:                   {}/{}".
              format(n_outright_breach, n_outright))
        print("Number of violations to non-negative "
              "and unit-bounded vertical spread: {}/{}".
              format(n_vs_breach, n_vs))
        print("Number of violations to non-negative "
              "butterfly spread:                 {}/{}".
              format(n_vb_breach, n_vb))
        print("Number of violations to non-negative "
              "calendar (horizontal) spread:     {}/{}".
              format(n_cs_breach, n_cs))
        print("Number of violations to non-negative "
              "calendar vertical spread:         {}/{}".
              format(n_cvs_breach, n_cvs))
        print("Number of violations to non-negative "
              "calendar butterfly spread:        {}/{}".
              format(n_cbs_breach, n_cbs))

    n_cond = [n_outright, n_vs, n_vb, n_cs, n_cvs, n_cbs]
    n_breach = [n_outright_breach, n_vs_breach, n_vb_breach,
                n_cs_breach, n_cvs_breach, n_cbs_breach]

    return mat_A, vec_b, n_cond, n_breach


def _prepare_dataframe(T, K, C):

    """
    Prepares the pandas.DataFrame that pre-computes quantities that are needed
    for efficient construction of static arbitrage constraints.

    Parameters
    ----------
    T: array_like, 1D
        The 1D array of time-to-expiries.

    K: array_like, 1D
        The 1D array of normalised strike prices.

    C: array_like, 1D
        The 1D array of normalised call prices.

    Returns
    -------
    df: pandas.DataFrame
        Headers include:
        (1) T, K, C, T_order
        (2) raw_index (this is used to map index in the augmented data frame to
        raw index in the original data frame)
        (3) K_m1, K_m2, K_p1, K_order (m means minus, p means plus)
        (4) K_order_max (the maximal order of strike within the same expiry)

    """
    # create the data frame
    df = _pd.DataFrame(data={'T': T, 'K': K, 'C': C})

    df['raw_index'] = df.index.values
    df['T_order'] = df.groupby(['T']).ngroup()

    # augment the data frame by inserting zero-strike prices
    unique_tte, unique_tte_idx = _np.unique(T, return_index=True)
    n_expiry = unique_tte.shape[0]
    T_K0 = unique_tte
    K_K0 = _np.zeros(n_expiry)
    C_K0 = 1.0
    df_k0 = _pd.DataFrame(data={'T': T_K0, 'K': K_K0, 'C': C_K0})
    df = _pd.concat([df, df_k0], sort=True)  # concatenate two dfs
    del df_k0

    df.sort_values(by=['T', 'K'],
                   inplace=True)  # sort values by order: T --> K
    df.reset_index(drop=True, inplace=True)  # reset df index
    df['T_order'].fillna(method='bfill', inplace=True)

    # pre-compute quantities for constraints construction
    df['K_m1'] = df.groupby(['T_order'])['K'].shift(1)
    df['K_m2'] = df.groupby(['T_order'])['K'].shift(2)
    df['K_m1'].fillna(0, inplace=True)
    df['K_m2'].fillna(0, inplace=True)
    df['K_p1'] = df.groupby(['T_order'])['K'].shift(-1)
    df['K_p1'].fillna(1e10, inplace=True)
    df['K_order'] = df.groupby(['T_order']).cumcount()
    df['K_order_max'] = df.groupby('T_order')['K_order']. \
        transform(lambda x: x.max())

    return df


def _constrain_outright(data_frame, n_quote, n_expiry):
    """
    Constructs the coefficient matrix A and the vector of constant terms b for
    the outright constraints.

    Only options with the largest strike within the same expiry need to satisfy
    the non-negativity constraints. Other options' prices are automatically
    non-negative if all vertical spreads are non-negative.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame that contains call options' information.

    n_quote: int
        Number of call options.

    n_expiry: int
        Number of expiries.

    Returns
    -------
    mat_A: numpy.array, 2D, shape = (R, n_quote)
        Coefficient matrix. R = n_expiry is the number of constraints.

    vec_b: numpy.array, 1D, shape = (R, )
        Vector of constant terms. R = n_expiry is the number of constraints.

    """
    df = data_frame.copy()

    # locate the largest strike for each expiry
    idx = df[df['K_order'] == df['K_order_max']]['raw_index']. \
        values.astype(int)

    mat_A = _np.zeros([n_expiry, n_quote])
    mat_A[range(n_expiry), idx] = 1

    vec_b = _np.zeros(n_expiry)

    return mat_A, vec_b


def _constrain_vs(data_frame, n_quote, n_expiry):
    """
    Construct the coefficient matrix A and the vector of constant
    terms b for the Vertical Spread (VS) constraints:

    (1) non-negative vertical spread;
    (2) unit-bounded vertical spread.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame that contains call options' information.

    n_quote: int
        Number of call options.

    n_expiry: int
        Number of expiries.

    Returns
    -------
    mat_A: numpy.array, 2D, shape = (R, n_quote)
        Coefficient matrix. R = n_expiry + n_quote is the number of constraints.

    vec_b: numpy.array, 1D, shape = (R, )
        Vector of constant terms. R = n_expiry + n_quote is the number of
        constraints.

    """
    df = data_frame.copy()
    unique_T = df['T_order'].unique()

    mat_A1 = _np.zeros([n_quote, n_quote])  # non-negative constraints
    mat_A2 = _np.zeros([n_expiry, n_quote])  # unit-bounded constraints
    vec_b1 = _np.zeros(n_quote)
    vec_b2 = -_np.ones(n_expiry)

    i_start = 0
    j_start = 0
    k = 0
    for T in unique_T:
        K_diff = df[df['T_order'] == T].diff()['K'].values[1:]
        n_K = K_diff.shape[0]  # number of strikes at this expiry
        mat_diag = _np.diag(-1. / K_diff)
        mat_lower = _np.zeros([n_K, n_K])
        mat_lower[1:, :-1] = _np.diag(1. / K_diff[1:])
        mat_block_T = mat_diag + mat_lower
        mat_A1[i_start:i_start + n_K, j_start:j_start + n_K] = mat_block_T

        C0 = df[df['T_order'] == T]['C'].values[0]
        vec_b1[i_start] = -1. / K_diff[0] * C0

        mat_A2[k, k] = -1. / K_diff[0]
        vec_b2[k] = -1. / K_diff[0] * C0

        i_start += n_K
        j_start += n_K
        k += 1

    mat_A = _np.vstack((mat_A1, mat_A2))
    vec_b = _np.hstack((vec_b1, vec_b2))

    return mat_A, vec_b


def _constrain_vb(data_frame, n_quote, n_expiry):
    """
    Construct the coefficient matrix A and the vector of constant
    terms b for the Vertical Butterfly (VB) constraints.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame that contains call options' information.

    n_quote: int
        Number of call options.

    n_expiry: int
        Number of expiries.

    Returns
    -------
    mat_A: numpy.array, 2D, shape = (R, n_quote)
        Coefficient matrix. R = n_quote - n_expiry is the number of constraints.

    vec_b: numpy.array, 1D, shape = (R, )
        Vector of constant terms. R = n_quote - n_expiry is the number of
        constraints.

    """
    df = data_frame.copy()

    n_conds_bs = n_quote - n_expiry
    mat_A = _np.zeros([n_conds_bs, n_quote])
    vec_b = _np.zeros(n_conds_bs)
    unique_T = df['T_order'].unique()

    i_start = 0
    j_start = 0
    for T in unique_T:
        K_diff = df[df['T_order'] == T].diff()['K'].values[1:]
        K_diff_reciprocal = -1. / K_diff
        n_K = K_diff.shape[0]  # number of strikes at this maturity

        # constraints cannot be constructed if the number of strikes on the
        # specified expiry is less than 2, where no butterfly spread could be
        # constructed. The minimal number of strikes is 2 because the forward of
        # the same expiry could be one leg of the  butterfly spread.
        if n_K > 1:
            mat_block_T = _np.zeros((n_K - 1, n_K))
            mat_block_T[:, :-1] += _np.diag(
                K_diff_reciprocal[:-1] + K_diff_reciprocal[1:])
            mat_block_T[1:, :-2] += _np.diag(-K_diff_reciprocal[1:-1])
            mat_block_T[:, 1:] += _np.diag(-K_diff_reciprocal[1:])
            mat_A[i_start:i_start + n_K - 1, j_start:j_start + n_K] = \
                mat_block_T

            C0 = df[df['T_order'] == T]['C'].values[0]
            vec_b[i_start] = K_diff_reciprocal[0] * C0

        i_start += n_K - 1
        j_start += n_K

    return mat_A, vec_b


def _constrain_cs(data_frame, n_quote):
    """
    Constructs the coefficient matrix A and the vector of constant terms b for
    the Calendar Spread (CS) constraints.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame that contains call options' information.

    n_quote: int
        Number of call options.

    Returns
    -------
    mat_A: numpy.array, 2D, shape = (R, n_quote)
        Coefficient matrix. R ~ O(n_expiry * n_quote) is the number of
        constraints.

    vec_b: numpy.array, 1D, shape = (R, )
        Vector of constant terms. R ~ O(n_expiry * n_quote) is the number of
        constraints.

    """
    df = data_frame.copy()
    raw_index = df['raw_index'].values.astype(int)

    # indicator matrix and indices where conditions are satisfied
    ind_cond_T = _np.greater.outer(df['T'].values, df['T'].values)
    ind_cond_K = _np.equal.outer(df['K'].values, df['K'].values)
    true_idx = _np.nonzero(ind_cond_T * ind_cond_K)

    zero_K = df.loc[true_idx[0], 'K'] == 0
    mask = ~zero_K.values
    n_cond_cs = _np.sum(mask)  # number of conditions/constraints

    mat_A = _np.zeros([n_cond_cs, n_quote])
    row_idx = _np.arange(0, n_cond_cs)

    mat_A[row_idx, raw_index[true_idx[0][mask]]] = 1
    mat_A[row_idx, raw_index[true_idx[1][mask]]] = -1
    vec_b = _np.zeros(n_cond_cs)

    return mat_A, vec_b


def _constrain_cvs(data_frame, n_quote):
    """
    Constructs the coefficient matrix A and the vector of constant terms b for
    the Calendar Vertical Spread (CVS) constraints.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame that contains call options' information.

    n_quote: int
        Number of call options.

    Returns
    -------
    mat_A: numpy.array, 2D, shape = (R, n_quote)
        Coefficient matrix. R ~ O(n_expiry * n_quote) is the number of
        constraints.

    vec_b: numpy.array, 1D, shape = (R, )
        Vector of constant terms. R ~ O(n_expiry * n_quote) is the number of
        constraints.

    """
    df = data_frame.copy()
    raw_index = df['raw_index'].values.astype(int)

    # indicator matrix and indices where conditions are satisfied
    ind_cond_T = _np.less.outer(df['T'].values, df['T'].values)
    ind_cond_K_upper = _np.greater_equal.outer(df['K'].values, df['K'].values)
    ind_cond_K_lower = _np.less.outer(df['K_m1'].values, df['K'].values)
    true_idx = _np.nonzero(ind_cond_K_upper * ind_cond_T * ind_cond_K_lower)

    zero_K2 = df.loc[true_idx[1], 'K'] == 0
    zero_K1 = df.loc[true_idx[0], 'K'] == 0

    # case 1: K1, K2 != 0
    mask = ~zero_K2.values * ~zero_K1.values
    n_conds_cvs1 = _np.sum(mask)
    mat_A1 = _np.zeros([n_conds_cvs1, n_quote])
    row_idx = _np.arange(0, n_conds_cvs1)
    mat_A1[row_idx, raw_index[true_idx[0][mask]]] = -1
    mat_A1[row_idx, raw_index[true_idx[1][mask]]] = 1
    vec_b1 = _np.zeros(n_conds_cvs1)

    # case 2: K1 != 0, K2 = 0
    mask = zero_K2.values * ~zero_K1.values
    n_conds_cvs2 = _np.sum(mask)
    mat_A2 = _np.zeros([n_conds_cvs2, n_quote])
    row_idx = _np.arange(0, n_conds_cvs2)
    mat_A2[row_idx, raw_index[true_idx[0][mask]]] = -1
    vec_b2 = -df.loc[true_idx[1][mask], 'C'].values

    # sum up
    mat_A = _np.vstack((mat_A1, mat_A2))
    vec_b = _np.hstack((vec_b1, vec_b2))

    return mat_A, vec_b


def _constrain_cbs(data_frame, n_quote):
    """
    Constructs the coefficient matrix A and the vector of constant terms b for
    the Calendar Butterfly Spread (CBS) constraints.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        The data frame that contains call options' information.

    n_quote: int
        Number of call options.

    Returns
    -------
    mat_A: numpy.array, 2D, shape = (R, n_quote)
        Coefficient matrix. R ~ O(n_expiry^2 * n_quote) is the number of
        constraints.

    vec_b: numpy.array, 1D, shape = (R, )
        Vector of constant terms. R ~ O(n_expiry^2 * n_quote) is the number of
        constraints.

    """
    df = data_frame.copy()
    raw_index = df['raw_index'].values.astype(int)

    ##############################################
    # Step 1: ensure absolute location convexity #
    ##############################################
    ind_cond_T = _np.greater.outer(df['T'].values, df['T'].values)
    ind_cond_K_upper = _np.less.outer(df['K'].values, df['K'].values)
    ind_cond_K_lower = _np.greater.outer(df['K'].values, df['K_m1'].values)
    true_idx = _np.nonzero(ind_cond_K_upper * ind_cond_T * ind_cond_K_lower)

    idx_K = true_idx[0]
    idx_K_star = true_idx[1]
    df_cbs = _pd.DataFrame(
        data={'idx_K': idx_K, 'idx_K_star': idx_K_star})
    df_cbs['K_star_order'] = \
        df.loc[df_cbs['idx_K_star'].values, 'K_order'].values
    df_cbs['K_star_order_max'] = \
        df.loc[df_cbs['idx_K_star'].values, 'K_order_max'].values

    # Conditions for j* >= 1 and j* < n_i*
    mask = df_cbs['K_star_order'] < df_cbs['K_star_order_max']
    n_conds_cbs1 = _np.sum(mask)
    mat_A1 = _np.zeros([n_conds_cbs1, n_quote])

    array_K_star = df.loc[df_cbs[mask]['idx_K_star'].values, 'K'].values
    array_K = df.loc[df_cbs[mask]['idx_K'].values, 'K'].values
    array_K_star_p1 = \
        df.loc[df_cbs[mask]['idx_K_star'].values + 1, 'K'].values
    coef_C = -1. / (array_K - array_K_star)
    coef_C_star_p1 = -1. / (array_K_star - array_K_star_p1)
    coef_C_star = -coef_C - coef_C_star_p1

    row_idx = _np.arange(0, n_conds_cbs1)
    mat_A1[row_idx, raw_index[df_cbs[mask]['idx_K'].values]] = coef_C
    mat_A1[row_idx, raw_index[df_cbs[mask]['idx_K_star'].values]] = \
        coef_C_star
    mat_A1[row_idx, raw_index[df_cbs[mask]['idx_K_star'].values + 1]] = \
        coef_C_star_p1
    vec_b1 = _np.zeros(n_conds_cbs1)

    # conditions for j* > 1 and j* <= n_i*
    mask = df_cbs['K_star_order'] > 2
    n_conds_cbs2 = _np.sum(mask)
    mat_A2 = _np.zeros([n_conds_cbs2, n_quote])

    array_K = df.loc[df_cbs[mask]['idx_K'].values, 'K'].values
    array_K_star_m1 = \
        df.loc[df_cbs[mask]['idx_K_star'].values - 1, 'K'].values
    array_K_star_m2 = \
        df.loc[df_cbs[mask]['idx_K_star'].values - 2, 'K'].values
    coef_C = -1. / (array_K_star_m1 - array_K)
    coef_C_star_m2 = -1. / (array_K_star_m2 - array_K_star_m1)
    coef_C_star_m1 = -coef_C - coef_C_star_m2

    row_idx = _np.arange(0, n_conds_cbs2)
    mat_A2[row_idx, raw_index[df_cbs[mask]['idx_K'].values]] = coef_C
    mat_A2[row_idx, raw_index[df_cbs[mask]['idx_K_star'].values - 1]] = \
        coef_C_star_m1
    mat_A2[row_idx, raw_index[df_cbs[mask]['idx_K_star'].values - 2]] = \
        coef_C_star_m2
    vec_b2 = _np.zeros(n_conds_cbs2)

    # conditions for j* = 2
    mask = df_cbs['K_star_order'] == 2
    n_conds_cbs3 = _np.sum(mask)
    mat_A3 = _np.zeros([n_conds_cbs3, n_quote])

    array_K = df.loc[df_cbs[mask]['idx_K'].values, 'K'].values
    array_K_star_m1 = \
        df.loc[df_cbs[mask]['idx_K_star'].values - 1, 'K'].values
    array_K_star_m2 = \
        df.loc[df_cbs[mask]['idx_K_star'].values - 2, 'K'].values
    coef_C = -1. / (array_K_star_m1 - array_K)
    coef_C_star_m2 = -1. / (array_K_star_m2 - array_K_star_m1)
    coef_C_star_m1 = -coef_C - coef_C_star_m2

    row_idx = _np.arange(0, n_conds_cbs3)
    mat_A3[row_idx, raw_index[df_cbs[mask]['idx_K'].values]] = coef_C
    mat_A3[row_idx, raw_index[df_cbs[mask]['idx_K_star'].values - 1]] = \
        coef_C_star_m1
    vec_b3 = -df.loc[df_cbs[mask]['idx_K_star'].values - 2, 'C'].values * \
             coef_C_star_m2

    del df_cbs

    # conditions for K > K(i*, j* = n_i*)
    ind_cond_T = _np.greater.outer(df['T'].values, df['T'].values)
    ind_cond_K_lower = _np.greater.outer(df['K'].values, df['K'].values)
    true_idx = _np.nonzero(ind_cond_T * ind_cond_K_lower)

    idx_K = true_idx[0]
    idx_K_star = true_idx[1]
    df_cbs = _pd.DataFrame(data={'idx_K': idx_K, 'idx_K_star': idx_K_star})
    df_cbs['K_star_order'] = \
        df.loc[df_cbs['idx_K_star'].values, 'K_order'].values
    df_cbs['K_star_order_max'] = \
        df.loc[df_cbs['idx_K_star'].values, 'K_order_max'].values

    df_cbs = df_cbs[df_cbs['K_star_order'] == df_cbs['K_star_order_max']]

    df_cbs_1 = df_cbs[df_cbs['K_star_order_max'] == 1]
    df_cbs_2 = df_cbs[df_cbs['K_star_order_max'] > 1]

    n_conds_cbs6_1 = df_cbs_1.shape[0]
    mat_A6_1 = _np.zeros([n_conds_cbs6_1, n_quote])
    array_K = df.iloc[df_cbs_1['idx_K'].values]['K'].values
    array_K_star = df.iloc[df_cbs_1['idx_K_star'].values]['K'].values
    coef_C = 1. / (array_K - array_K_star)
    coef_C_star = -coef_C - 1. / array_K_star
    row_idx = _np.arange(0, n_conds_cbs6_1)
    mat_A6_1[row_idx, raw_index[df_cbs_1['idx_K'].values]] = coef_C
    mat_A6_1[row_idx, raw_index[df_cbs_1['idx_K_star'].values]] = \
        coef_C_star
    vec_b6_1 = -df.loc[df_cbs_1['idx_K_star'].values, 'C'].values * \
               array_K_star

    n_conds_cbs6_2 = df_cbs_2.shape[0]
    mat_A6_2 = _np.zeros([n_conds_cbs6_2, n_quote])
    array_K = df.iloc[df_cbs_2['idx_K'].values]['K'].values
    array_K_star = df.iloc[df_cbs_2['idx_K_star'].values]['K'].values
    array_K_star_m1 = df.iloc[df_cbs_2['idx_K_star'].values - 1]['K'].values
    coef_C = -1. / (array_K_star - array_K)
    coef_C_star_m1 = -1. / (array_K_star_m1 - array_K_star)
    coef_C_star = -coef_C - coef_C_star_m1
    row_idx = _np.arange(0, n_conds_cbs6_2)
    mat_A6_2[row_idx, raw_index[df_cbs_2['idx_K'].values]] = coef_C
    mat_A6_2[row_idx, raw_index[df_cbs_2['idx_K_star'].values]] = \
        coef_C_star
    mat_A6_2[row_idx, raw_index[df_cbs_2['idx_K_star'].values - 1]] = \
        coef_C_star_m1
    vec_b6_2 = _np.zeros(n_conds_cbs6_2)

    mat_A6 = _np.vstack([mat_A6_1, mat_A6_2])
    vec_b6 = _np.hstack([vec_b6_1, vec_b6_2])

    n_conds_cbs6 = n_conds_cbs6_1 + n_conds_cbs6_2
    del df_cbs, df_cbs_1, df_cbs_2

    ##############################################
    # Step 2: ensure relative location convexity #
    ##############################################

    ind_cond_T = _np.greater.outer(df['T'].values, df['T'].values)
    ind_cond_K1_upper = _np.less.outer(df['K'].values, df['K'].values)
    ind_cond_K1_lower = _np.greater.outer(df['K'].values, df['K_m1'].values)
    true_idx_K1 = _np.nonzero(
        ind_cond_K1_upper * ind_cond_T * ind_cond_K1_lower)

    ind_cond_K2_upper = _np.less.outer(df['K'].values, df['K_p1'].values)
    ind_cond_K2_lower = _np.greater.outer(df['K'].values, df['K'].values)
    true_idx_K2 = _np.nonzero(
        ind_cond_K2_upper * ind_cond_T * ind_cond_K2_lower)

    df_cbs_1 = _pd.DataFrame(
        data={'idx_K_star': true_idx_K1[1], 'idx_K1': true_idx_K1[0]})
    df_cbs_2 = _pd.DataFrame(
        data={'idx_K_star': true_idx_K2[1], 'idx_K2': true_idx_K2[0]})
    df_cbs = _pd.merge(df_cbs_1, df_cbs_2, on='idx_K_star')[
        ['idx_K_star', 'idx_K1', 'idx_K2']]

    df_cbs['K_star_order'] = \
        df.loc[df_cbs['idx_K_star'].values, 'K_order'].values
    df_cbs['K_star_order_max'] = \
        df.loc[df_cbs['idx_K_star'].values, 'K_order_max'].values

    mask = df_cbs['K_star_order'] != df_cbs['K_star_order_max']
    df1 = df_cbs[mask].copy()

    n_conds_cbs4 = df1.shape[0]
    mat_A4 = _np.zeros([n_conds_cbs4, n_quote])

    array_K1 = df.iloc[df1['idx_K1'].values]['K'].values
    array_K2 = df.iloc[df1['idx_K2'].values]['K'].values
    array_K_star = df.iloc[df1['idx_K_star'].values]['K'].values
    coef_C1 = -1. / (array_K1 - array_K_star)
    coef_C2 = -1. / (array_K_star - array_K2)
    coef_C_star = -coef_C1 - coef_C2

    row_idx = _np.arange(0, n_conds_cbs4)
    mat_A4[row_idx, raw_index[df1['idx_K1'].values]] = \
        coef_C1
    mat_A4[row_idx, raw_index[df1['idx_K2'].values]] = \
        coef_C2
    mat_A4[row_idx, raw_index[df1['idx_K_star'].values]] = \
        coef_C_star

    vec_b4 = _np.zeros(n_conds_cbs4)

    del df1, df_cbs_1, df_cbs_2, df_cbs

    ind_cond_T = _np.greater.outer(df['T'].values, df['T'].values)
    ind_cond_T2 = _np.equal.outer(df['T_order'].values, df['T_order'].values)

    ind_cond_K1_upper = _np.less.outer(df['K'].values, df['K'].values)
    ind_cond_K1_lower = _np.greater.outer(df['K'].values, df['K_m1'].values)
    true_idx_K1 = _np.nonzero(
        ind_cond_K1_upper * (ind_cond_T + ind_cond_T2) * ind_cond_K1_lower)
    idx_K_1 = true_idx_K1[1]
    mask = df.loc[idx_K_1, 'K_order'] == df.loc[idx_K_1, 'K_order_max']
    idx_K_1 = idx_K_1[mask.values]
    idx_K1 = true_idx_K1[0][mask.values]

    ind_cond_K2_lower = _np.greater.outer(df['K'].values, df['K'].values)
    true_idx_K2 = _np.nonzero(ind_cond_T * ind_cond_K2_lower)
    idx_K_2 = true_idx_K2[1]
    mask = df.loc[idx_K_2, 'K_order'] == df.loc[idx_K_2, 'K_order_max']
    idx_K_2 = idx_K_2[mask.values]
    idx_K2 = true_idx_K2[0][mask.values]

    df_cbs_1 = _pd.DataFrame(
        data={'idx_K_star': idx_K_1, 'idx_K1': idx_K1})
    df_cbs_2 = _pd.DataFrame(
        data={'idx_K_star': idx_K_2, 'idx_K2': idx_K2})
    df_cbs = _pd.merge(df_cbs_1, df_cbs_2, on='idx_K_star')[
        ['idx_K_star', 'idx_K1', 'idx_K2']]
    df_cbs = df_cbs[df_cbs['idx_K_star'] != df_cbs['idx_K1']]

    n_conds_cbs5 = df_cbs.shape[0]
    mat_A5 = _np.zeros([n_conds_cbs5, n_quote])

    array_K1 = df.iloc[df_cbs['idx_K1'].values]['K'].values
    array_K2 = df.iloc[df_cbs['idx_K2'].values]['K'].values
    array_K_star = df.iloc[df_cbs['idx_K_star'].values]['K'].values
    coef_C1 = -1. / (array_K1 - array_K_star)
    coef_C2 = -1. / (array_K_star - array_K2)
    coef_C_star = -coef_C1 - coef_C2

    row_idx = _np.arange(0, n_conds_cbs5)
    mat_A5[row_idx, raw_index[df_cbs['idx_K1'].values]] = coef_C1
    mat_A5[row_idx, raw_index[df_cbs['idx_K2'].values]] = coef_C2

    vec_b5 = -df.loc[df_cbs['idx_K_star'].values, 'C'].values * \
             coef_C_star

    del df_cbs, df_cbs_1, df_cbs_2

    # summarise
    mat_A = _np.vstack((mat_A1, mat_A2, mat_A3, mat_A4, mat_A5, mat_A6))
    vec_b = _np.hstack((vec_b1, vec_b2, vec_b3, vec_b4, vec_b5, vec_b6))
    n_conds_cbs = n_conds_cbs1 + n_conds_cbs2 + n_conds_cbs3 + \
                  n_conds_cbs4 + n_conds_cbs5 + n_conds_cbs6

    return mat_A, vec_b

