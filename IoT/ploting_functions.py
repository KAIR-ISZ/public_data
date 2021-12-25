import numpy as np
import warnings
import matplotlib.pyplot as plt


LIGHT = "#FFFCDC"
LIGHT_HIGHLIGHT = "#FEF590"
MID = "#FDED2A"
MID_HIGHLIGHT = "#F0DC05"
DARK = "#EECA02"
DARK_HIGHLIGHT = "#BB9700"
GREEN = "#00FF00"
LIGHT_GREY = "#DDDDDD"


def is_sorted(a):
    """Check if numpy 1d-array is sorted"""
    if type(a) != np.ndarray:
        raise TypeError("Argument must be a numpy array but is {}".format(type(a)))
    if len(a.shape) > 1:
        raise ValueError("Array must be 1 dimensional but has shape {}".format(a.shape))
    return np.all(a[:-1] <= a[1:])


def sort_1d_array_and_2d_array_by_1d_array(x, fx):
    if (type(x) != np.ndarray) or (type(fx) != np.ndarray):
        raise TypeError(
            "At least one of the arguments is not a numpy array type(x)={}, type(fx)={}",
            format(type(x), type(fx)),
        )
    if len(x) != fx.shape[1]:
        raise ValueError(
            "2d array number of columns is not matching the 1d array. Expected {} got {}".format(
                len(x), fx.shape[1]
            )
        )
    arr2D = np.concatenate([np.expand_dims(x, axis=0), fx], axis=0)
    sortedArr = arr2D[:, arr2D[0].argsort()]
    return sortedArr[0, :], sortedArr[1:, :]


def get_quantiles(fx, probs=None):
    if probs is None:
        probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    if len(probs) % 2 == 0:
        raise ValueError("Number of quantiles must be even")
    if len(probs) > 11:
        raise ValueError("Too many quantiles (max is 11)")
    if probs[int(len(probs) / 2)] != 50:
        raise ValueError(
            "Middle quantile should be 50 but is {}".format(probs(int(len(probs) / 2)))
        )
    return np.percentile(fx, probs, axis=0)


def ribbon_plot(x, fx, ax=None, zorder=0, probs=None, supress_warning=False):
    """Plot a ribbon plot for regression and similar.
    Plot consists of quantiles (by 10%) of a variate (fx) as a function of covariate (x).
    x has shape (n, )
    fx has shape (N,n)
    """
    if ax is None:
        ax = plt.gca()
    if not is_sorted(x):
        x, fx = sort_1d_array_and_2d_array_by_1d_array(x, fx)
    if (len(set(x)) != len(x)) and (not supress_warning):
        warnings.warn("x variable has repeated values, which can influence the plot")
    perc_interv = get_quantiles(fx, probs)
    nq = perc_interv.shape[0]
    colortab = [LIGHT, LIGHT_HIGHLIGHT, MID, MID_HIGHLIGHT, DARK, DARK_HIGHLIGHT]

    for i in range(int(nq / 2)):
        ax.fill_between(
            x,
            perc_interv[i, :],
            perc_interv[-(i + 1), :],
            color=colortab[i],
            zorder=zorder,
        )
    ax.plot(x, perc_interv[int(nq / 2), :], color=colortab[int(nq / 2)], zorder=zorder)
    return ax


def integer_histogram_matrix(max_y, y_ppc):
    if len(y_ppc.shape) == 1:
        y_ppc = np.expand_dims(y_ppc, axis=0)
    B = max_y + 1
    bins = np.array([*range(B + 1)]) - 0.5
    counts = [np.histogram(y_ppc[n], bins=bins)[0] for n in range(y_ppc.shape[0])]
    return bins, np.array(counts)


def real_histogram_matrix(bins, y_ppc):
    if len(y_ppc.shape) == 1:
        y_ppc = np.expand_dims(y_ppc, axis=0)
    counts = [np.histogram(y_ppc[n], bins=bins)[0] for n in range(y_ppc.shape[0])]
    return bins, np.array(counts)


def pad_hist_for_plot(bins, counts):
    if len(counts.shape) == 1:
        ax = 0
    else:
        ax = 1

    xs = (np.repeat(bins, repeats=2))[1:-1]
    pad_counts = np.repeat(counts, repeats=2, axis=ax)
    return xs, pad_counts


def visualise_integer_predictions(
    data, sample, ax, xlabel="resets", ylabel="Counts", PPC=False
):
    """[summary]

    Args:
        data ([type]): [description]
        sample ([type]): [description]
        ax ([type]): [description]
        xlabel (str, optional): [description]. Defaults to 'resets'.
        ylabel (str, optional): [description]. Defaults to 'Counts'.

    Returns:
        [type]: [description]
    """
    max_y = min(
        max(np.max(sample).astype(int), np.max(data).astype(int)),
        2 * np.max(data).astype(int),
    )
    probs = None
    if PPC:
        max_y = (1.5 * np.max(data)).astype(int)
        probs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    bins, counts = integer_histogram_matrix(max_y, sample)
    xs, pad_counts_pred = pad_hist_for_plot(bins, counts)
    obs_counts = np.histogram(data, bins=bins)[0]

    _, pad_obs_counts = pad_hist_for_plot(bins, obs_counts)
    ax = ribbon_plot(xs, pad_counts_pred, ax, supress_warning=True, probs=probs)
    ax.plot(xs, pad_obs_counts, linewidth=2.5, color="white", zorder=1)
    ax.plot(xs, pad_obs_counts, linewidth=2.0, color="black", zorder=2)
    ax.set_xlim([min(bins), max(bins)])
    ax.set_xlabel(xlabel)
    ax.set_ylim([0, max(max(obs_counts), np.quantile(counts, 0.99)) + 1])
    if PPC:
        ax.set_ylim([0, 1.5 * max(obs_counts)])
    ax.set_ylabel(ylabel)
    return ax


def visualise_continuous_predictions(
    data, sample, ax, xlabel="Failure time", ylabel="Counts", binwidth=2.5, PPC=False
):
    """[summary]

    Args:
        data ([type]): [description]
        sample ([type]): [description]
        ax ([type]): [description]
        xlabel (str, optional): [description]. Defaults to "Failure time".
        ylabel (str, optional): [description]. Defaults to 'Counts'.
        binwidth (int, optional): [description]. Defaults to 2.5.

    Returns:
        [type]: [description]
    """

    max_y = min(
        np.around(np.max(sample) / binwidth) * binwidth,
        np.around(1.5 * np.max(data) / binwidth) * binwidth,
    )
    if PPC:
        max_y = np.around(1.5 * np.max(data) / binwidth) * binwidth

    obs_counts, bins = np.histogram(data, bins=np.arange(0, max_y, binwidth))
    _, pad_obs_counts = pad_hist_for_plot(bins, obs_counts)
    bins, counts = real_histogram_matrix(bins, sample)
    xs, pad_counts_pred = pad_hist_for_plot(bins, counts)

    ax = ribbon_plot(xs, pad_counts_pred, ax, supress_warning=True)
    ax.plot(xs, pad_obs_counts, linewidth=2.5, color="white", zorder=1)
    ax.plot(xs, pad_obs_counts, linewidth=2.0, color="black", zorder=2)
    ax.set_xlim([min(bins), max(bins)])
    ax.set_xlabel(xlabel)
    ax.set_ylim([0, max(max(obs_counts), np.max(counts)) + 1])
    if PPC:
        ax.set_ylim([0, 1.5 * max(obs_counts)])
    ax.set_ylabel(ylabel)
    return ax


def plot_individual(resets_sample, failures_sample, df, name_prefix="", close=False):
    y_prediction_1 = resets_sample
    y_prediction_2 = failures_sample

    lista = list(df.batch.value_counts().index)
    lista.sort()

    for batch_no in lista:
        batch_reset_prediction = np.array(
            [resets[df.batch == batch_no] for resets in y_prediction_1]
        )
        batch_reset_data = df.resets[df.batch == batch_no]

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7, 4))

        ax = visualise_integer_predictions(batch_reset_data, batch_reset_prediction, ax)

        if batch_no == 3:
            ax.set_xlim([100, ax.get_xlim()[1]])

        batch_failure_prediction = np.array(
            [failure_time[df.batch == batch_no] for failure_time in y_prediction_2]
        )
        batch_failure_data = df.failure_time[df.batch == batch_no]

        if batch_no == 3:
            binwidth = 0.2
        else:
            binwidth = 2.5
        ax2 = visualise_continuous_predictions(
            batch_failure_data, batch_failure_prediction, ax2, binwidth=binwidth
        )

        if batch_no == 3:
            ax2.set_xlim([0, 5])

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.suptitle("Batch no. {}".format(batch_no))
        fig.savefig(name_prefix + "individual_ppd_batch_{}.png".format(batch_no))
        if close:
            plt.close()


def plot_failures_batches(failures_sample, df, fig, axes, name_prefix="", close=False):

    lista = list(df.batch.value_counts().index)
    lista.sort()
    axes = axes.flatten()[: len(lista)]
    it_over = zip(lista, axes)

    for batch_no, ax2 in it_over:

        batch_failure_prediction = np.array(
            [failure_time[df.batch == batch_no] for failure_time in failures_sample]
        )
        batch_failure_data = df.failure_time[df.batch == batch_no]

        if batch_no == 3:
            binwidth = 0.2
        else:
            binwidth = 2.5
        ax2 = visualise_continuous_predictions(
            batch_failure_data, batch_failure_prediction, ax2, binwidth=binwidth
        )
        if batch_no == 3:
            ax2.set_xlim([0, 5])

        ax2.set_title("Batch no. {}".format(batch_no))
    fig.tight_layout()
    fig.savefig(name_prefix + "failure_ppd_batches.png")
    if close:
        plt.close()


def plot_resets_batches(resets_sample, df, fig, axes, name_prefix="", close=False):

    lista = list(df.batch.value_counts().index)
    lista.sort()
    axes = axes.flatten()[: len(lista)]
    it_over = zip(lista, axes)

    for batch_no, ax in it_over:

        batch_reset_prediction = np.array(
            [resets[df.batch == batch_no] for resets in resets_sample]
        )
        batch_reset_data = df.resets[df.batch == batch_no]

        ax = visualise_integer_predictions(batch_reset_data, batch_reset_prediction, ax)

        if batch_no == 3:
            ax.set_xlim([100, ax.get_xlim()[1]])

        ax.set_title("Batch no. {}".format(batch_no))
    fig.tight_layout()
    fig.savefig(name_prefix + "resets_ppd_batches.png")
    if close:
        plt.close()
