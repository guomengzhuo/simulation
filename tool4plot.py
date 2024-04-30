import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")


    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, fontdict={'size':18})
    ax.set_yticklabels(row_labels, fontdict={'size':18})
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def barPlot(data1, data2):
    fig, ax = plt.subplots(figsize=(6,4))
    cat_name = ['True', 'Prediction']
    cat_values1 = [data1[0], data2[0]]
    cat_values2 = [data1[1], data2[1]]
    width = .3
    p = ax.bar([0.0, 0.7], cat_values1, width=width, color=rgb_to_hex(14, 62, 135), label=r'$\pi_0 = P(i_1=0)$', )
    ax.bar_label(p, label_type='center', fmt="%.4f", size=18)
    p = ax.bar([0.0, 0.7], cat_values2, width=width, color=rgb_to_hex(216, 178, 58), bottom=cat_values1, label=r'$\pi_1 = P(i_1=1)$')
    ax.bar_label(p, label_type='center', fmt="%.4f", size=18)

    ax.set_xticks([0, 0.7])
    ax.set_xticklabels(cat_name, fontdict={'size':18})

    ax.set_yticks(np.arange(0, max(cat_values1 + cat_values2) + 1, 2))

    # ax.set_xlabel('X Axis Label')
    # ax.set_ylabel('Prob.')

    ax.legend(loc='upper right', fontsize=15, ncol=2)
    plt.savefig('pi_pred_true.pdf', format='pdf', dpi=300)
    plt.show()

def tranMat(data):
    rank = ['low', 'high']
    # Plot Q mat
    fig, ax_all = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    ax = ax_all[0]
    im, cbar = heatmap(data['pred_Q'], rank, rank, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Predicted Probability', fontdict={"size": 18})

    ax = ax_all[1]
    im, cbar = heatmap(data['true_Q'], rank, rank, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('True Probability', fontdict={"size": 18})
    fig.tight_layout()
    plt.savefig('Q.pdf',format='pdf',dpi=300)
    plt.show()

    # Plot F mat
    fig, ax_all = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax = ax_all[0]
    im, cbar = heatmap(data['pred_F'], rank, rank, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Predicted Probability', fontdict={"size": 18})

    ax = ax_all[1]
    im, cbar = heatmap(data['true_F'], rank, rank, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('True Probability', fontdict={"size": 18})
    fig.tight_layout()
    plt.savefig('F.pdf', format='pdf', dpi=300)
    plt.show()

    # Plot N mat
    fig, ax_all = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax = ax_all[0]
    im, cbar = heatmap(data['pred_N'], rank, rank, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Predicted Probability', fontdict={"size": 18})

    ax = ax_all[1]
    im, cbar = heatmap(data['true_N'], rank, rank, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('True Probability', fontdict={"size": 18})
    fig.tight_layout()
    plt.savefig('N.pdf', format='pdf', dpi=300)
    plt.show()

def plotMat2(data):
    rank = ['low', 'high']
    for i in [0, 1]:
        # Plot M mat
        fig, ax_all = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax = ax_all[0]
        im, cbar = heatmap(data['pred_M'][i], rank, rank, ax=ax,
                           cmap="YlGn", cbarlabel="Probability")
        texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Predicted Probability', fontdict={"size": 18})

        ax = ax_all[1]
        im, cbar = heatmap(data['true_M'][i], rank, rank, ax=ax,
                           cmap="YlGn", cbarlabel="Probability")
        texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('True Probability', fontdict={"size": 18})
        fig.tight_layout()
        plt.savefig('M_'+str(i)+'.pdf', format='pdf', dpi=300)
        plt.show()

        # Plot H mat
        fig, ax_all = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax = ax_all[0]
        im, cbar = heatmap(data['pred_H'][i], rank, rank, ax=ax,
                           cmap="YlGn", cbarlabel="Probability")
        texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Predicted Probability', fontdict={"size": 18})

        ax = ax_all[1]
        im, cbar = heatmap(data['true_H'][i], rank, rank, ax=ax,
                           cmap="YlGn", cbarlabel="Probability")
        texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('True Probability', fontdict={"size": 18})
        fig.tight_layout()
        plt.savefig('H_' + str(i) + '.pdf', format='pdf', dpi=300)
        plt.show()


def plotObser(data):
    rank = ['low', 'high']
    obser = ['Obs.=0', 'Obs.=1']
    # Plot Q mat
    fig, ax_all = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax = ax_all[0]
    im, cbar = heatmap(data['pred_C'], rank, obser, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Predicted Probability', fontdict={"size": 18})

    ax = ax_all[1]
    im, cbar = heatmap(data['true_C'], rank, obser, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('True Probability', fontdict={"size": 18})
    fig.tight_layout()
    plt.savefig('C.pdf', format='pdf', dpi=300)
    plt.show()

    # Plot E mat
    fig, ax_all = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax = ax_all[0]
    im, cbar = heatmap(data['pred_E'], rank, obser, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Predicted Probability', fontdict={"size": 18})

    ax = ax_all[1]
    im, cbar = heatmap(data['true_E'], rank, obser, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('True Probability', fontdict={"size": 18})
    fig.tight_layout()
    plt.savefig('E.pdf', format='pdf', dpi=300)
    plt.show()

    # Plot D mat
    fig, ax_all = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax = ax_all[0]
    im, cbar = heatmap(data['pred_D'], rank, obser, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Predicted Probability', fontdict={"size": 18})

    ax = ax_all[1]
    im, cbar = heatmap(data['true_D'], rank, obser, ax=ax,
                       cmap="YlGn", cbarlabel="Probability")
    texts = annotate_heatmap(im, valfmt="{x:.4f}", size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('True Probability', fontdict={"size": 18})
    fig.tight_layout()
    plt.savefig('D.pdf', format='pdf', dpi=300)
    plt.show()

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"