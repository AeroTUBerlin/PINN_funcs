import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "serif"

def make_2D(arrays, xi=0, yi=1):
    x = np.asarray(arrays[xi]).ravel()
    y = np.asarray(arrays[yi]).ravel()

    x_unique, xi_idx = np.unique(x, return_inverse=True)
    y_unique, yi_idx = np.unique(y, return_inverse=True)
    nx = len(x_unique)
    ny = len(y_unique)

    if nx * ny != x.size:
        raise ValueError(
            f"Data does not form a regular grid: "
            f"{nx} unique x-values × {ny} unique y-values = {nx * ny} ≠ {x.size} points."
        )
    result = []
    for arr in arrays:
        grid = np.empty((ny, nx))
        grid[yi_idx, xi_idx] = np.asarray(arr).ravel()
        result.append(grid)

    return result, ny, nx

def contourf(x, y, c,
             sname=None, title=None, zero = False,
             size = 8, show = True,
             **kwargs):

    out, M, N = make_2D([x, y, c])
    x, y, c = out

    AR = (y.max() - y.min()) / (x.max() - x.min()) # Aspect Ratio

    # if AR > 1:
    #     AR = 1 / AR
    #     y,x,c = np.transpose([x,y,c], (0,2,1))

    fig, ax = plt.subplots(figsize=(size, size * AR + 0.5), constrained_layout = True)
    fig.gca().set_aspect("equal")

    p = ax.contourf(x, y, c, **kwargs)
    if zero:
        ax.contour(x, y, c, [0], linewidths = 0.4, colors = 'black')
    fig.colorbar(p, ax = ax, orientation = 'vertical', shrink = 0.7)
    if title:
        plt.title(title)
    if sname:
        plt.savefig(sname+'.png', dpi=200)
    if show:
        plt.show()
    plt.close()
    return fig

def r2_plot(ref, pred, title='Test', sname=None, show=True, size=4.5):
    r2 = 1 - np.sum((pred - ref) ** 2) / np.sum((ref - np.mean(ref)) ** 2)
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    ax.plot([ref.min(), ref.max()], [ref.min(), ref.max()], 'k--')
    ax.scatter(ref, pred, s=1, c='b', alpha=0.5)
    ax.set_xlabel(f'{title} Reference')
    ax.set_ylabel(f'{title} Prediction')
    ax.set_title(f'R² = {r2:.4f}, {title}')
    ax.set_aspect('equal')

    ## Equal range and ticks for both axes
    lim = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ## Set ticks with nice values
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(True, linestyle='--', alpha=0.5)

    if sname:
        plt.savefig(sname+'.png', dpi=200)
    if show:
        plt.show()
    plt.close()
    return fig

def plot_results(x, y, result_dict, spath='results', levels = None, cmap = None, mask=None, **kwargs):
    """
    Plot results from a dictionary of values.
    
    Parameters:
    -----------
    x, y : array-like
        The x and y coordinates for the plot
    result_dict : dict
        A dictionary of results to plot, where keys are the names of the results and values are the corresponding values
    spath : str, optional
        The path to save the plot (without extension)
    levels : array-like, optional
        The levels for the contour plot
    cmap : str, optional
        The colormap to use for the plot
    **kwargs : dict
        Additional arguments to pass to contourf
    """
    os.makedirs(spath, exist_ok = True)
    if cmap is None:
        cmap = np.tile('viridis', len(result_dict))
    elif isinstance(cmap, str):
        cmap = np.tile(cmap, len(result_dict))
    else:
        raise ValueError("cmap must be a single string or a list of strings with the same length as result_dict")

    if levels is None:
        levels = np.tile(50, len(result_dict)).tolist()
    elif isinstance(levels, (int, float)):
        levels = np.tile(levels, len(result_dict)).tolist()
    elif isinstance(levels, np.ndarray):
        levels = np.vstack([levels] * len(result_dict))
    else:
        raise ValueError("levels must be a single number or a list of numbers with the same length as result_dict")
    
    pp = PdfPages(os.path.join(spath, 'output.pdf'))
    for i, (key, value) in enumerate(result_dict.items()):
        if isinstance(value, float):
            continue
        if 'err_' in key:
            cmap[i] = 'RdBu'
            levels[i] = np.linspace(-abs(value).max(), abs(value).max(), 40)

        plot = contourf(x, y, value,
                        title=key, sname=os.path.join(spath, key),
                        cmap = cmap[i], levels = levels[i],
                        show = False, **kwargs)
        pp.savefig(plot)
        
        if 'ref' in key:
            pred = result_dict[key.replace('_ref', '')]
            if mask is not None:
                value = value[mask]
                pred = pred[mask]
            plot = r2_plot(value, pred, 
                           title=key.replace('_ref', ''), 
                           sname=os.path.join(spath, 'r2_'+key.replace('_ref', '')), 
                           show = False
                           )
            pp.savefig(plot)
    pp.close()
