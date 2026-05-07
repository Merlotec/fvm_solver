import numpy as np
import torch
from matplotlib import pyplot as plt, tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_points(Xs, values, lims=None, title="", show_index=False, Xlims=None):
    Xs = Xs.cpu()
    values = values.cpu()

    if len(values.shape) == 1:
        values = values.unsqueeze(0)
        fig, axes = plt.subplots(1, 1, figsize=(12, 9))
        axes = [axes]
    else:
        n_plots = values.shape[0]
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 4))

    # Loop over each batch
    if Xlims is None:
        Xlims = (Xs[:, 0].min(), Xs[:, 0].max()), (Xs[:, 1].min(), Xs[:, 1].max())

    for i, ax in enumerate(axes):
        ax.set_title(f"{title} - Batch {i}")
        if lims is None:
            sc = ax.scatter(Xs[:, 0], Xs[:, 1], c=values[i], cmap='viridis')
        else:
            sc = ax.scatter(Xs[:, 0], Xs[:, 1], c=values[i], cmap='viridis',
                            vmin=lims[0], vmax=lims[1])

        if show_index:
            for i, X in enumerate(Xs):
                x, y = X
                if (Xlims[0][0] <= x <= Xlims[0][1]) and (Xlims[1][0] <= y <= Xlims[1][1]):
                    ax.text(x, y, f"{i}", fontsize=8)
        fig.colorbar(sc, ax=ax)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlim(Xlims[0])
        ax.set_ylim(Xlims[1])

    # plt.tight_layout()
    plt.show()


def plot_interp_cell(Xs, values, triangles, Xlims=None, title="", edgecolors="none"):
    """
    Xs: Tensor of vertex coordinates (N x 2)
    values: Tensor of face-based values.
            If values is 1D, it's assumed to be defined on the triangulation faces.
            If 2D, each row is treated as a separate batch. shape = (B, M)
    triangles: Tensor of triangle vertex indices (M x 3).
    Xlims: Optional tuple ((xmin, xmax), (ymin, ymax)) to set the plot limits.
    title: Plot title.
    """
    # Convert to numpy arrays.
    Xs = Xs.cpu().numpy()
    values = values.cpu().numpy()
    triangles = triangles.cpu().numpy()

    # If values is 1D, expand to a batch of one.
    values = values.squeeze()
    if len(values.shape) == 1:
        values = values[None, :]

    n_plots = values.shape[0]
    if n_plots <= 3:
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 6))
    else:
        # Use a near-square layout for larger batches.
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    axes = np.atleast_1d(axes).ravel()
    plot_axes = axes[:n_plots]
    for ax in axes[n_plots:]:
        ax.axis('off')

    # Create a triangulation from the vertex locations.
    triang = tri.Triangulation(Xs[:, 0], Xs[:, 1], triangles)

    if isinstance(title, str):
        title = [title] * n_plots

    # Determine plot limits.
    if Xlims is not None:
        xlim, ylim = Xlims
    else:
        xlim = (Xs[:, 0].min(), Xs[:, 0].max())
        ylim = (Xs[:, 1].min(), Xs[:, 1].max())

    triangles = triang.triangles  # shape: (n_triangles, 3)
    verts = Xs[triangles]  # shape: (n_triangles, 3, 2)

    in_region = (
            (verts[:, :, 0] >= xlim[0]) &
            (verts[:, :, 0] <= xlim[1]) &
            (verts[:, :, 1] >= ylim[0]) &
            (verts[:, :, 1] <= ylim[1])
    ).all(axis=1)

    # Create a new triangulation using only the triangles inside the region.
    new_triangles = triang.triangles[in_region]
    new_triang = tri.Triangulation(Xs[:, 0], Xs[:, 1], triangles=new_triangles)

    # Loop over each batch and plot only the triangles inside the region.
    for i, ax in enumerate(plot_axes):
        ax.set_title(f"{title[i]}")
        # Filter the face-based values for the triangles inside the region.
        new_facecolors = values[i][in_region]

        # Plot using the new triangulation and corresponding facecolors.
        tc = ax.tripcolor(new_triang, facecolors=new_facecolors, edgecolors=edgecolors,
                          cmap='viridis', shading='flat')

        # Attach a dedicated colorbar axis to keep colorbar height matched to this subplot.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        fig.colorbar(tc, cax=cax)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_interp_vertex(Xs, values, triangles=None, Xlims=None, title="", edgecolors="none"):
    """
    Xs:     shape = [n_points, 2] Tensor of vertex coordinates
    values: shape = [n_plots, n_points]. Tensor of face-based values.
            If values is 1D, it's assumed to be defined on the triangulation faces.
            If 2D, each row is treated as a separate batch.
    Xlims: Optional tuple ((xmin, xmax), (ymin, ymax)) to set the plot limits.
    title: Plot title.
    """

    # Convert to numpy arrays.
    Xs = Xs.cpu().numpy()
    values = values.cpu().numpy()

    # If values is 1D, expand to a batch of one.
    if len(values.shape) == 1:
        values = values[None, :]
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes = [axes]
    else:
        n_plots, n_points = values.shape
        if n_plots > n_points:
            print("Warning ")
            raise ValueError(f"Number of plots ({n_plots}) exceeds number of points ({n_points}).")

        fig, axes = plt.subplots(n_plots, 1, figsize=(6, n_plots * 4))
        if n_plots == 1:
            axes = [axes]

    # Make title into list[str] for each plot.
    if isinstance(title, str):
        title = [title] * len(axes)

    # Determine plot limits.
    if Xlims is not None:
        xlim, ylim = Xlims
    else:
        xlim = (Xs[:, 0].min(), Xs[:, 0].max())
        ylim = (Xs[:, 1].min(), Xs[:, 1].max())


    triang = tri.Triangulation(Xs[:, 0], Xs[:, 1], triangles)
    # Filter out triangles that are outside the specified region.
    tri_mask = []
    for tri_indices in triang.triangles:
        x_vert, y_vert = Xs[tri_indices, 0], Xs[tri_indices, 1]
        if np.all((x_vert[0] >= xlim[0]) & (x_vert[0] <= xlim[1]) &
                  (y_vert[1] >= ylim[0]) & (y_vert[1] <= ylim[1])):
            tri_mask.append(True)
        else:
            tri_mask.append(False)
    tri_mask = np.array(tri_mask)
    vert_idx = np.unique(triang.triangles[tri_mask])
    vertex_mask = np.zeros(Xs.shape[0], dtype=bool)
    vertex_mask[vert_idx] = True
    triang.set_mask(~tri_mask)

    # Loop over each batch and plot only the triangles inside the region.
    for i, ax in enumerate(axes):
        ax.set_title(f"{title[i]}")
        v = np.ma.array(values[i], mask=~vertex_mask)
        tc = ax.tripcolor(triang, v, shading='flat', cmap='viridis', edgecolors=edgecolors)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        fig.colorbar(tc, cax=cax)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_edges(coords, edge_idx, colors=None, title="", show_index=False, lims=None, Xlims=None):
    """ Plot the edges of the mesh.
        coords.shape = (n, 2)
        edge_idx.shape = (m, 2)
        If 'color' is provided, it should be a torch tensor. In the case that
        'color' is 1D, it is assumed to be (m,) and converted to (m,1) for plotting.
    """
    # Convert inputs from torch tensors to numpy arrays.
    coords = coords.cpu().detach().numpy()
    edge_idx = edge_idx.cpu().detach().numpy()

    if colors is None:
        colors = torch.zeros(len(edge_idx))

    # If color is a 1D tensor, unsqueeze to (m, 1) and create a single subplot.
    if len(colors.shape) == 1:
        colors = colors.unsqueeze(-1)
        fig, axes = plt.subplots(1, 1, figsize=(16, 12))
        axes = [axes]
    else:
        n_plots = colors.shape[1]
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, n_plots * 4))

    colormap = plt.get_cmap("viridis")
    edge_colors = colors.cpu().detach().numpy().T  # shape = (n_plots, m)

    # Extract the coordinates of each edge. Each row in 'points' is an edge defined
    # by its two endpoints (shape: (m, 2, 2)).
    points = coords[edge_idx]

    # Plot each batch (subplot).
    for i, ax in enumerate(axes):
        ax.set_aspect('equal', adjustable='box')

        # Plot each edge using the corresponding color.
        if Xlims is None:
            xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
            ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        else:
            (xmin, xmax), (ymin, ymax) = Xlims

        edge_nums, edge_scalars = [], []
        for j, (edge, s) in enumerate(zip(points, edge_colors[i], strict=True)):
            midpoint = edge.mean(axis=0)
            if not ((xmin <= midpoint[0] <= xmax) and (ymin <= midpoint[1] <= ymax)):
                continue
            edge_scalars.append(s)
            edge_nums.append(j)

        edge_scalars = np.array(edge_scalars)
        min_c, max_c = edge_scalars.min(), edge_scalars.max()
        edge_scalars = (edge_scalars - min_c) / (max_c - min_c + 1e-9)

        for j, s in zip(edge_nums, edge_scalars, strict=True):
            c = colormap(s)
            edge = points[j]
            ax.plot(edge[:, 0], edge[:, 1], color=c)
            if show_index:
                midpoint = edge.mean(axis=0)
                ax.text(midpoint[0], midpoint[1], f"{j}", fontsize=8)
                ax.annotate(
                    '',  # No text
                    xy=(edge[1]),  # Arrow tip (end of the line)
                    xytext=midpoint,  # Arrow tail (start of the line)
                    arrowprops=dict(arrowstyle='->', lw=.5)
                )

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

        # If colors are provided, create a ScalarMappable for the colorbar.
        ax.set_title(f"{title} - Batch {i}, [min={min_c.item():.3g}, max={max_c.item():.3g}]")

        # Use the original scalar range for this batch.
        norm = plt.Normalize(vmin=min_c.item(), vmax=max_c.item())
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # Optional: attach the actual scalar array (could also use an empty array)
        cbar = fig.colorbar(sm, ax=ax)
    plt.tight_layout()
    plt.show()
