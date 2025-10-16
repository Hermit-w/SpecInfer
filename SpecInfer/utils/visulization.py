import numpy as np
from matplotlib.figure import Figure


def create_interactive_heatmap(
    data,
    title: str,
    cmap: str = "viridis",
    save_path: str = 'heatmap.png',
):
    """
    Create high-resolution image for detailed viewing
    """
    data_array = np.array(data)

    # Create very high resolution image
    fig = Figure(figsize=(24, 15), dpi=200)
    ax = fig.add_subplot(111)

    # Use high quality rendering
    im = ax.imshow(data_array, cmap=cmap, aspect='auto', interpolation='none')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Value')

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Column Index', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)

    # Simplify ticks
    rows, cols = data_array.shape

    x_ticks = np.arange(0, cols, max(1, cols // 15))
    y_ticks = np.arange(0, rows, max(1, rows // 15))

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    fig.tight_layout()

    # Save high resolution image
    fig.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')

