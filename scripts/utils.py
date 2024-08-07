import matplotlib.patches as mpatches

# colormap for initial and final conditions
COLORS = {
    "initial": {
        "table": "blue",
        "mb": "red",
        "kr": "green",
        "kl": "gray",
        "base": (0.859, 0.102, 0.78, 0.239),
        "pivot": (0.086, 0.902, 0.773)
    },
    # for final differentiate the colors
    "final": {
        "table": "blue",
        "mb": "red",
        "kr": "green",
        "kl": "gray",
        "base": (0.459, 0.067, 0.42, 0.239),
        "pivot": (0.051, 0.522, 0.447)
    },
}

LWS = {
    "initial": {
        "table": 4,
        "mb": 4,
        "kr": 3,
        "kl": 3,
    },
    "final": {
        "table": 2,
        "mb": 2,
        "kr": 1,
        "kl": 1,
    },
}


def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    p = mpatches.FancyArrow(
        0, 0.5 * height, width, 0, length_includes_head=True, head_width=0.75 * height
    )
    return p
