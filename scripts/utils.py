import matplotlib.patches as mpatches

def normalize(r, g, b):
    return tuple(element / 255 for element in (r, g, b))

# colormap for initial and final conditions
COLORS = {
    "initial": {
        "table": normalize(139, 128, 0),
        "mb": normalize(255, 0, 255),
        "kr": normalize(139, 0, 0),
        "kl": normalize(250, 95, 85),
        "base": (0.859, 0.102, 0.78, 0.239),
        "base_center": normalize(34, 139, 34),
        "kr_f": normalize(255, 49, 49),
        "kl_f": normalize(255, 191, 0),
        "base_f": normalize(255, 121, 0),
        "pivot": normalize(46, 139, 87)
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

# def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
#     return mpatches.FancyArrow(
#         xdescent, ydescent + height / 2, width, 0,
#         length_includes_head=True,
#         head_width=0.02 * fontsize,
#         head_length=0.02 * fontsize,
#         linewidth=orig_handle.get_linewidth(),
#         color=orig_handle.get_edgecolor()
#     )