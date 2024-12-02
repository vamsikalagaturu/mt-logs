import matplotlib.patches as mpatches

# def normalize(r, g, b):
#     return tuple(element / 255 for element in (r, g, b))

def normalize(r, g, b, a=1):
    rgb = (element / 255 for element in (r, g, b))
    return tuple(list(rgb) + [a])

# colormap for initial and final conditions
COLORS = {
    "initial": {
        "table": normalize(139, 128, 0),
        "mb": normalize(31, 207, 169),
        "kr": normalize(139, 0, 0),
        "kl": normalize(250, 95, 85),
        "base": normalize(240, 30, 255, 0.1),
        "base_center": normalize(34, 139, 34),
        "kr_f": normalize(255, 49, 49),
        "kl_f": normalize(255, 191, 0),
        "base_f": normalize(255, 121, 0),
        "pivot": normalize(46, 139, 87)
    },
    # for final differentiate the colors
    "final": {
        "table": normalize(139, 128, 0),
        "mb": normalize(31, 207, 169),
        "kr": normalize(139, 0, 0),
        "kl": normalize(250, 95, 85),
        "base": normalize(240, 30, 255, 0.8),
        "base_center": normalize(34, 139, 34),
        "kr_f": normalize(255, 49, 49),
        "kl_f": normalize(255, 191, 0, 1),
        "base_f": normalize(255, 121, 0),
        "pivot": normalize(100, 1, 0, 0.6)
    },
}

LWS = {
    "initial": {
        "table": 4,
        "mb": 2,
        "kr": 1,
        "kl": 1,
    },
    "final": {
        "table": 4,
        "mb": 4,
        "kr": 3,
        "kl": 3,
    },
}


def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    p = mpatches.FancyArrow(
        0, 0.5 * height, width, 0, length_includes_head=True, head_width=0.75 * height
    )
    return p

def make_legend_arrow2(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    print("legend", legend)
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