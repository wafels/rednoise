#
# Utilities to plot out details of step 1 of the analysis
#
import numpy as np
import matplotlib.pyplot as plt


layer_index_prop = {"label": "layer index",
                    "color": "k",
                    "linestyle": "--"}

zero_prop = {"label": None,
             "color": "k",
             "linestyle": ":"}

shift_prop = {"x": {"label": "x",
                    "color": "b",
                    "linestyle": '-'},
              "y": {"label": "y",
                    "color": "g",
                    "linestyle": '-'}}

def plot_shifts(shifts, title, layer_index,
                unit='arcsec',
                filepath=None,
                x=None,
                xlabel='mapcube layer index'):
    """
    Plot out the shifts used to move layers in a mapcube
    :param shifts:
    :param title:
    :param layer_index:
    :param unit:
    :return:
    """
    if x is None:
        xx = np.arange(0, len(shifts['x']))
    else:
        xx = x

    plt.close('all')
    for c in ['x', 'y']:
        plt.plot(shifts[c].to(unit),
                 label=shift_prop[c]['label'],
                 color=shift_prop[c]['color'],
                 linestyle=shift_prop[c]['linestyle'])
    plt.plot(xx, shifts[c].to(unit),
             label=shift_prop[c]['label'],
             color=shift_prop[c]['color'],
             linestyle=shift_prop[c]['linestyle'])

    plt.axvline(xx[layer_index],
                label=layer_index_prop['layer index'],
                color=layer_index_prop['color'],
                linestyle=layer_index_prop['linestyle'])
    plt.axhline(0.0,
                label=zero_prop['layer index'],
                color=zero_prop['color'],
                linestyle=zero_prop['linestyle'])
    plt.legend(framealpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(unit)
    plt.title(title)
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    return None


#
# Make a plot with the locations of the regions
#
def plot_regions(image, regions, filepath):
    """

    :param image:
    :param regions:
    :param filepath:
    :return:
    """
    plt.close('all')
    fig, ax = plt.subplots()
    z = image.plot()
    #for patch in patches:
    for region in sorted(regions.keys()):
        patch = regions[region]["patch"]
        label_offset = regions[region]["label_offset"]
        ax.add_patch(patch)
        llxy = patch.get_xy()
        plt.text(llxy[0] + label_offset['x'],
                 llxy[1] - label_offset['y'],
                 patch.get_label(),
                 bbox=dict(facecolor='w', alpha=0.5))
    #plt.show()
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    return None
