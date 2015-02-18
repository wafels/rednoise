#
# Utilities to plot out details of step 1 of the analysis
#
import numpy as np
import matplotlib.pyplot as plt


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
        height = patch.get_height()
        width = patch.get_width()
        plt.text(llxy[0] + width + label_offset['x'],
                 llxy[1] + height + label_offset['y'],
                 patch.get_label(),
                 bbox=dict(facecolor='w', alpha=0.5))
    #plt.show()
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    return None
