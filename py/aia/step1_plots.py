#
# Utilities to plot out details of step 1 of the analysis
#
import matplotlib.pyplot as plt
import analysis_get_data


#
#
#
def plot_exact_map(image, filepath, colorbar=False):

    plt.close('all')
    fig, ax = plt.subplots()
    ret = image.plot()

    # Get the sunspot outline
    sunspot_outline = analysis_get_data.sunspot_outline()
    _polygon, collection = analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], image.date)
    ax.add_collection(collection)
    ax.autoscale_view()
    if colorbar:
        plt.colorbar()

    print('Saving to {:s}'.format(filepath))
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')
    else:
        plt.show()
    return None


def plot_regions_hsr2015_nanoflares(image, regions, filepath):
    """

    :param image:
    :param regions:
    :param filepath:
    :return:
    """
    # Cut down the map
    image = analysis_get_data.hsr2015_map(image)

    plt.close('all')
    fig, ax = plt.subplots()
    ret = image.plot()
    # for patch in patches:
    for region in sorted(regions.keys()):
        patch = regions[region]["patch"]
        label_offset = regions[region]["label_offset"]
        ax.add_patch(patch)
        llxy = patch.get_xy()
        height = patch.get_height()
        width = patch.get_width()
        """
        plt.text(llxy[0] + width + label_offset['x'],
                 llxy[1] + height + label_offset['y'],
                 patch.get_label(),
                 bbox=dict(facecolor='w', alpha=0.5))
        """

    # Get the sunspot outline
    sunspot_outline = analysis_get_data.sunspot_outline()
    ax.add_collection(analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], image.date))
    ax.autoscale_view()

    cbar = fig.colorbar(ret, extend='both', orientation='vertical',
                        shrink=0.8, label="emission")

    cbar.ax.tick_params(labelsize=8)

    ax.autoscale_view()

    #plt.show()
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    return None